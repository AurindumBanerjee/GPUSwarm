"""
Validation V3 -- achievable floors of the SP PDNs over the FULL band (111 Hz - 997 MHz) and, as a sanity check, over <= 50 MHz.

For every PDN in DATASETS (default mer1, mer2, pll1v0, pll1v8, mphyvdd, ddr21, ddr79) and every band in BANDS (default full,le50):
    ga                 staged GA (tournament 3, uniform crossover 0.9, Gaussian mutation, 2 elites) on the full library
    pso_polish         staged PSO + gradient polish of EVERY stage best (exp05 Polisher: R-L-C relaxation, Adam through the exact
                       Woodbury cost, snap, exact local sweep; never worse than the PSO best)
    ga_dom             ga on the dominance-pruned library
    pso_polish_dom     pso_polish on the dominance-pruned library
Full capacitor budget (stages n = 1 .. one capacitor per pad, i.e. N-1; ddr79 is capped at V3_CAPS_DDR79 = 40 of its 78 pads because the
cost grows with k^3) and NO early stop (target 1e-12 ohm is unreachable). V3_RUNS runs per arm (default 10), V3_ITERS iterations
(default 80) x N_PARTICLES (50). Paired seeds across arms. Every score is the exact Woodbury peak on the tabulated library (placements
found on the pruned library are mapped back to library rows and re-scored on the FULL library).
The floor of a PDN = the best exact peak over all runs and arms.

Recorded per PDN and band: floor, median / spread, capacitors at the best stage, peak frequency, whether that peak lies at or above the
extrap_start_hz of the placed parts (extrapolated region), per-arm statistics, before/after plots; the <= 50 MHz floors are compared with
DecapStudy2/targets.json. Output: round2/floors_full_band.json (same layout as DecapStudy2/targets.json, full-band floors at
floors_ohm / targets_ohm, <= 50 MHz floors at floors_ohm_le50MHz) with targets at floor x {1.05, 1.1, 1.25, 1.5}; a level is flagged when it
lies below the MEDIAN peak of the best arm (inside the run-to-run spread, i.e. not reliably reachable).
Run:  bash GPUExperiments/round2/run_round2.sh --quick --only v3     |     bash GPUExperiments/round2/run_round2.sh --only v3
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import scipy.io as sio
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync
import exp05_gradient_polish as P5

EXPERIMENT = "v3_full_band_floors"
ALL = "mer1,mer2,pll1v0,pll1v8,mphyvdd,ddr21,ddr79"
DATASETS = R.env_list("DATASETS", "mer2,ddr21" if S["QUICK"] else ALL)
BANDS = R.env_list("BANDS", "full" if S["QUICK"] else "full,le50")
BAND_FMAX = {"full": 1e9, "le50": 50e6}
NUM_RUNS = int(os.environ.get("V3_RUNS", "2" if S["QUICK"] else "10"))
ITERS = int(os.environ.get("V3_ITERS", "10" if S["QUICK"] else "80"))
CAPS_DDR79 = int(os.environ.get("V3_CAPS_DDR79", "40"))
MULTS = [1.05, 1.1, 1.25, 1.5]
THR = 1e-12                                                    # unreachable -> no early stop
GA_SIGMA = float(os.environ.get("GA_SIGMA", "0.1"))
OLD_KEY = {"ddr79": "ddr_full"}                                # name in DecapStudy2/targets.json
OUT_JSON = os.environ.get("FLOORS_OUT", os.path.join(R.ROOT, "GPUExperiments", "round2", "floors_full_band.json"))
EXTRAP = sio.loadmat(R.SP_LIB)["extrap_start_hz"].ravel().astype(float)


def max_caps_for(name, prob):
    if S["QUICK"]:
        return min(6, prob.n - 1)
    return min(prob.n - 1, CAPS_DDR79) if name == "ddr79" else prob.n - 1


def dominance_keep(absz):
    """Indices of parts not dominated pointwise (|Z_b| <= |Z_a| everywhere, < somewhere); exact duplicates keep the lowest index."""
    Cn = absz.shape[0]
    dominated = torch.zeros(Cn, dtype=torch.bool, device=absz.device)
    ch = max(1, int(1e8 // (Cn * absz.shape[1])))
    for s in range(0, Cn, ch):
        a = absz[s:s + ch]
        le = (absz.unsqueeze(0) <= a.unsqueeze(1)).all(-1)
        lt = (absz.unsqueeze(0) < a.unsqueeze(1)).any(-1)
        eq = (absz.unsqueeze(0) == a.unsqueeze(1)).all(-1)
        idx = torch.arange(s, s + a.shape[0], device=absz.device).unsqueeze(1)
        dup = eq & (torch.arange(Cn, device=absz.device).unsqueeze(0) < idx)
        dominated[s:s + a.shape[0]] = ((le & lt) | dup).any(1)
    return torch.nonzero(~dominated).flatten().cpu().numpy()


def decode_cfg(prob, particle, n):
    m, p = prob.decode_particle(particle, n)
    m, p = prob.resolve_adjacent_ports(m, p)
    return m.tolist(), p.tolist()


def run_ga(prob, ev, run_id, max_caps):
    """Staged GA with the PSO's seeding / warm start / per-stage seeding, no early stop. Returns (record, stage_cfgs)."""
    NP, NI = S["N_PARTICLES"], S["N_ITERATIONS"]
    ev.reset()
    cuda_sync()
    t0 = time.perf_counter()
    n_evals, prev, stage_cfgs = 0, None, []
    for n in range(1, max_caps + 1):
        np.random.seed(S["BASE_SEED"] + run_id * 10000 + n)
        ev.begin_stage(n)
        pop = C.seed_particles(n, prev)
        dim = 2 * n
        gbest, gpart = np.inf, None

        def score(X, bound):
            nonlocal n_evals, gbest, gpart
            M, Pp = prob.decode_swarm(X, n)
            c = ev.evaluate(M, Pp, bound)
            for i in range(len(X)):
                n_evals += 1
                if c[i] < gbest:
                    gbest, gpart = c[i], X[i].copy()
            return c

        val = score(pop, np.full(NP, np.inf))
        for it in range(1, NI):
            order = np.argsort(val)
            elite = pop[order[:2]].copy()
            kids = []
            while len(kids) < NP - 2:
                def tour():
                    c = np.random.randint(0, NP, 3)
                    return pop[c[np.argmin(val[c])]]
                a, b = tour(), tour()
                if np.random.rand() < 0.9:
                    a = np.where(np.random.rand(dim) < 0.5, a, b)
                mu = np.random.rand(dim) < 1.0 / dim
                kids.append(np.clip(a + mu * np.random.normal(0, GA_SIGMA, dim), 0, 1))
            kids = np.array(kids)
            kv = score(kids, np.full(len(kids), np.inf))
            pop = np.vstack([elite, kids])
            val = np.concatenate([val[order[:2]], kv])
        if gpart is None:
            prev = None
            continue
        prev = gpart
        m, p = decode_cfg(prob, gpart, n)
        stage_cfgs.append((n, float(gbest), m, p))
    cuda_sync()
    rec = {"total_time_s": time.perf_counter() - t0, "n_evals": n_evals}
    rec.update(ev.stats())
    return rec, stage_cfgs


def run_psopol(prob, ev, pol, run_id, polish):
    """C.run_pso with a stage-end hook that (optionally) polishes and always records the stage placement."""
    cfgs = []

    def hook(n_caps, gbest, gpart, prev_best):
        out = None
        cost, part = gbest, gpart
        if polish:
            r = P5.make_hook(prob, pol, 1e30)(n_caps, gbest, gpart, prev_best)      # ratio check passes for every stage (thr huge)
            if r is not None and r[0] < gbest:
                cost, part, out = r[0], r[1], r
        m, p = decode_cfg(prob, part, n_caps)
        cfgs.append((n_caps, float(cost), m, p))
        return out
    rec = C.run_pso(prob, ev, [THR], run_id, label="pso", on_stage_end=hook)
    return rec, cfgs


def best_of(prob_full, keep, stage_cfgs, model_ids):
    """Best stage (by recorded cost) -> exact re-score on the FULL library + peak frequency + extrapolation flags."""
    i = int(np.argmin([c for _, c, _, _ in stage_cfgs]))
    n, cost, m, p = stage_cfgs[i]
    mfull = [int(keep[x]) for x in m] if keep is not None else list(m)
    pk, am = prob_full.wb_peak(torch.tensor([mfull], device=DEVICE), torch.tensor([p], device=DEVICE))
    peak, f = float(pk[0]), float(prob_full.freq[int(am[0])])
    ext = EXTRAP[model_ids[mfull]]
    return {"n_caps_best": n, "stage_cost": cost, "peak_exact": peak, "rescore_dev": abs(peak - cost) / max(cost, 1e-30),
            "peak_f_Hz": f, "models": mfull, "ports": p, "extrap_min_Hz": float(ext.min()), "extrap_max_Hz": float(ext.max()),
            "peak_above_all_extrap_start": bool(f >= ext.max()), "peak_above_any_extrap_start": bool(f >= ext.min()),
            "frac_parts_extrapolated_at_peak": float((ext <= f).mean()), "n_stages": len(stage_cfgs)}


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASETS": DATASETS, "BANDS": BANDS, "V3_RUNS": NUM_RUNS, "V3_ITERS": ITERS,
                                       "V3_CAPS_DDR79": CAPS_DDR79, "LIB_SOURCE": R.LIB_SOURCE, "THR": THR, "MULTS": MULTS})
    C.init_device_log(run)
    S["N_ITERATIONS"] = ITERS
    recs = []
    best_place = {}                                                 # (dataset, band) -> (peak, record, prob_full)
    for name in DATASETS:
        for band in BANDS:
            prob, info = R.load_problem(name, fmax=BAND_FMAX[band])
            S["MAX_CAPS"] = max_caps_for(name, prob)
            run.lg.info("STAGE | %s %s | %s | max_caps=%d iters=%d", name, band, info, S["MAX_CAPS"], ITERS)
            absz = 1.0 / prob.D.abs()
            keep = dominance_keep(absz)
            del absz
            p_dom = C.Problem(y=prob.y_np, d=prob.d_np[keep], freq=prob.freq, label=f"{prob.label}/dom")
            run.lg.info("STAGE | %s %s | dominance-pruned library: %d of %d parts", name, band, len(keep), prob.C)
            ev_f, ev_d = C.SwarmEvaluator(prob, True, True), C.SwarmEvaluator(p_dom, True, True)
            pol_f, pol_d = P5.Polisher(prob), P5.Polisher(p_dom)
            C.warmup(prob, [ev_f])
            C.warmup(p_dom, [ev_d])
            P5.POLISH_RATIO = 1e9
            arms = {"ga": (prob, ev_f, None, None, "ga"), "pso_polish": (prob, ev_f, pol_f, None, "pso"),
                    "ga_dom": (p_dom, ev_d, None, keep, "ga"), "pso_polish_dom": (p_dom, ev_d, pol_d, keep, "pso")}
            for arm, (pr, ev, pol, kp, kind) in arms.items():
                for r in range(1, NUM_RUNS + 1):
                    run.lg.info("RUN_START | dataset=%s band=%s method=%s run=%d", name, band, arm, r)
                    try:
                        if kind == "ga":
                            rec, cfgs = run_ga(pr, ev, r, S["MAX_CAPS"])
                        else:
                            rec, cfgs = run_psopol(pr, ev, pol, r, polish=True)
                        info_b = best_of(prob, kp, cfgs, prob.model_ids)
                    except Exception as e:                                   # noqa: BLE001
                        run.fail(f"{name}/{band}/{arm}", THR, r, e)
                        continue
                    row = {"method": f"{name}/{band}/{arm}", "dataset": name, "band": band, "arm": arm, "run": r, "threshold": THR,
                           "success": False, "time_s": -1.0, "total_time_s": rec["total_time_s"], "caps_at_target": None,
                           "final_n_caps": info_b["n_stages"], "best_minz": info_b["peak_exact"], "n_evals": rec["n_evals"],
                           "best_models": info_b["models"], "best_ports": info_b["ports"], "max_caps": S["MAX_CAPS"],
                           "lib_size": int(len(kp)) if kp is not None else prob.C, **{k: v for k, v in info_b.items()
                                                                                       if k not in ("models", "ports")}}
                    run.add(row)
                    recs.append(row)
                    run.lg.info("RESULT | %s | run=%d | peak=%.6g | caps=%d | f=%.4g MHz | extrap_any=%s | time=%.1fs",
                                row["method"], r, row["peak_exact"], row["n_caps_best"], row["peak_f_Hz"] / 1e6,
                                row["peak_above_any_extrap_start"], row["total_time_s"])
                    key = (name, band)
                    if key not in best_place or row["peak_exact"] < best_place[key][0]:
                        best_place[key] = (row["peak_exact"], row, prob)
            del ev_f, ev_d, pol_f, pol_d, p_dom
            torch.cuda.empty_cache()
        # per-PDN before/after plots (band by band, best placement of any arm)
        for band in BANDS:
            if (name, band) in best_place:
                pk, row, pr = best_place[(name, band)]
                R.impedance_before_after(pr, [(row["best_models"], row["best_ports"])], None,
                                         os.path.join(run.run_dir, "plots", f"before_after_{name}_{band}.png"),
                                         title=f"{name} ({band}): best of all arms = {row['arm']}, {row['n_caps_best']} caps, "
                                               f"peak {pk:.4g} ohm @ {row['peak_f_Hz'] / 1e6:.4g} MHz")
    summarise(run, recs)
    run.finish()


def summarise(run, recs):
    import pandas as pd
    df = pd.DataFrame(recs)
    old = json.load(open(os.path.join(R.ROOT, "DecapStudy2", "targets.json")))
    lines, floors = [], {"full": {}, "le50": {}}
    per_key = {"full": {}, "le50": {}}
    rows = []
    for (name, band), g in df.groupby(["dataset", "band"]):
        flo = float(g.peak_exact.min())
        b = g.loc[g.peak_exact.idxmin()]
        arm_stats = {a: {"min": float(x.peak_exact.min()), "median": float(x.peak_exact.median()), "max": float(x.peak_exact.max()),
                         "caps_median": float(x.n_caps_best.median()), "time_mean_s": float(x.total_time_s.mean())}
                     for a, x in g.groupby("arm")}
        best_arm = min(arm_stats, key=lambda a: arm_stats[a]["median"])
        med = arm_stats[best_arm]["median"]
        floors[band][name] = flo
        per_key[band][name] = {"n_runs": int(len(g)), "min": flo, "median_best_arm": med, "best_arm_by_median": best_arm,
                               "median_all_runs": float(g.peak_exact.median()), "max": float(g.peak_exact.max()),
                               "spread_pct_best_arm": 100 * (arm_stats[best_arm]["max"] - arm_stats[best_arm]["min"]) / flo,
                               "arms": arm_stats, "floor_arm": b.arm, "caps_at_floor": int(b.n_caps_best), "peak_f_Hz": float(b.peak_f_Hz),
                               "peak_above_any_extrap_start": bool(b.peak_above_any_extrap_start),
                               "peak_above_all_extrap_start": bool(b.peak_above_all_extrap_start),
                               "frac_parts_extrapolated_at_peak": float(b.frac_parts_extrapolated_at_peak),
                               "max_caps_allowed": int(b.max_caps), "rescore_dev_max": float(g.rescore_dev.max())}
        o = old["floors_ohm"].get(OLD_KEY.get(name, name))
        lines.append(f"{name:8s} {band:4s}: floor {flo:.5g} ohm ({b.arm}, {int(b.n_caps_best)}/{int(b.max_caps)} caps) @ {b.peak_f_Hz / 1e6:.4g} MHz | "
                     f"best-arm median {med:.5g} ({best_arm}) | spread {per_key[band][name]['spread_pct_best_arm']:.1f}% | "
                     f"peak >= extrap_start of any/all placed parts: {bool(b.peak_above_any_extrap_start)}/{bool(b.peak_above_all_extrap_start)} "
                     f"({100 * b.frac_parts_extrapolated_at_peak:.0f}% of parts) | rescore dev {g.rescore_dev.max():.1e}"
                     + (f" | DecapStudy2 <=50MHz floor {o:.5g} -> ratio {flo / o:.3f}" if (band == "le50" and o) else ""))
        for a, st in arm_stats.items():
            lines.append(f"    {a:15s} min {st['min']:.5g} median {st['median']:.5g} max {st['max']:.5g} | caps median {st['caps_median']:.0f} | "
                         f"{st['time_mean_s']:.1f}s/run")
            rows.append({"dataset": name, "band": band, "arm": a, **st})
    pd.DataFrame(rows).to_csv(os.path.join(run.run_dir, "summary", "floors_by_arm.csv"), index=False)
    # ---- json in the layout of DecapStudy2/targets.json
    targets, flags = {}, {}
    for name, flo in floors["full"].items():
        med = per_key["full"][name]["median_best_arm"]
        targets[name] = {str(m): round(m * flo, 9) for m in MULTS}
        flags[name] = {str(m): bool(m * flo < med) for m in MULTS}
    out = {"_comment": "Written by validation/v3_full_band_floors.py. floors_ohm[key] = best exact peak |Z_kk| over the FULL band (111 Hz-997 MHz, "
                       "2588 pts, combined shunt library incl. extrapolated parts above extrap_start_hz); floors_ohm_le50MHz = same procedure "
                       "limited to <= 50 MHz. targets_ohm[key][level] = floors_ohm[key] * level.",
           "multipliers": MULTS, "floors_ohm": floors["full"], "floors_ohm_le50MHz": floors["le50"], "targets_ohm": targets,
           "targets_ohm_le50MHz": {n: {str(m): round(m * f, 9) for m in MULTS} for n, f in floors["le50"].items()},
           "flags": flags,
           "flag_rule": "flags[key][level] is true when targets_ohm[key][level] < the MEDIAN peak of the best arm (lowest median over arms) -- the level "
                        "lies inside that arm's run-to-run spread, so it is not reliably reachable",
           "source": {"arms": ["ga", "pso_polish", "ga_dom", "pso_polish_dom"], "runs_per_arm": NUM_RUNS, "n_particles": S["N_PARTICLES"],
                      "n_iterations": ITERS, "warm_start": [S["WARM_START_FRACTION"], S["WARM_START_JITTER"]], "no_early_stop": True,
                      "ddr79_max_caps": CAPS_DDR79, "per_key_full": per_key["full"], "per_key_le50MHz": per_key["le50"],
                      "run_id": os.environ.get("RUN_ID"), "quick": S["QUICK"]}}
    json.dump(out, open(os.path.join(run.run_dir, "summary", "floors_full_band.json"), "w"), indent=2)
    if not S["QUICK"]:
        json.dump(out, open(OUT_JSON, "w"), indent=2)
        lines.append(f"wrote {OUT_JSON}")
    for n, fl in flags.items():
        bad = [m for m, v in fl.items() if v]
        if bad:
            lines.append(f"FLAG {n}: levels {bad} lie below the median achieved peak")
    R.write_lines(run, "floors_summary.txt", lines)
    run.summarise(group=("dataset", "band", "arm"), extra_cols=("peak_exact", "n_caps_best", "peak_f_Hz", "frac_parts_extrapolated_at_peak"))


if __name__ == "__main__":
    main()
