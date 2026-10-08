"""
Experiment 2 runners. Every finished run is appended to results_long.jsonl at
once (one JSON object per line, all factor levels as fields), and every runner
skips keys already present -- so a crashed or interrupted study resumes where
it stopped.
"""
import os
import json
import time
import numpy as np

import study_config as C
import study_core as S

RESULTS = os.path.join(C.OUT_ROOT, "results_long.jsonl")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def append_record(rec):
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    rec["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec, default=_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_records(record_type=None):
    out = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if record_type is None or r.get("record_type") == record_type:
                        out.append(r)
    return out


def _key(r):
    return (r["stage"], r["scope"], r["band"], r["run_id"], r["rail"],
            r.get("method", "numpy"))


def _done():
    return {_key(r) for r in load_records("run")}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "study.log"), "a") as fh:
        fh.write(line + "\n")


# ============================================================ Grid 1

def _run_and_log(stage, scope, band, run_id, prob, rail, cfg, extra=None):
    res = S.run_pso(prob, band, run_id, cfg)
    rec = {
        "record_type": "run", "stage": stage, "scope": scope, "band": band,
        "run_id": run_id, "rail": rail, "method": cfg.get("method", "numpy"),
        "mode": cfg.get("mode", "batched"), "base_seed": C.BASE_SEED,
        "n_particles": cfg["n_particles"], "n_iters": cfg["n_iters"],
        "N": prob.N, "n_obs": prob.n_obs, "n_pads": len(prob.pads),
        "rails": prob.rails, "target_keys": prob.target_keys,
        "targets_ohm": prob.targets.tolist(),       # tightest level; all levels are in "levels"
        "multipliers": [m for _, m, _ in prob.levels],
    }
    rec.update(extra or {})
    rec.update(res)
    append_record(rec)
    return rec


def run_grid1(stage, cells, overrides=None):
    """cells: iterable of (scope, band). Sequential; resumable."""
    scfg = dict(C.STAGES[stage])
    scfg.update(overrides or {})
    runs = scfg.pop("runs")
    done = _done()
    cells = list(cells)
    total = len(cells) * runs
    log(f"=== Grid 1 / stage {stage}: {len(cells)} cells x {runs} runs "
        f"({scfg}) ===")
    n = 0
    for scope, band in cells:
        probs = S.scope_problems(scope)
        for run_id in range(1, runs + 1):
            n += 1
            for prob in probs:
                rail = prob.rails[0] if scope == "A1" else "all"
                if (stage, scope, band, run_id, rail, "numpy") in done:
                    continue
                t = time.perf_counter()
                rec = _run_and_log(stage, scope, band, run_id, prob, rail, dict(scfg))
                hit = ",".join(f"x{lab}" for lab, v in rec["levels"].items() if v["hit"]) or "-"
                log(f"[{n}/{total}] {scope} {band} run{run_id} {rail}: "
                    f"levels_hit={hit} caps={rec['caps_used']} "
                    f"own={rec['best_cost_own_band']:.4g}{rec['cost_unit']} "
                    f"care({C.CARE_BAND})={rec['rescored'][C.CARE_BAND]['ratio']:.3f} "
                    f"evals={rec['n_evals']} {time.perf_counter() - t:.0f}s")


def all_cells():
    return [(s, b) for s in C.SCOPES for b in C.BANDS]


# ============================================================ Grid 2

def fixed_configs(prob, k, seed):
    """k fixed (model, pad) configurations, identical for every method at a size; the number of
    capacitors cycles through 1..min(FIXED_EVAL_CAPS, pads)."""
    rng = np.random.default_rng(seed)
    top = min(C.FIXED_EVAL_CAPS, prob.max_caps)
    out = []
    for i in range(k):
        n_caps = 1 + (i % top)
        pads = rng.choice(prob.pads, size=n_caps, replace=False)
        models = rng.integers(0, S.decaps().shape[0], size=n_caps)
        out.append([(int(m), int(p)) for m, p in zip(models, pads)])
    return out


def time_fixed_evals(band):
    """PRIMARY Grid 2 metric: wall-time for a FIXED evaluation count. Every method evaluates the
    same FIXED_EVALS (model, pad) configurations in per-frequency loop mode; the cost agreement
    with numpy is recorded alongside. `iterative` is an approximation: timing only, plus its
    series fallback rate; its cost deviation is recorded but never used as a quality figure."""
    for N, key in C.GRID2_SIZES.items():
        prob = S.build_problem(key)
        bs = S.band_spec(prob, band)
        cfgs = fixed_configs(prob, C.FIXED_EVALS, C.FIXED_EVAL_SEED + N)
        existing = {r["method"]: r for r in load_records("eval_timing") if r["size_N"] == N}
        methods = [m for m in C.GRID2_TIMING_METHODS
                   if not (m == "pure_python" and N not in C.GRID2_PURE_SIZES)]
        ref = np.array(existing["numpy"]["costs"]) if "numpy" in existing else None
        for method in ["numpy"] + [m for m in methods if m != "numpy"]:
            if method in existing or method not in methods:
                continue
            S.evaluate(prob, bs, cfgs[0][:1], method, "loop")            # warm-up
            t = time.perf_counter()
            costs = np.array([S.evaluate(prob, bs, c, method, "loop")[0] for c in cfgs])
            total = time.perf_counter() - t
            if method == "numpy":
                ref = costs
            ok = (costs < 1e199) & (ref < 1e199)
            rel = np.abs(costs[ok] - ref[ok]) / np.abs(ref[ok])
            rec = {"record_type": "eval_timing", "band": band, "size_N": N, "problem": prob.name,
                   "method": method, "approximation": method in C.APPROXIMATE_METHODS,
                   "n_evals": len(cfgs), "total_s": total, "sec_per_eval": total / len(cfgs),
                   "nf": bs[0], "caps_min": 1, "caps_max": min(C.FIXED_EVAL_CAPS, prob.max_caps),
                   "max_rel_cost_dev_vs_numpy": float(rel.max()) if rel.size else None,
                   "median_rel_cost_dev_vs_numpy": float(np.median(rel)) if rel.size else None,
                   "n_failed": int((costs >= 1e199).sum())}
            if method == "numpy":
                rec["costs"] = costs.tolist()
            if method in C.APPROXIMATE_METHODS:
                rec["series"] = S.iterative_fallback_rate(prob, bs, cfgs[:C.FALLBACK_PROBE_CONFIGS])
            append_record(rec)
            extra = (f", series fallback {rec['series']['fallback_rate']:.1%}"
                     if "series" in rec else "")
            log(f"fixed-eval timing N={N} {method}{' (approximation)' if rec['approximation'] else ''}: "
                f"{total:.1f} s for {len(cfgs)} evals = {1e3 * total / len(cfgs):.1f} ms/eval "
                f"(max cost dev vs numpy {rec['max_rel_cost_dev_vs_numpy']:.1e}{extra})")


def run_grid2(band, overrides=None):
    """Cost study at the winning band.

    1. fixed-evaluation timing, all five methods (iterative = approximation, timing only)
    2. PSO runs for the EXACT methods only, stopping at the tightest NOT noise-flagged level,
       under the unchanged wall-clock cap: secondary time-to-target, and placement matching
       against numpy. pure_python runs only at GRID2_PURE_SIZES. If the first numpy run at a size
       hits the cap without reaching the stop level, the rest of that size's PSO runs are skipped
       (time-to-target and placement matching are moot there) and a grid2_skip record is written.
    """
    pso = dict(C.GRID2_PSO)
    pso.update(overrides or {})
    pso["mode"] = "loop"
    pso["time_limit_s"] = pso.get("time_limit_s", C.GRID2_TIME_LIMIT_S)
    pso["stop_level"] = C.GRID2_STOP_LEVEL
    log(f"=== Grid 2 at band {band} ({pso}) ===")
    time_fixed_evals(band)
    done = _done()
    skipped = {r["size_N"] for r in load_records("grid2_skip")}
    order = ["numpy"] + [m for m in C.GRID2_PSO_METHODS if m != "numpy"]

    def numpy_run1(N):
        return next((r for r in load_records("run") if r.get("grid") == "G2" and r.get("size_N") == N
                     and r["method"] == "numpy" and r["run_id"] == 1), None)

    for N, key in C.GRID2_SIZES.items():
        prob = S.build_problem(key)
        stop_lab = prob.stop_label(C.GRID2_STOP_LEVEL)
        log(f"Grid 2 N={N} ({prob.name}): PSO stop level x{stop_lab}"
            + (" (flagged: every level is inside the floor spread)" if prob.level_flagged[stop_lab] else ""))
        for method in order:
            if method == "pure_python" and N not in C.GRID2_PURE_SIZES:
                continue
            runs = C.GRID2_PURE_RUNS if method == "pure_python" else C.GRID2_RUNS
            for run_id in range(1, runs + 1):
                if N in skipped:
                    break
                if (3, f"G2_N{N}", band, run_id, "all", method) not in done:
                    cfg = dict(pso, method=method)
                    t = time.perf_counter()
                    rec = _run_and_log(3, f"G2_N{N}", band, run_id, prob, "all", cfg,
                                       extra={"grid": "G2", "size_N": N})
                    log(f"G2 N={N} {method} run{run_id}: reached x{stop_lab}={rec['success']} "
                        f"timeout={rec['timed_out']} caps={rec['caps_used']} "
                        f"evals={rec['n_evals']} {time.perf_counter() - t:.0f}s")
                if method == "numpy" and run_id == 1 and C.GRID2_SKIP_UNCONVERGED:
                    r1 = numpy_run1(N)
                    if r1 and r1["timed_out"] and not r1["success"]:
                        append_record({"record_type": "grid2_skip", "size_N": N, "problem": prob.name,
                                       "band": band, "stop_level": stop_lab,
                                       "reason": "first numpy run hit the wall-clock cap without "
                                                 "reaching the stop level; remaining PSO runs skipped"})
                        skipped.add(N)
                        log(f"G2 N={N}: numpy run1 hit the {pso['time_limit_s']} s cap without reaching "
                            f"x{stop_lab} -> remaining PSO runs at this size skipped "
                            f"(fixed-evaluation timing stands)")
            if N in skipped:
                break


# ============================================================ rail responsiveness

def rail_response(stride=7):
    """How much can capacitors move each rail at all? Bare peak vs the best of
    'one library model on EVERY pad' (every `stride`-th model), per band. This is
    the evidence for mer1 / pll1v8 barely responding (motivates scope A3)."""
    path = os.path.join(C.OUT_ROOT, "rail_response.json")
    if os.path.exists(path):
        return json.load(open(path))
    d = S.decaps()
    out = {}
    keys = [f"mphy_{r[0]}" for r in C.MPHY_RAILS] + ["ddr21"]
    for key in keys:
        prob = S.build_problem(key)
        nfs = {b: S.band_spec(prob, b) for b in C.BANDS}
        bare = np.abs(S.diag_z(prob, len(S.freqs()), []))[:, 0]
        best = {b: np.inf for b in C.BANDS}
        for m in range(0, d.shape[0], stride):
            z = np.abs(S.diag_z(prob, len(S.freqs()), [(m, int(p)) for p in prob.pads]))
            for b, (nf, W) in nfs.items():
                zz = z[:nf] * W[:nf] if W is not None else z[:nf]
                best[b] = min(best[b], float(zz[:, 0].max()))
        row = {}
        for b, (nf, W) in nfs.items():
            zb = bare[:nf] * W[:nf, 0] if W is not None else bare[:nf]
            bp = float(zb.max())
            row[b] = {"bare_peak_ohm": bp, "all_pads_filled_ohm": best[b],
                      "gain_pct": 100.0 * (1 - best[b] / bp)}
        out[prob.rails[0] if key != "ddr21" else "ddr21"] = row
        log(f"rail response {key}: " + ", ".join(
            f"{b} {row[b]['gain_pct']:.1f}%" for b in C.BANDS))
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"models_sampled_stride": stride, "rails": out}, fh, indent=2)
    return out


# ============================================================ consistency

def _best_rail_runs(band):
    """Best (lowest own-band peak, ohm) A1 run per rail; prefer stage 2, else stage 1."""
    recs = [r for r in load_records("run") if r["scope"] == "A1" and r["band"] == band]
    best = {}
    for r in recs:
        k = r["rail"]
        rank = (0 if r["stage"] == 2 else 1, r["best_cost_own_band"])
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return {k: v[1] for k, v in best.items()}


def consistency_check(band):
    """Install A1's five rail solutions together in y_sp_mphy_full and compare
    every rail's Z_kk(f) with the isolated (rail-file) result."""
    best = _best_rail_runs(band)
    if len(best) < len(C.MPHY_RAILS):
        log(f"consistency[{band}]: A1 records missing for "
            f"{set(r[0] for r in C.MPHY_RAILS) - set(best)}; skipped")
        return None
    full = S.build_problem("mphy_full")
    nfull = len(S.freqs())

    def to_full(placement):                # records carry pad_global = index in y_sp_mphy_full
        return [(int(c["model"]), int(c["pad_global"])) for c in placement]

    combined = [c for r in C.MPHY_RAILS for c in to_full(best[r[0]]["placement"])]
    assert all(p not in set(full.obs.tolist()) for _, p in combined), "cap on observation port"
    z_comb = S.diag_z(full, nfull, combined)                     # (F, 5)

    rows, worst_all, worst_alone = [], 0.0, 0.0
    for k, r in enumerate(C.MPHY_RAILS):
        name = r[0]
        rp = S.build_problem(f"mphy_{name}")
        cfg_iso = [(int(c["model"]), int(c["pad"])) for c in best[name]["placement"]]
        z_iso = S.diag_z(rp, nfull, cfg_iso)[:, 0]
        z_alone = S.diag_z(full, nfull, to_full(best[name]["placement"]))[:, k]
        z_c = z_comb[:, k]
        rel = lambda a: float(np.max(np.abs(a - z_iso) / np.abs(z_iso)))
        pk = lambda z: float(np.abs(z).max())
        rows.append({
            "rail": name, "caps": len(cfg_iso),
            "peak_isolated_ohm": pk(z_iso), "peak_alone_in_full_ohm": pk(z_alone),
            "peak_combined_ohm": pk(z_c),
            "max_rel_dev_alone": rel(z_alone), "max_rel_dev_combined": rel(z_c),
            "peak_rel_dev_combined": abs(pk(z_c) - pk(z_iso)) / pk(z_iso),
        })
        worst_alone = max(worst_alone, rows[-1]["max_rel_dev_alone"])
        worst_all = max(worst_all, rows[-1]["peak_rel_dev_combined"])
    ok = worst_all <= C.CONSISTENCY_TOL
    out = {
        "band": band, "tolerance_peak_rel_dev": C.CONSISTENCY_TOL,
        "reduction_check_max_rel_dev": worst_alone,
        "worst_peak_rel_dev_combined": worst_all, "rails": rows,
        "decomposition_valid": bool(ok),
        "verdict": ("A1 decomposition validated: combined-network rail impedances match "
                    "the isolated results." if ok else
                    "MISMATCH: A2 (whole package) is the correct formulation; "
                    "A1 is an approximation."),
    }
    with open(os.path.join(C.OUT_ROOT, f"consistency_{band}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"consistency[{band}]: reduction check {worst_alone:.2e}, combined peak dev "
        f"{worst_all:.2e} -> {out['verdict']}")
    return out
