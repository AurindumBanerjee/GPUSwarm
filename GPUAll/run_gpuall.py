#!/usr/bin/env python3
"""
GPUAll driver -- the AllCPU study (4 PDNs over their own frequency ranges, bands x levels x runs, 5 swarm configurations) on ONE GPU,
as a comparative study of separate arms:

  exact arms (identical trajectories to AllCPU's CPU numpy; measure SPEED)       gpu_numpy | woodbury_batch | fast
  search arms (different search, exact scores; measure SOLUTION QUALITY)        pso_polish | ga | ga_polish
  (greedy_oracle and the cost model are separate studies: ../GPUAll/greedy_oracle, ../GPUAll/cost_model)

   python run_gpuall.py --dry                                         plan
   python run_gpuall.py --floor [--pdns pdn3,pdn4] [--bands ...]      floors of PDN 3/4 for the active configuration (AllCPU's floor_pass rule,
                                                                      on the GPU: 50x15, 3 runs, patience 6) -> targets_gpuall.json
   python run_gpuall.py                                               the study (resumable: finished runs are skipped)
   python run_gpuall.py --pdns pdn1,pdn2 --swarms 50x15,100x40 --arms fast,ga,pso_polish --bands B50,B1G --levels T1,T2 --runs 4
Data configuration as AllCPU: ALLCPU_GRID (5g default) / ALLCPU_LIB (murata default). Targets: GPUAll/targets_gpuall.json (copied from
AllCPU/common/targets_allcpu.json on first use, floors filled by --floor); override with ALLCPU_TARGETS.
Output: out_<cfg>/<pdn>/<swarm>/<arm>/results_long.jsonl  and  .../<band>/<level>/run_<k>/ (log, curve, convergence, impedance plots).
"""
import os
import sys
import json
import time
import shutil
import argparse
import statistics

H = os.path.dirname(os.path.abspath(__file__))
ROOTG = os.path.dirname(H)
os.environ.setdefault("GPUSWARM_ROOT", ROOTG)
COMMON = os.path.join(ROOTG, "AllCPU", "common")
TGT = os.environ.setdefault("ALLCPU_TARGETS", os.path.join(H, "targets_gpuall.json"))
if not os.path.exists(TGT):
    shutil.copy(os.path.join(COMMON, "targets_allcpu.json"), TGT)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("WARM_START_PCT", "20")
sys.path.insert(0, COMMON)
sys.path.insert(0, H)

import numpy as np                                              # noqa: E402
import allcpu_pdns as R                                         # noqa: E402
import gpuall_eval as G                                         # noqa: E402
import gpuall_search as S                                       # noqa: E402

# Limit on the max relative deviation of the GPU arms vs AllCPU numpy. PDN 2's admittance matrix is ill-conditioned (cond up to 1.3e11, two ports
# 4e-7 ohm apart), so double precision alone gives 1e-7-level differences between CPU LAPACK and cuSOLVER: a looser limit, see PDN2_NOTES.md.
VERIFY_TOL = {"default": 1e-8, "pdn2": 1e-6}

EXACT_ARMS = ["gpu_numpy", "woodbury_batch", "fast"]
ALL_ARMS = EXACT_ARMS + list(S.SEARCH_ARMS)

ap = argparse.ArgumentParser()
ap.add_argument("--pdns", default=",".join(R.PDN_KEYS))
ap.add_argument("--swarms", default=",".join(R.SWARMS))
ap.add_argument("--arms", default=",".join(ALL_ARMS))
ap.add_argument("--numpy-swarms", default=os.environ.get("NUMPY_SWARMS", "50x15,100x40"),
                help="swarms on which the slow gpu_numpy reference arm runs (speed-up reference)")
ap.add_argument("--bands"); ap.add_argument("--levels")
ap.add_argument("--runs", type=int, default=int(os.environ.get("NUM_RUNS", 4)))
ap.add_argument("--out", default=os.path.join(H, "out_" + R.CFG))
ap.add_argument("--dry", action="store_true"); ap.add_argument("--floor", action="store_true")
ap.add_argument("--floor-swarm", default="50x15"); ap.add_argument("--floor-runs", type=int, default=3)
a = ap.parse_args()


def log(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


def prep_cell(key, P=50, I=15, load_problem=True):
    cell = R.Cell(key, P, I, load_problem=load_problem)
    core = cell.core
    core._mask_cache.clear()
    core.PATIENCE = int(os.environ.get("PATIENCE", "6")) if key in ("pdn3", "pdn4") else None
    core.TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", "28800"))
    return cell


# ============================================================ floors (PDN 3 / PDN 4)
def floor_pass():
    P, I = (int(x) for x in a.floor_swarm.split("x"))
    T = R.targets()
    for key in [k for k in a.pdns.split(",") if k in ("pdn3", "pdn4")]:
        cell = prep_cell(key, P, I, load_problem=False)
        core = cell.core
        core.N_PARTICLES, core.N_ITERATIONS = P, I
        bands = a.bands.split(",") if a.bands else R.SPEC[key]["opt_bands"]
        targets = R.MPHY_RAILS if key == "pdn3" else [None]
        for tgt in targets:
            path = R.RAIL_FILES[tgt] if tgt else R.SPEC[key]["path"]
            prob = core.Problem(f"floor_{key}_{tgt or 'full'}", path, n_obs=1)
            ev = G.register(core, prob)
            core.OBJ_WEIGHTS = None
            for b in bands:
                core.RANK_BAND = b
                vals = []
                for k in range(1, a.floor_runs + 1):
                    d = os.path.join(H, "floor_pass", R.CFG, key, tgt or "full", b, f"run{k}")
                    res = os.path.join(d, "results_long.jsonl")
                    if os.path.exists(res) and os.path.getsize(res):
                        vals.append(json.loads(open(res).readline())["best_cost"])
                        continue
                    r = core.run_pso(prob, "woodbury_batch", k, os.path.join(d, "run"), res, pool=G.GpuPool(ev, "woodbury_batch"),
                                     out_root=d, threshold=None, level="floor")
                    vals.append(r["best_cost"])
                    log(f"FLOOR {key} {tgt or 'ddr_full'} {b} run {k}: {r['best_cost']:.6g} @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s")
                spread = (max(vals) - min(vals)) / min(vals)
                fl = statistics.median(vals) if spread > 0.10 else min(vals)
                sec = T[key][R.CFG][b]
                if key == "pdn3":
                    sec.setdefault("floors_ohm", {})[tgt] = fl
                else:
                    sec["floor_ohm"] = fl
                sec["floor_source"] = f"GPUAll run_gpuall.py --floor swarm {a.floor_swarm} runs {a.floor_runs} (woodbury_batch = numpy trajectory)"
                json.dump(T, open(TGT, "w"), indent=1)
                log(f"floor {key} {tgt or 'ddr_full'} {b} = {fl:.6g} (spread {100 * spread:.1f} %)")
    log("floors written to", TGT)


# ============================================================ study
def study():
    pdns, swarms, arms = a.pdns.split(","), a.swarms.split(","), a.arms.split(",")
    for x in arms:
        assert x in ALL_ARMS, f"unknown arm {x}; choose from {ALL_ARMS}"
    miss = R.targets_status(pdns, a.bands.split(",") if a.bands else None)
    if miss:
        print("targets not ready -- run  python run_gpuall.py --floor  first:", ", ".join(miss))
        sys.exit(2)
    numpy_sw = set(a.numpy_swarms.split(","))
    for key in pdns:
        cell = prep_cell(key)
        core, prob = cell.core, cell.prob
        ev = G.register(core, prob)
        bands = a.bands.split(",") if a.bands else R.SPEC[key]["opt_bands"]
        worst = G.verify(core, prob, ev)
        log(f"{R.SPEC[key]['label']}: N={prob.N} obs={prob.n_obs} pads={len(prob.pads)} F={prob.F} models={ev.C} device={G.DEV} | "
            f"verify vs CPU numpy (max rel, all bands): " + ", ".join(f"{m} {v:.1e}" for m, v in worst.items()))
        tol = VERIFY_TOL.get(key, VERIFY_TOL["default"])
        assert max(worst.values()) < tol, f"GPU evaluation disagrees with AllCPU numpy (limit {tol:g}; see PDN2_NOTES.md)"
        pol = S.Polisher(ev) if any(x.endswith("polish") for x in arms) else None
        if pol is not None:
            log(f"   R-L-C fit of the library: median relative |Z| error {float(pol.fit_err.median()):.3g}")
            cell.select_band(bands[0])                                       # warm up autograd / cuSOLVER kernels (untimed)
            pol.polish([0, 1][:min(2, len(prob.pads))], [int(x) for x in prob.pads[:min(2, len(prob.pads))]])
            pol.log.clear()
        for sw in swarms:
            P, I = (int(x) for x in sw.split("x"))
            core.N_PARTICLES, core.N_ITERATIONS = P, I
            for arm in arms:
                if arm == "gpu_numpy" and sw not in numpy_sw:
                    continue
                base = os.path.join(a.out, key, sw, arm)
                res = os.path.join(base, "results_long.jsonl")
                done = set()
                if os.path.exists(res):
                    for l in open(res):
                        if l.strip():
                            r = json.loads(l)
                            done.add((r["rank_band"], r["threshold_level"], r["run_id"]))
                plan = []
                for b in bands:
                    levels, info = cell.select_band(b)
                    for lv in cell.order(levels):
                        if a.levels and lv not in a.levels.split(","):
                            continue
                        plan += [(b, lv, k, levels[lv], info) for k in range(1, a.runs + 1) if (b, lv, k) not in done]
                log(f"{key} {sw} {arm}: {len(plan)} runs to do ({len(done)} done)")
                if a.dry:
                    continue
                os.makedirs(base, exist_ok=True)
                cur = None
                for b, lv, k, thr, info in plan:
                    if cur != b:
                        cell.select_band(b)
                        cur = b
                    folder = os.path.join(base, b, lv, f"run_{k}")
                    try:
                        if arm == "gpu_numpy":
                            r = core.run_pso(prob, "gpu_numpy", k, folder, res, pool=None, out_root=base, threshold=thr, level=lv)
                        elif arm in ("woodbury_batch", "fast"):
                            pool = G.GpuPool(ev, arm)
                            r = core.run_pso(prob, arm, k, folder, res, pool=pool, out_root=base, threshold=thr, level=lv)
                            with open(os.path.join(base, "pool_stats.jsonl"), "a") as fh:     # cache / pruning counters (AllCPU records lack them)
                                fh.write(json.dumps({"rank_band": b, "threshold_level": lv, "run_id": k, **pool.stats()}) + "\n")
                        else:
                            r = S.run_search(cell, ev, arm, k, folder, res, threshold=thr, level=lv, polisher=pol, out_root=base)
                        log(f"   {key} {sw} {arm} {b} {lv}={thr:.6g} run {k}: best {r['best_cost']:.6g} @ {r['best_n_caps']} caps, "
                            f"{r['total_wall_time_s']:.2f}s, " + ("HIT" if r["success"] else "miss") + f", evals {r['n_fitness_evals']}")
                    except Exception as exc:                                     # noqa: BLE001
                        log(f"   [ERROR] {key} {sw} {arm} {b} {lv} run {k}: {exc!r}")
    log("GPUALL_DONE")


if __name__ == "__main__":
    floor_pass() if a.floor else study()
