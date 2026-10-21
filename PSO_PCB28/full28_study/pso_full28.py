#!/usr/bin/env python3
"""Method time-to-target study on the FULL 28-port PDN (Data/y_pcb28.mat, 28x28 matrices, 27 pads -- the 16 pads that cannot
influence port 1 stay in the search on purpose). Single objective: peak |Z11| over the whole 100.9 Hz - 3 GHz grid.
Targets from thresholds.json (T1 hardest ... T4 easiest), each method x target x NUM_RUNS paired seeds; a run stops the
instant the running best meets the target.   Methods: numpy, solve, sm (Sherman-Morrison), iterative (approximation).

  python pso_full28.py --validate
  NUM_RUNS=10 python pso_full28.py            # results/ , per-run logs in results/pcb28_full/<method>/<T>/run_N/
env: NET (full = 28x28 default, netA = 12x12 reduction, same targets), METHODS (numpy,solve,sm,iterative), NUM_RUNS (10), MAX_CAPS (14), STUDY_THREADS (1), TIME_LIMIT_S (1800 per run), OUT, PCB_DATA, MURATA_MAT"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
os.environ.setdefault("METHODS", "numpy,solve,sm,iterative"); os.environ.setdefault("NUM_RUNS", "10"); os.environ["RANK_BAND"] = "B3G"
import numpy as np
import pso_pcb28 as base
core = base.core
core.BANDS.clear(); core.BANDS["B3G"] = None
T = json.load(open(os.path.join(H, "thresholds.json")))["targets"]
LEVELS = sorted(T, key=lambda k: -T[k])                              # easiest (highest ohm) first
MAX_CAPS = int(os.environ.get("MAX_CAPS", 14))
NET = os.environ.get("NET", "full")
PROB = (dict(name="pcb28_full", path=os.path.join(base.BOARD, "y_pcb28.mat"), n_obs=1, obs_labels=["port1"]) if NET == "full" else
        dict(name="pcb28_netA", path=os.path.join(base.BOARD, "y_pcb_netA.mat"), n_obs=1, obs_labels=["port1"]))
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    core.library(); prob = core.Problem(**PROB); out = os.environ.get("OUT", os.path.join(H, f"results_{NET}"))
    assert prob.N == (28 if NET == "full" else 12), f"unexpected matrix size {prob.N} for NET={NET}"
    core.selftest(prob, core.METHODS); bare = float(np.max(core.bare_peaks(prob)["B3G"]))
    print(f"N={prob.N} pads={len(prob.pads)} bare {bare:.5g} ohm | targets " + "  ".join(f"{k}={T[k]:g}" for k in LEVELS))
    assert all(117.18 < T[k] < bare for k in LEVELS), "a target is outside (floor 117.18, bare)"
    if "--validate" in sys.argv: print("--validate: OK"); sys.exit(0)
    res = os.path.join(out, "results_long.jsonl"); os.makedirs(out, exist_ok=True)
    pool = ThreadPoolExecutor(core.THREADS) if core.THREADS > 1 else None
    for meth in core.METHODS:
        for lvl in LEVELS:
            for run in range(1, core.NUM_RUNS + 1):
                print(f"{prob.name} {meth} {lvl}={T[lvl]:g} run {run}/{core.NUM_RUNS}", flush=True)
                try:
                    r = core.run_pso(prob, meth, run, os.path.join(out, prob.name, meth, lvl, f"run_{run}"), res, pool=pool,
                                     out_root=out, threshold=T[lvl], level=lvl, max_caps=MAX_CAPS)
                    print(f"    best {r['best_peak_ohm']['B3G']:.6g} ohm @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.1f}s, "
                          + (f"hit in {r['time_to_target_s']:.1f}s" if r["success"] else "target missed"), flush=True)
                except Exception as exc: print("    [ERROR]", exc, flush=True)
