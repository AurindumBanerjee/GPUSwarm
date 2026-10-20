#!/usr/bin/env python3
"""Time-to-target ablation on the 28-port PDN (netA), single objective: peak |Z11| over the whole 100.9 Hz - 3 GHz grid.
Targets 200 / 180 / 150 / 120 ohm (T4 easiest ... T1 hardest, thresholds.json), each method x each target x NUM_RUNS paired
seeds; every run stops the instant the running best meets the target and records time_to_target / caps_at_target / n_evals.
Methods: numpy (full inverse), solve (batched solve), sm (Sherman-Morrison), iterative (2nd-order Neumann; an
APPROXIMATION, timing reference only). pure_python lives in ../baseline_pure_python.

  python pso_pcb28_time.py --validate
  NUM_RUNS=10 python pso_pcb28_time.py          # results in ./results, per-run logs in results/<prob>/<method>/<T>/run_N/
env: METHODS (default numpy,solve,sm,iterative), NUM_RUNS (10), STUDY_THREADS (1), OUT, PCB_DATA, MURATA_MAT"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
os.environ.setdefault("METHODS", "numpy,solve,sm,iterative"); os.environ.setdefault("NUM_RUNS", "10"); os.environ["RANK_BAND"] = "B3G"
import numpy as np
import pso_pcb28 as base
core = base.core
core.BANDS.clear(); core.BANDS["B3G"] = None                      # one band only: the whole grid
T = json.load(open(os.path.join(H, "thresholds.json")))["problems"]["pcb28_netA"]
LEVELS = ["T4", "T3", "T2", "T1"]                                  # easiest first
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    core.library(); prob = core.Problem(**base.PROBLEMS[0]); out = os.environ.get("OUT", os.path.join(H, "results"))
    core.selftest(prob, core.METHODS); bare = float(np.max(core.bare_peaks(prob)["B3G"]))
    print(f"bare {bare:.5g} ohm | targets " + "  ".join(f"{k}={T[k]:g}" for k in LEVELS))
    assert all(T["min_reached_ohm"] < T[k] < bare for k in LEVELS), "a target is outside (floor, bare)"
    if "--validate" in sys.argv: print("--validate: OK"); sys.exit(0)
    res = os.path.join(out, "results_long.jsonl"); os.makedirs(out, exist_ok=True)
    pool = ThreadPoolExecutor(core.THREADS) if core.THREADS > 1 else None
    for meth in core.METHODS:
        for lvl in LEVELS:
            for run in range(1, core.NUM_RUNS + 1):
                print(f"{prob.name} {meth} {lvl}={T[lvl]:g} run {run}/{core.NUM_RUNS}", flush=True)
                try:
                    r = core.run_pso(prob, meth, run, os.path.join(out, prob.name, meth, lvl, f"run_{run}"), res, pool=pool,
                                     out_root=out, threshold=T[lvl], level=lvl)
                    print(f"    best {r['best_peak_ohm']['B3G']:.6g} ohm @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.1f}s, "
                          + (f"hit in {r['time_to_target_s']:.1f}s" if r["success"] else "target missed"), flush=True)
                except Exception as exc: print("    [ERROR]", exc, flush=True)
