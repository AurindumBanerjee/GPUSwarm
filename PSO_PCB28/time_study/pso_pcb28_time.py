#!/usr/bin/env python3
"""Time-to-target ablation on the 28-port PDN (netA), single objective: peak |Z11| over the whole 100.9 Hz - 3 GHz grid.
Same protocol as the original timing study: each method x each target (T3 easy, T2, T1 hard) x NUM_RUNS paired seeds, every
run stops the instant the running best meets the target and records time_to_target / caps_at_target / n_evals.
Methods: numpy (full inverse), solve (batched solve), sm (Sherman-Morrison), iterative (2nd-order Neumann; an
APPROXIMATION, timing reference only). pure_python is not included (hours per evaluation).

  python pso_pcb28_time.py --validate
  NUM_RUNS=10 python pso_pcb28_time.py          # results in ./results, per-run logs in results/<prob>/<method>/<T>/run_N/
env: METHODS (default numpy,solve,sm,iterative), NUM_RUNS (10), STUDY_THREADS (1), OUT, PCB_DATA, MURATA_MAT
Targets live in thresholds.json next to this file and must lie strictly between the best floor and the bare peak."""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
os.environ.setdefault("METHODS", "numpy,solve,sm,iterative"); os.environ.setdefault("NUM_RUNS", "10"); os.environ["RANK_BAND"] = "B3G"
import pso_pcb28 as base
core = base.core
core.BANDS.clear(); core.BANDS["B3G"] = None                      # one band only: the whole grid
core._thr.update(json.load(open(os.path.join(H, "thresholds.json"))))
if __name__ == "__main__":
    core.library()
    core.main("PCB28-time", base.PROBLEMS, os.environ.get("OUT", os.path.join(H, "results")), argv=sys.argv[1:] + ["--thresholds"])
