#!/usr/bin/env python3
"""Baseline: the ORIGINAL pure-Python fitness (SB.py 'pure_python': Gauss-Jordan on plain lists, no numpy in the inversion)
inside the same PSO, on the 28-port PDN (netA) with the Murata library. 4 runs by default, at EVERY target (200/180/150/120 ohm = T4..T1), round-robin: run 1 of every target, then run 2 of every target, ... The inversion routine is core._gauss_jordan_inverse, taken unchanged
from SB.py; only the library rows are converted to lists lazily (a full 16171x2828 list copy would need ~2 GB).

  python pso_pcb28_pure.py --validate        # also times one fitness evaluation and prints an ETA
  NUM_RUNS=4 python pso_pcb28_pure.py
env: NUM_RUNS (4), LEVELS (T4,T3,T2,T1; comma list of T1..T4 from ../time_study/thresholds.json), N_PARTICLES/N_ITERATIONS
     (default 50/15 = SB.py's values; the numpy runs use 100/40), TIME_LIMIT_S (per run, default 1 week), OUT, PCB_DATA, MURATA_MAT
A run stops the moment the target is met; a run that hits TIME_LIMIT_S (1 week) is flagged timed_out. Results: results/results_long.jsonl."""
import os, sys, json, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
os.environ["RANK_BAND"] = "B3G"
import numpy as np
import pso_pcb28 as base
core = base.core
core.BANDS.clear(); core.BANDS["B3G"] = None
core.N_PARTICLES = int(os.environ.get("N_PARTICLES", 50)); core.N_ITERATIONS = int(os.environ.get("N_ITERATIONS", 15))
core.TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", 7 * 24 * 3600))
_rows = {}
def inv_pure_python(prob, config, f):                          # same math as core.inv_pure_python, lazy library rows
    Yeq = [row[:] for row in prob.lists[f]]
    for cap, port in config:
        r = _rows.get(cap)
        if r is None: r = _rows[cap] = core._lib["d"][cap].tolist()
        Yeq[port][port] += r[f]
    inv = core._gauss_jordan_inverse(Yeq)
    return [abs(inv[k][k]) for k in range(prob.n_obs)]
core.inv_pure_python = inv_pure_python
T = json.load(open(os.path.join(os.path.dirname(H), "time_study", "thresholds.json")))["problems"]["pcb28_netA"]
LEVELS = os.environ.get("LEVELS", "T4,T3,T2,T1").split(","); NUM_RUNS = int(os.environ.get("NUM_RUNS", 4))
OUT = os.environ.get("OUT", os.path.join(H, "results"))

if __name__ == "__main__":
    core.library(); prob = core.Problem(**base.PROBLEMS[0])
    cfg = [(5, 1), (700, 3), (9000, 7)]
    t0 = time.perf_counter(); zp = core.peaks_all_bands(prob, cfg, "pure_python")["B3G"].max(); tp = time.perf_counter() - t0
    zn = core.peaks_all_bands(prob, cfg, "numpy")["B3G"].max()
    print(f"pure_python vs numpy on one placement: {zp:.9g} vs {zn:.9g} (rel diff {abs(zp-zn)/zn:.2e}); one evaluation {tp:.2f} s")
    assert abs(zp - zn) / zn < 1e-8
    ev_stage = core.N_PARTICLES * core.N_ITERATIONS
    print(f"ETA: one PSO stage = {ev_stage} evals = {ev_stage*tp/3600:.1f} h; a run needs >= 1 stage per capacitor (T3 typically 1-2 stages)")
    if "--validate" in sys.argv: sys.exit(0)
    res = os.path.join(OUT, "results_long.jsonl"); os.makedirs(OUT, exist_ok=True)
    for run in range(1, NUM_RUNS + 1):                      # round-robin: every target finishes run k before any starts run k+1
        for lvl in LEVELS:
            print(f"[{time.strftime('%H:%M:%S')}] pure_python {lvl}={T[lvl]:g} run {run}/{NUM_RUNS}", flush=True)
            r = core.run_pso(prob, "pure_python", run, os.path.join(OUT, prob.name, "pure_python", lvl, f"run_{run}"), res,
                             out_root=OUT, threshold=T[lvl], level=lvl)
            print(f"    best {r['best_peak_ohm']['B3G']:.6g} ohm @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s, "
                  + (f"hit in {r['time_to_target_s']:.0f}s" if r["success"] else "target not met (timed_out=%s)" % r["timed_out"]), flush=True)
