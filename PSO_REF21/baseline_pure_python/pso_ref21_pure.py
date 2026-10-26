#!/usr/bin/env python3
"""Pure-Python baseline on the 21-port reference PDN (port 0 = observation, 20 pads), same updated PSO formulation as every other
method (pso_sp_core.run_pso: per-stage warm start, floor discretisation, resolver, immediate stop at target). Only the fitness
differs: SB.py's Gauss-Jordan on plain lists (core._gauss_jordan_inverse, unchanged). Library rows are converted to lists lazily.
Run-major round-robin over thresholds (run 1 of every target, then run 2, ...).

  python pso_ref21_pure.py --validate      # checks pure_python == numpy on one placement, times one evaluation, prints ETA
  NUM_RUNS=5 python pso_ref21_pure.py
env: LIB (ref | murata; default ref), TARGETS (0.06,0.05,0.045 ohm, hardest listed last -> run easiest first), NUM_RUNS (5),
     N_PARTICLES/N_ITERATIONS (50/15), TIME_LIMIT_S (1 week), OUT, DATA
Warm start: core.WARM_START_PCT = 20 % of the swarm per stage (the rest random)."""
import os, sys, json, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(H))
os.environ.setdefault("LIB", "ref")
import numpy as np
import pso_ref21 as base
core = base.core
core.TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", 7 * 24 * 3600))
LIB = os.environ["LIB"]; assert LIB in ("ref", "murata")
T = [float(x) for x in os.environ.get("TARGETS", "0.06,0.05,0.045").split(",")]
LEVELS = {f"T{i+1}": t for i, t in enumerate(sorted(T, reverse=True))}      # T1 = easiest listed first
NUM_RUNS = int(os.environ.get("NUM_RUNS", 5)); OUT = os.environ.get("OUT", os.path.join(H, "results"))
_rows = {}
def inv_pure_python(prob, config, f):
    Yeq = [row[:] for row in prob.lists[f]]
    for cap, port in config:
        r = _rows.get(cap)
        if r is None: r = _rows[cap] = core._lib["d"][cap].tolist()
        Yeq[port][port] += r[f]
    inv = core._gauss_jordan_inverse(Yeq)
    return [abs(inv[k][k]) for k in range(prob.n_obs)]
core.inv_pure_python = inv_pure_python

if __name__ == "__main__":
    base.load(LIB); prob = core.Problem("ref21", os.path.join(base.DATA, "y2.mat"), n_obs=1, obs_labels=["port0"])
    bare = float(np.max(core.bare_peaks(prob)["BFULL"]))
    assert all(base.FLOOR_OHM < t < bare for t in LEVELS.values()), f"a target is outside (floor {base.FLOOR_OHM}, bare {bare})"
    cfg = [(5, 1), (700, 3), (1500, 7)]
    t0 = time.perf_counter(); zp = core.peaks_all_bands(prob, cfg, "pure_python")["BFULL"].max(); tp = time.perf_counter() - t0
    zn = core.peaks_all_bands(prob, cfg, "numpy")["BFULL"].max()
    print(f"[{LIB}] pure_python vs numpy: {zp:.9g} vs {zn:.9g} (rel {abs(zp-zn)/zn:.2e}); one evaluation {tp:.2f} s; bare {bare:.5g} ohm")
    assert abs(zp - zn) / zn < 1e-8
    ev = core.N_PARTICLES * core.N_ITERATIONS
    print(f"warm start: {core.n_warm_count()} of {core.N_PARTICLES} particles; ETA one stage = {ev} evals = {ev*tp/3600:.2f} h")
    if "--validate" in sys.argv: sys.exit(0)
    out = os.path.join(OUT, LIB); res = os.path.join(out, "results_long.jsonl"); os.makedirs(out, exist_ok=True)
    for run in range(1, NUM_RUNS + 1):
        for lvl, thr in LEVELS.items():
            print(f"[{time.strftime('%H:%M:%S')}] pure_python {lvl}={thr:g} run {run}/{NUM_RUNS}", flush=True)
            r = core.run_pso(prob, "pure_python", run, os.path.join(out, prob.name, "pure_python", lvl, f"run_{run}"), res,
                             out_root=out, threshold=thr, level=lvl)
            print(f"    best {r['best_peak_ohm']['BFULL']:.6g} ohm @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s, "
                  + (f"hit in {r['time_to_target_s']:.0f}s" if r["success"] else f"target not met (timed_out={r['timed_out']})"), flush=True)
