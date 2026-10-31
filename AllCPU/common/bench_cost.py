#!/usr/bin/env python3
"""Measured seconds per evaluation (1 thread, this machine) of every method on every PDN of the ACTIVE configuration, at 5 and 20 capacitors, and the
resulting hours per capacitor count / per 10-capacitor run for each swarm. Writes nothing; prints a table (prep_all.py puts it in the report).
   python bench_cost.py [--pdns pdn1,pdn2,pdn3,pdn4] [--swarms 50x15,100x15,50x40,100x40,200x40]"""
import os, sys, time, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
ap = argparse.ArgumentParser(); ap.add_argument("--pdns", default="pdn1,pdn2,pdn3,pdn4"); ap.add_argument("--swarms", default="50x15,100x15,50x40,100x40,200x40"); a = ap.parse_args()
import numpy as np, allcpu_pdns as R
print(f"configuration {R.CFG}; seconds per evaluation, 1 thread (mean of 3 random placements)")
for key in a.pdns.split(","):
    cell = R.Cell(key, 24, 2); core = cell.core; p = cell.prob; d = core.library()["d"]; rng = np.random.default_rng(11); tab = {}
    for m in ("numpy", "solve", "sm", "iterative"):
        for nc in (5, 20):
            nc = min(nc, len(p.pads)); cf = [[(int(rng.integers(d.shape[0])), int(q)) for q in rng.choice(p.pads, nc, replace=False)] for _ in range(3)]
            core.peaks_all_bands(p, cf[0], m); t0 = time.perf_counter()
            for c in cf: core.peaks_all_bands(p, c, m)
            tab[(m, nc)] = (time.perf_counter() - t0) / 3
    print(f"\n{cell.spec['label']}  N={p.N} F={p.F}")
    print("  method      s/eval@5   s/eval@20   " + "   ".join(f"{s}: h/cap-count" for s in a.swarms.split(",")))
    for m in ("numpy", "solve", "sm", "iterative"):
        t5, t20 = tab[(m, 5)], tab[(m, min(20, len(p.pads)))]; tm = 0.5 * (t5 + t20)
        print(f"  {m:10} {t5:9.4f}  {t20:10.4f}   " + "   ".join(f"{int(s.split('x')[0]) * int(s.split('x')[1]) * tm / 3600:>{len(s) + 14}.2f}" for s in a.swarms.split(",")))
    print("  (a run that needs k capacitors costs about k x the h/cap-count figure)")
