#!/usr/bin/env python3
"""Legacy vs Murata: success, time-to-target, caps, evals per target, and the floor reached (results/<lib>/results_long.jsonl)."""
import os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = {}
for lib in ("ref", "murata"):
    p = os.path.join(H, "results", lib, "results_long.jsonl")
    if os.path.exists(p): R[lib] = [json.loads(l) for l in open(p)]
print(f"{'level':6} {'thr mOhm':>8} {'lib':7} {'hit':>7} {'mean s':>8} {'mean caps':>9} {'mean evals':>10} {'best peak mOhm':>15}")
for lv in ("T1", "T2", "T3", "T4", "floor"):
    for lib, rows in R.items():
        g = [r for r in rows if r["threshold_level"] == lv]
        if not g: continue
        ok = [r for r in g if r["success"]]; thr = g[0]["threshold_ohm"]
        best = min(r["best_peak_ohm"]["BFULL"] for r in g)
        print(f"{lv:6} {(thr*1e3 if thr else float('nan')):8.1f} {lib:7} {len(ok):3d}/{len(g):<3d} "
              f"{np.mean([r['time_to_target_s'] for r in ok]) if ok else float('nan'):8.2f} "
              f"{np.mean([r['caps_at_target'] for r in ok]) if ok else float('nan'):9.2f} "
              f"{np.mean([r['n_fitness_evals'] for r in ok]) if ok else float('nan'):10.0f} {best*1e3:15.2f}")
