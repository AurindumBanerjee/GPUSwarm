#!/usr/bin/env python3
"""Time-to-target per method and target, speed-up vs numpy.   python summarize_methods.py [results/results_long.jsonl ...]"""
import os, sys, json, glob, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); files = sys.argv[1:] or [os.path.join(H, "results", "results_long.jsonl")]
rows = [json.loads(l) for f in files for p in glob.glob(f) for l in open(p)]
G = collections.defaultdict(list)
for r in rows: G[(r["threshold_ohm"], r["method"])].append(r)
print(f"{'target':>9} {'method':10} {'hit':>6} {'mean s':>9} {'median s':>9} {'caps':>6} {'evals':>8} {'speedup vs numpy':>17}")
for thr in sorted({k[0] for k in G if k[0] is not None}, reverse=True):
    base = None
    for m in ("numpy", "solve", "sm", "iterative"):
        g = G.get((thr, m))
        if not g: continue
        ok = [r for r in g if r["success"]]; t = np.array([r["time_to_target_s"] for r in ok]) if ok else np.array([np.nan])
        if m == "numpy": base = np.nanmean(t)
        sp = f"{base/np.nanmean(t):.2f}x" if base and ok else "-"
        print(f"{thr:9.4g} {m:10} {len(ok):2d}/{len(g):<3d} {np.nanmean(t):9.3f} {np.nanmedian(t):9.3f} "
              f"{np.mean([r['caps_at_target'] for r in ok]) if ok else float('nan'):6.1f} {np.mean([r['n_fitness_evals'] for r in ok]) if ok else float('nan'):8.0f} {sp:>17}")
