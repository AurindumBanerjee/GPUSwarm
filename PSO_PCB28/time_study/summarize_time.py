#!/usr/bin/env python3
"""Table of time-to-target per method and target from results/results_long.jsonl (+ speed-up vs numpy)."""
import os, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); rows = [json.loads(l) for l in open(os.path.join(H, "results", "results_long.jsonl"))]
G = collections.defaultdict(list)
for r in rows: G[(r["threshold_level"], r["method"])].append(r)
print(f"{'target':>6} {'ohm':>7} {'method':10} {'ok':>6} {'time_to_target s (mean)':>24} {'median':>8} {'caps':>6} {'evals':>8} {'speedup vs numpy':>17}")
for lv in ("T4", "T3", "T2", "T1"):
    base = None
    for m in ("numpy", "solve", "sm", "iterative"):
        g = G.get((lv, m))
        if not g: continue
        ok = [r for r in g if r["success"]]; t = np.array([r["time_to_target_s"] for r in ok]) if ok else np.array([np.nan])
        if m == "numpy": base = np.nanmean(t)
        sp = f"{base/np.nanmean(t):.2f}x" if base and ok else "-"
        print(f"{lv:>6} {g[0]['threshold_ohm']:7.1f} {m:10} {len(ok):2d}/{len(g):<3d} {np.nanmean(t):24.3f} {np.nanmedian(t):8.3f} "
              f"{np.mean([r['caps_at_target'] for r in ok]) if ok else float('nan'):6.1f} {np.mean([r['n_fitness_evals'] for r in ok]) if ok and 'n_fitness_evals' in ok[0] else float('nan'):8.0f} {sp:>17}")
