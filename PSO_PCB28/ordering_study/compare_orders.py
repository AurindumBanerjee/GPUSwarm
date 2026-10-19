#!/usr/bin/env python3
"""Paired comparison across orderings from results_<order>/results_long.jsonl (same run ids = same seeds).
Reports best peak |Z11| per band (median, mean, wins vs srf) and the whole peak-vs-capacitor-count curve; writes orders_compare.png."""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); BAND = os.environ.get("RANK_BAND", "B3G")
R = {}
for d in sorted(glob.glob(os.path.join(H, "results_*"))):
    p = os.path.join(d, "results_long.jsonl")
    if os.path.exists(p): R[os.path.basename(d)[8:]] = {r["run_id"]: r for r in map(json.loads, open(p))}
if not R: raise SystemExit("no results yet")
ref = "srf" if "srf" in R else next(iter(R))
print(f"band {BAND}; paired by run id against '{ref}'")
print(f"{'order':8s} {'runs':>4s} {'median':>9s} {'mean':>9s} {'best':>9s} {'wins vs ref':>11s} {'median caps':>11s}")
plt.figure(figsize=(8, 5))
for k, rr in R.items():
    v = np.array([rr[i]["best_peak_ohm"][BAND] for i in sorted(rr)]); caps = np.median([rr[i]["best_n_caps"] for i in rr])
    com = sorted(set(rr) & set(R[ref])); w = sum(rr[i]["best_peak_ohm"][BAND] < R[ref][i]["best_peak_ohm"][BAND] for i in com) if k != ref else "-"
    print(f"{k:8s} {len(v):4d} {np.median(v):9.4g} {v.mean():9.4g} {v.min():9.4g} {str(w)+'/'+str(len(com)) if k!=ref else '-':>11s} {caps:11.1f}")
    cur = [r.get("curve_peak_ohm") for r in rr.values()]
    cur = [c for c in cur if c]
    if cur: m = min(map(len, cur)); plt.plot(range(1, m+1), np.median([np.minimum.accumulate(c[:m]) for c in cur], 0), label=k)
plt.yscale("log"); plt.xlabel("decaps"); plt.ylabel(f"median best peak |Z11| ({BAND}) [ohm]"); plt.legend(); plt.grid(alpha=.3)
plt.savefig(os.path.join(H, "orders_compare.png"), dpi=130); print("wrote orders_compare.png")
