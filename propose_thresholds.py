#!/usr/bin/env python3
"""Propose time-to-target thresholds from full-budget runs, so that consecutive targets need about 2-3 more decaps.
Reads results_long.jsonl records WITHOUT a threshold (they hold the best-so-far peak after 1..N capacitors in
curve_caps / curve_peak_ohm), builds runs x caps, and for each requested capacitor count k sets
    T_k = the QUANTILE-th quantile of the best-so-far peak after k caps     (default 0.9: 90 % of runs hit it with <= k caps)
Then reports, for every proposed target, how many caps the median / worst run really needed.

  python propose_thresholds.py --band B3G --caps 2,4,6,8 PSO_PCB28/ordering_study/results_*/results_long.jsonl
  python propose_thresholds.py --band BFULL --scale 1000 --unit mOhm --caps 4,7,10,13 PSO_REF21/results/ref/results_long.jsonl PSO_REF21/results/murata/results_long.jsonl
Prints a table of the median / best / worst curve and writes proposed_thresholds.json (T1 hardest ... T4 easiest)."""
import sys, json, argparse, glob, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--band", default="B3G"); ap.add_argument("--caps", default="2,4,6,8")
ap.add_argument("--quantile", type=float, default=0.9); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--unit", default="ohm")
ap.add_argument("--out", default="proposed_thresholds.json"); a = ap.parse_args()
curves = []
for pat in a.files:
    for fn in glob.glob(pat):
        for line in open(fn):
            r = json.loads(line)
            if r.get("threshold_ohm") is None and r.get("curve_peak_ohm"): curves.append(np.minimum.accumulate(r["curve_peak_ohm"]))
if not curves: sys.exit("no full-budget (threshold-free) records found")
n = min(map(len, curves)); C = np.array([c[:n] for c in curves]) * a.scale
print(f"{len(curves)} full-budget runs, {n} capacitor counts, unit {a.unit}")
print(f"{'caps':>4} {'best':>10} {'median':>10} {'p90':>10} {'worst':>10}")
for k in range(n): print(f"{k+1:4d} {C[:,k].min():10.4g} {np.median(C[:,k]):10.4g} {np.quantile(C[:,k], a.quantile):10.4g} {C[:,k].max():10.4g}")
K = [int(x) for x in a.caps.split(",")]; T = [float(np.quantile(C[:, k-1], a.quantile)) for k in K]
def r3(x): return float(f"{x:.4g}")
print(f"\nproposed (quantile {a.quantile}): " + "  ".join(f"{k} caps -> {r3(t)}" for k, t in zip(K, T)))
print(f"{'target':>10} {'median caps':>12} {'worst caps':>11} {'hit rate':>9}")
for t in T:
    need = [int(np.argmax(c <= t)) + 1 if (c <= t).any() else None for c in C]; ok = [x for x in need if x]
    print(f"{r3(t):10.4g} {np.median(ok) if ok else float('nan'):12.1f} {max(ok) if ok else float('nan'):11} {len(ok)}/{len(need)}")
if any(abs(T[i]-T[i+1]) < 0.003*T[i] for i in range(len(T)-1)): print("WARNING: curve is flat here; some targets differ by < 0.3 %, capacitor counts cannot separate them.")
names = {f"T{len(T)-i}": r3(t) for i, t in enumerate(T)}          # last (most caps) = T1 hardest
json.dump({"unit": a.unit, "scale_to_ohm": 1/a.scale, "caps_for_targets": K, "targets": names}, open(a.out, "w"), indent=1); print("wrote", a.out, names)
