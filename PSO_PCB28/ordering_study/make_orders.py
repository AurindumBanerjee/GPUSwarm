#!/usr/bin/env python3
"""Library orderings for the encoding study. The Murata library file is already sorted by ascending SRF
(measured argmin|Z|, see murata_index_rlc.csv). This writes permutations of those rows instead of 733 MB copies:

  srf     identity (the shipped file; nearby index = similar resonance)
  cap     ascending nominal capacitance            (a second electrical ordering)
  part    alphabetical part number                 (family/series grouping, electrically arbitrary)
  random  fixed-seed shuffle                       (no locality at all)

orders.npz holds perm_<name>: new_row i  ->  original row perm[i]. orders/<name>.csv lists it with part, SRF, C.
Also prints a smoothness score: mean |d log10|Y|| between index-neighbours over the grid (lower = smoother).
env: PCB_DATA (default ../Data), MURATA_MAT (to compute the smoothness score; skipped if missing)"""
import os, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get("PCB_DATA", os.path.join(H, "..", "Data"))
R = list(csv.DictReader(open(os.path.join(D, "murata_index_rlc.csv")))); n = len(R)
srf = np.array([float(r["srf_hz"]) for r in R]); cap = np.array([float(r["capacitance_pF"]) for r in R]); pn = np.array([r["part_number"] for r in R])
assert np.all(np.diff(srf) >= 0), "shipped library is not SRF-sorted"
P = {"srf": np.arange(n), "cap": np.lexsort((np.arange(n), cap)), "part": np.lexsort((np.arange(n), pn)),
     "random": np.random.default_rng(20261020).permutation(n)}
os.makedirs(os.path.join(H, "orders"), exist_ok=True)
np.savez(os.path.join(H, "orders.npz"), **{f"perm_{k}": v for k, v in P.items()})
for k, p in P.items():
    with open(os.path.join(H, "orders", f"{k}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["new_row", "orig_row", "part_number", "srf_hz", "capacitance_pF"])
        for i, o in enumerate(p): w.writerow([i, int(o), pn[o], srf[o], cap[o]])
print("wrote orders.npz and orders/*.csv for", n, "parts")
mat = os.environ.get("MURATA_MAT", os.path.join(D, "murata_decaps_pcb_rlc.mat"))
if os.path.exists(mat):
    import scipy.io as sio
    L = np.log10(np.abs(sio.loadmat(mat)["decaps"][:, ::20])); print("smoothness (mean |d log10|Y|| between neighbouring indices):")
    for k, p in P.items(): print(f"  {k:7s} {np.abs(np.diff(L[p], axis=0)).mean():.4f}")
