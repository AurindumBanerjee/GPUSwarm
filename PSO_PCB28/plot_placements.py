#!/usr/bin/env python3
"""|Z11| of the PDN at the observation port with and without decaps (100.9 Hz - 3 GHz), one PNG per placement (run),
with a table of which part sits on which pad / original board port.
  python plot_placements.py --jsonl results/results_long.jsonl [--top N]   # every record (default) or the N best, one PNG each
  python plot_placements.py --placements placements_user_runs.json                 # [{label, placement:[[pad, part], ...]}]
Pad = matrix index in y_pcb_netA (0 = observation port 1); original port = [1,3,5,6,10,11,12,18,19,22,23,25][pad].
env: PCB_DATA (default ./Data), MURATA_MAT, OUT (png prefix)"""
import os, sys, json, csv, argparse, numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get("PCB_DATA", os.path.join(H, "Data"))
PORTS = [1, 3, 5, 6, 10, 11, 12, 18, 19, 22, 23, 25]
ap = argparse.ArgumentParser(); ap.add_argument("--jsonl"); ap.add_argument("--placements"); ap.add_argument("--top", type=int, default=0, help="0 = every record")
ap.add_argument("--out", default=os.environ.get("OUT", os.path.join(H, "pdn_with_without_decaps")))
a = ap.parse_args()
if a.jsonl:
    R = [json.loads(l) for l in open(a.jsonl)]; R.sort(key=lambda r: min(r["best_peak_ohm"].values()) if "B3G" not in r["best_peak_ohm"] else r["best_peak_ohm"]["B3G"])
    P = [{"label": f"{r.get('method','')} {r.get('threshold_level') or ''} run {r['run_id']} ({r['best_n_caps']} caps, {r['best_peak_ohm']['B3G']:.1f} ohm)",
          "placement": [[p["pad"], p["name"]] for p in r["placement"]]} for r in (R[:a.top] if a.top else R)]
else: P = json.load(open(a.placements))
f = sio.loadmat(os.path.join(D, "freq_pcb.mat"))["freq"].ravel()
y = np.transpose(sio.loadmat(os.path.join(D, "y_pcb_netA.mat"))["y"], (2, 0, 1)).astype(complex)
names = [r["part_number"] for r in csv.DictReader(open(os.path.join(D, "murata_index_rlc.csv")))]; row = {n: i for i, n in enumerate(names)}
Y = sio.loadmat(os.environ.get("MURATA_MAT", os.path.join(D, "murata_decaps_pcb_rlc.mat")))["decaps"]
z0 = np.abs(np.linalg.inv(y)[:, 0, 0]); import re
for k, p in enumerate(P, 1):
    A = y.copy()
    for pad, nm in p["placement"]: A[:, pad, pad] += Y[row[nm]]
    z = np.abs(np.linalg.inv(A)[:, 0, 0])
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[3, 1.3]), constrained_layout=True)
    ax[0].loglog(f, z0, "k", lw=2, label=f"bare PDN (peak {z0.max():.1f} ohm)")
    ax[0].loglog(f, z, "C0", lw=1.5, label=f"with {len(p['placement'])} decaps (peak {z.max():.1f} ohm)")
    for pad, nm in p["placement"]: ax[0].axvline(f[np.argmin(np.abs(Y[row[nm]]*0 + 1/np.abs(Y[row[nm]])))], color="C1", alpha=.25, lw=.8)
    for t in (146.5, 176.0, 205.5): ax[0].axhline(t, color="0.6", ls=":", lw=.8)
    ax[0].set_xlabel("Frequency [Hz]"); ax[0].set_ylabel("|Z11| at port 1 [ohm]"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=9)
    ax[0].set_title(f"{p['label']}: |Z11| with vs without decaps (orange lines = each part's SRF, dotted = study targets)", fontsize=10)
    ax[1].axis("off"); pl = sorted(p["placement"])
    ax[1].text(0, 1, "\n".join(f"pad {pd:2d} (board port {PORTS[pd]:2d}): {nm}" for pd, nm in pl), va="top", family="monospace", fontsize=8)
    out = f"{a.out}_{k}_{re.sub(r'[^A-Za-z0-9]+', '_', p['label']).strip('_')}.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "peak", round(z.max(), 2))
