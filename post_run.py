#!/usr/bin/env python3
"""Post-run report for every PDN study in GPUSwarm (28-port full / netA, 21-port), identical output style.

 1. Method table per (problem, target): hits, mean/median time-to-target, decaps at target, evals, best peak,
    speedup vs pure_python (ratio of mean time-to-target; also the median of per-run paired ratios, same run_id = same seeds).
 2. One placement plot per run, saved as <run folder>/placement.png: bare vs optimised |Z11| at the observation port, target line,
    and a table of pad (matrix index) -> board port -> part number.

  python post_run.py                       # discover every results_long.jsonl under GPUSwarm (skips *_old_*, time_study, ordering_study)
  python post_run.py path/to/results_long.jsonl ... [--no-plots] [--methods numpy,pure_python]
Output: post_run_out/summary_<problem>.txt/.csv. Table values use successful runs only; misses/time-outs are counted in 'hit'."""
import os, sys, glob, json, csv, collections, numpy as np, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "post_run_out"); os.makedirs(OUT, exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--")]; PLOTS = "--no-plots" not in sys.argv
only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--methods=")), None)
if args: files = args
else:
    files = [f for f in glob.glob(os.path.join(H, "**", "results_long.jsonl"), recursive=True)
             if not any(s in f for s in ("_old_", "time_study", "ordering_study", "results_murata"))]
D28 = os.path.join(H, "PSO_PCB28", "Data"); D21 = os.path.join(H, "PSO_REF21", "Data")
NETA = [1, 3, 5, 6, 10, 11, 12, 18, 19, 22, 23, 25]
_cache = {}
def pdn(problem, lib):
    k = (problem, lib)
    if k in _cache: return _cache[k]
    if problem.startswith("pcb28"):
        f = sio.loadmat(os.path.join(D28, "freq_pcb.mat"))["freq"].ravel()
        y = sio.loadmat(os.path.join(D28, "y_pcb28.mat" if problem == "pcb28_full" else "y_pcb_netA.mat"))["y"]
        Y = sio.loadmat(os.path.join(D28, "murata_decaps_pcb_rlc.mat"))["decaps"]
        names = [r["part_number"] for r in csv.DictReader(open(os.path.join(D28, "murata_index_rlc.csv")))]
        port = (lambda p: p + 1) if problem == "pcb28_full" else (lambda p: NETA[p])
    else:
        f = sio.loadmat(os.path.join(D21, "freq2.mat"))["freq"].ravel(); y = sio.loadmat(os.path.join(D21, "y2.mat"))["y"]
        if lib == "ref": Y = sio.loadmat(os.path.join(D21, "decaps.mat"))["decaps"]; names = [f"ref_{i}" for i in range(Y.shape[0])]
        else:
            Y = sio.loadmat(os.path.join(D21, "murata_decaps_ref21.mat"))["decaps"]
            names = [r["part_number"] for r in csv.DictReader(open(os.path.join(D21, "murata_index_ref21.csv")))]
        port = lambda p: p + 1
    y = np.transpose(y, (2, 0, 1)).astype(complex); z0 = np.abs(np.linalg.inv(y)[:, 0, 0])
    _cache[k] = (f, y, z0, Y, {n: i for i, n in enumerate(names)}, port); return _cache[k]

rows = []
for fp in files:
    root = os.path.dirname(fp); lib = os.path.basename(root) if os.path.basename(root) in ("ref", "murata") else "murata"
    for l in open(fp):
        r = json.loads(l); r["_root"] = root; r["_lib"] = lib
        if only is None or r["method"] in only: rows.append(r)
print(f"{len(rows)} run records from {len(files)} files")

# ---- 1. tables
G = collections.defaultdict(list)
for r in rows:
    if r["threshold_ohm"] is not None: G[(r["problem"], r["threshold_ohm"], r["method"])].append(r)
ORDER = ["pure_python", "numpy", "solve", "sm", "iterative"]
for prob in sorted({k[0] for k in G}):
    lines = [f"{'target[ohm]':>11} {'method':12} {'hit':>6} {'mean s':>10} {'median s':>10} {'caps':>5} {'evals':>8} {'best ohm':>10} {'speedup vs pure_python':>23} {'paired median':>14}"]
    csvr = [["problem", "target_ohm", "method", "hits", "runs", "mean_s", "median_s", "caps_at_target", "evals", "best_peak_ohm", "speedup_vs_pure_python", "paired_median_speedup"]]
    for thr in sorted({k[1] for k in G if k[0] == prob}, reverse=True):
        base = G.get((prob, thr, "pure_python")); bok = {r["run_id"]: r["time_to_target_s"] for r in (base or []) if r["success"]}
        for m in ORDER + sorted({k[2] for k in G if k[0] == prob} - set(ORDER)):
            g = G.get((prob, thr, m))
            if not g: continue
            ok = [r for r in g if r["success"]]; t = np.array([r["time_to_target_s"] for r in ok]) if ok else np.array([np.nan])
            sp = f"{np.mean(list(bok.values()))/np.nanmean(t):.1f}x" if bok and ok else "-"
            pr = [bok[r["run_id"]] / r["time_to_target_s"] for r in ok if r["run_id"] in bok]; ps = f"{np.median(pr):.1f}x" if pr else "-"
            caps = np.mean([r["caps_at_target"] for r in ok]) if ok else np.nan; ev = np.mean([r["n_fitness_evals"] for r in ok]) if ok else np.nan
            best = min(min(r["best_peak_ohm"].values()) for r in g)
            lines.append(f"{thr:11.4g} {m:12} {len(ok):2d}/{len(g):<3d} {np.nanmean(t):10.2f} {np.nanmedian(t):10.2f} {caps:5.1f} {ev:8.0f} {best:10.5g} {sp:>23} {ps:>14}")
            csvr.append([prob, thr, m, len(ok), len(g), np.nanmean(t), np.nanmedian(t), caps, ev, best, sp, ps])
    open(os.path.join(OUT, f"summary_{prob}.txt"), "w").write("\n".join(lines) + "\n")
    csv.writer(open(os.path.join(OUT, f"summary_{prob}.csv"), "w")).writerows(csvr)
    print(f"\n== {prob} ==\n" + "\n".join(lines))

# ---- 2. one placement plot per run
if PLOTS:
    n = 0
    for r in rows:
        if not r.get("placement"): continue
        f, y, z0, Y, row, port = pdn(r["problem"], r["_lib"]); A = y.copy()
        for p in r["placement"]: A[:, p["pad"], p["pad"]] += Y[row[p["name"]]]
        z = np.abs(np.linalg.inv(A)[:, 0, 0]); thr = r["threshold_ohm"]
        fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[3, 1.4]), constrained_layout=True)
        ax[0].loglog(f, z0, "k", lw=2, label=f"bare PDN (peak {z0.max():.4g} ohm)")
        ax[0].loglog(f, z, "C0", lw=1.5, label=f"{len(r['placement'])} decaps (peak {z.max():.4g} ohm)")
        if thr: ax[0].axhline(thr, color="C3", ls="--", lw=1, label=f"target {thr:g} ohm ({'met' if r['success'] else 'missed'})")
        ax[0].set_xlabel("Frequency [Hz]"); ax[0].set_ylabel("|Z11| at observation port [ohm]"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=9)
        ax[0].set_title(f"{r['problem']} | {r['method']} | {r.get('threshold_level') or 'no target'} | run {r['run_id']} | {r['total_wall_time_s']:.0f} s", fontsize=10)
        ax[1].axis("off"); pl = sorted(r["placement"], key=lambda p: p["pad"])
        ax[1].text(0, 1, "\n".join(f"pad {p['pad']:2d} (board port {port(p['pad']):2d}): {p['name']}" for p in pl), va="top", family="monospace", fontsize=8)
        fo = os.path.join(r["_root"], r["folder"]); os.makedirs(fo, exist_ok=True); fig.savefig(os.path.join(fo, "placement.png"), dpi=130); plt.close(fig); n += 1
    print(f"\nwrote {n} placement plots (<run folder>/placement.png); summaries in {OUT}")
