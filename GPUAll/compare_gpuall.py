#!/usr/bin/env python3
"""
Comparative study of the GPUAll arms (and of AllCPU's CPU methods, when their results exist).

   python compare_gpuall.py [--out out_<cfg>] [--cpu ../AllCPU/CPUParallel/out_<cfg>[,../AllCPU/CPUSerial/out_<cfg>]] [--overlay]

Per (pdn, swarm, band, level, arm): runs, success rate, mean wall time, time per success (all wall time / successes), capacitors at
target, best cost, fitness evaluations (+ polish evaluations).
Exact arms: trajectory identity (best cost, best capacitor count, evaluations) and paired speed-up against gpu_numpy and, if given,
against the CPU numpy / solve records of AllCPU on the same (pdn, swarm, band, level, run).
Search arms: Fisher exact test of the success count against 'fast' (the exact PSO).
Writes compare_<cfg>/summary.csv, summary.md, identity.csv, speedup.csv, fisher.csv and plots/ (success heat maps, time per success,
and with --overlay the bare vs best |Z| of every arm on one plot per (pdn, band, level), all observation ports).
"""
import os
import sys
import json
import glob
import argparse

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(H), "AllCPU", "common"))
os.environ.setdefault("GPUSWARM_ROOT", os.path.dirname(H))
os.environ.setdefault("ALLCPU_TARGETS", os.path.join(H, "targets_gpuall.json"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--cpu", default="")
ap.add_argument("--overlay", action="store_true")
a = ap.parse_args()
cfg = os.environ.get("ALLCPU_GRID", "5g") + "_" + os.environ.get("ALLCPU_LIB", "murata")
OUT = a.out or os.path.join(H, "out_" + cfg)
DST = os.path.join(H, "compare_" + cfg)
os.makedirs(os.path.join(DST, "plots"), exist_ok=True)
EXACT = ["gpu_numpy", "woodbury_batch", "fast"]


def load(root, source):
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "*", "results_long.jsonl")):
        pdn, sw, arm = f.split(os.sep)[-4:-1]
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                r.update(pdn=pdn, swarm=sw, arm=(arm if source == "gpu" else "cpu_" + arm), source=source)
                rows.append(r)
    return rows


rows = load(OUT, "gpu")
for c in [x for x in a.cpu.split(",") if x]:
    rows += load(c, "cpu")
if not rows:
    sys.exit(f"no results under {OUT}")
df = pd.DataFrame(rows)
df["n_polish_evals"] = df.get("n_polish_evals", pd.Series(0, index=df.index)).fillna(0)
K = ["pdn", "swarm", "rank_band", "threshold_level"]


def tps(g):
    s = int(g.success.sum())
    return g.total_wall_time_s.sum() / s if s else np.inf


summ = []
for key, g in df.groupby(K + ["arm"]):
    ok = g[g.success == True]                                                # noqa: E712
    summ.append(dict(zip(K + ["arm"], key), runs=len(g), successes=int(g.success.sum()), success_rate=g.success.mean(),
                     mean_wall_s=g.total_wall_time_s.mean(), time_per_success_s=tps(g),
                     caps_at_target=ok.caps_at_target.mean() if len(ok) else np.nan, best_cost_mean=g.best_cost.mean(),
                     best_cost_min=g.best_cost.min(), evals_mean=g.n_fitness_evals.mean(), polish_evals_mean=g.n_polish_evals.mean(),
                     stagnated=int(g.get("stagnated", pd.Series(False)).sum()), timed_out=int(g.get("timed_out", pd.Series(False)).sum())))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(DST, "summary.csv"), index=False)

# ---- identity + paired speed-up of exact arms against references
idn, spd = [], []
refs = [r for r in ("gpu_numpy", "cpu_numpy", "cpu_solve") if r in set(df.arm)]
for ref in refs:
    R_ = df[df.arm == ref].set_index(K + ["run_id"])
    for arm in [x for x in EXACT + ["cpu_numpy", "cpu_solve", "cpu_sm"] if x in set(df.arm) and x != ref]:
        A = df[df.arm == arm].set_index(K + ["run_id"])
        common = A.index.intersection(R_.index)
        if not len(common):
            continue
        x, y = A.loc[common], R_.loc[common]
        same = ((x.best_cost - y.best_cost).abs() <= 1e-9 * y.best_cost.abs().clip(lower=1)) & (x.best_n_caps == y.best_n_caps) \
            & (x.n_fitness_evals == y.n_fitness_evals) & (x.success == y.success)
        idn.append(dict(arm=arm, reference=ref, paired_runs=len(common), identical=int(same.sum()), rate=float(same.mean())))
        tmp = pd.DataFrame({"r": y.total_wall_time_s.values, "a": x.total_wall_time_s.values}, index=common).reset_index()
        for key, g in tmp.groupby(K):
            spd.append(dict(zip(K, key), arm=arm, reference=ref, paired_runs=len(g), speedup=g.r.sum() / g.a.sum(),
                            ref_mean_s=g.r.mean(), arm_mean_s=g.a.mean()))
pd.DataFrame(idn).to_csv(os.path.join(DST, "identity.csv"), index=False)
SP = pd.DataFrame(spd)
SP.to_csv(os.path.join(DST, "speedup.csv"), index=False)

# ---- Fisher tests of the search arms against fast
from scipy.stats import fisher_exact
fis = []
for key, g in df[df.source == "gpu"].groupby(K):
    f = g[g.arm == "fast"]
    if not len(f):
        continue
    for arm in ("pso_polish", "ga", "ga_polish"):
        x = g[g.arm == arm]
        if len(x):
            p = fisher_exact([[int(x.success.sum()), len(x) - int(x.success.sum())], [int(f.success.sum()), len(f) - int(f.success.sum())]])[1]
            fis.append(dict(zip(K, key), arm=arm, succ=f"{int(x.success.sum())}/{len(x)}", fast=f"{int(f.success.sum())}/{len(f)}", p=p,
                            tps_arm=tps(x), tps_fast=tps(f), best_arm=x.best_cost.mean(), best_fast=f.best_cost.mean()))
pd.DataFrame(fis).to_csv(os.path.join(DST, "fisher.csv"), index=False)

with open(os.path.join(DST, "summary.md"), "w") as fh:
    fh.write(f"# GPUAll comparative study ({cfg})\n\nsource: `{OUT}`" + (f" + CPU `{a.cpu}`" if a.cpu else "") + "\n\n")
    fh.write("## Trajectory identity of the exact arms\n\n" + (pd.DataFrame(idn).to_markdown(index=False) if idn else "(no paired references)") + "\n\n")
    if len(SP):
        fh.write("## Paired speed-up (sum of reference wall time / sum of arm wall time)\n\n" +
                 SP.pivot_table(index=["pdn", "swarm", "arm"], columns="reference", values="speedup", aggfunc="median").round(2).to_markdown() + "\n\n")
    fh.write("## Search arms vs fast (Fisher exact test on successes)\n\n" + (pd.DataFrame(fis).round(4).to_markdown(index=False) if fis else "-") + "\n\n")
    fh.write("## All cells\n\n" + S.round(5).to_markdown(index=False) + "\n")

# ---- plots: success heat map + time per success, per pdn
for pdn, g in S.groupby("pdn"):
    cells = g.assign(cell=g.rank_band + "/" + g.threshold_level)
    piv = cells.pivot_table(index="arm", columns=["swarm", "cell"], values="success_rate")
    if piv.empty:
        continue
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * piv.shape[1]), 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xticks(range(piv.shape[1]), [f"{s}\n{c}" for s, c in piv.columns], fontsize=7, rotation=90)
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=6)
    fig.colorbar(im, ax=ax, label="success rate")
    ax.set_title(f"{pdn}: success rate per arm (swarm / band / level)")
    fig.tight_layout()
    fig.savefig(os.path.join(DST, "plots", f"{pdn}_success.png"), dpi=150)
    plt.close(fig)
    t = cells.pivot_table(index=["swarm", "cell"], columns="arm", values="time_per_success_s").replace(np.inf, np.nan)
    if t.notna().any().any():
        ax = t.plot.bar(logy=True, figsize=(max(7, 0.5 * len(t)), 4))
        ax.set_ylabel("time per success (s)")
        ax.set_title(f"{pdn}: time per success")
        plt.tight_layout()
        plt.savefig(os.path.join(DST, "plots", f"{pdn}_time_per_success.png"), dpi=150)
        plt.close()

# ---- overlay: bare vs best placement of every arm, all observation ports (numpy, CPU)
if a.overlay:
    import allcpu_pdns as R
    for pdn in sorted(df[df.source == "gpu"].pdn.unique()):
        cell = R.Cell(pdn, load_problem=True)
        core, prob = cell.core, cell.prob
        d, f = core.library()["d"], core.library()["freq"]
        zb = np.abs(np.einsum("fkk->fk", np.linalg.inv(prob.y)[:, :prob.n_obs, :prob.n_obs]))
        for (b, lv), g in df[(df.source == "gpu") & (df.pdn == pdn)].groupby(["rank_band", "threshold_level"]):
            fig, axs = plt.subplots(1, prob.n_obs, figsize=(5.5 * prob.n_obs, 4), squeeze=False)
            for k in range(prob.n_obs):
                axs[0][k].loglog(f, zb[:, k], color="0.4", lw=1.6, label="bare")
            for arm, x in g.groupby("arm"):
                r = x.sort_values(["success", "best_cost"], ascending=[False, True]).iloc[0]
                if not r.placement:
                    continue
                A = prob.y.copy()
                for pl in r.placement:
                    A[:, pl["pad"], pl["pad"]] += d[pl["model"]]
                z = np.abs(np.einsum("fkk->fk", np.linalg.inv(A)[:, :prob.n_obs, :prob.n_obs]))
                for k in range(prob.n_obs):
                    axs[0][k].loglog(f, z[:, k], lw=1.1, label=f"{arm} ({len(r.placement)} caps)")
            for k in range(prob.n_obs):
                axs[0][k].set_title(f"{pdn} obs {k + 1}, opt. band {b}, {lv}")
                axs[0][k].grid(True, which="both", alpha=.3)
                axs[0][k].set_xlabel("Hz")
            axs[0][0].set_ylabel("|Z_kk| (ohm)")
            axs[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(DST, "plots", f"{pdn}_{b}_{lv}_impedance_overlay.png"), dpi=130)
            plt.close(fig)
print(open(os.path.join(DST, "summary.md")).read()[:4000])
print("written to", DST)
