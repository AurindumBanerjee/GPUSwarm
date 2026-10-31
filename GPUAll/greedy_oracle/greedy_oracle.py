#!/usr/bin/env python3
"""
Greedy oracle -- exact best single capacitor to add, over EVERY (part, pad) pair, for the AllCPU PDNs.

With the current placement installed, Zc = (Y + sum d)^-1 at every frequency. Adding part c on pad p changes the observation-port
impedances exactly (rank-1 / Sherman-Morrison, closed form):
        Z_oo'(f) = Z_oo(f) - Zc[o,p](f) Zc[p,o](f) / (Zc[p,p](f) + 1/d_c(f))
so the cost (peak over the ranking band, max over observation ports with the AllCPU weights) of all C x pads candidates is computed
exactly in one vectorised GPU pass per step, without any search. The best is added and Zc updated (exact rank-1 update).

    forward     add the best candidate until the target level is met, the cost stops improving, or the pad budget is used
    refine      (optional, --refine R) R passes of exact 1-swap: for every placed capacitor, remove it and re-add the best (part, pad)
                for the rest -- accepted only if the cost drops
Deterministic: one run per (pdn, band). Records the cost after every capacitor count, the capacitors needed for each level, the wall
time, and the placement; compares with the PSO / GA arms of GPUAll if their results exist (--gpuall-out).

   python greedy_oracle.py [--pdns pdn1,pdn2,pdn3,pdn4] [--bands B50,B1G] [--refine 2] [--max-caps 30] [--gpuall-out ../out_<cfg>]
Output: greedy_oracle/out_<cfg>/{greedy.jsonl, summary.md, plots/<pdn>_<band>_curve.png, plots/<pdn>_<band>_impedance.png}
"""
import os
import sys
import json
import time
import glob
import argparse

H = os.path.dirname(os.path.abspath(__file__))
G_DIR = os.path.dirname(H)
ROOTG = os.path.dirname(G_DIR)
os.environ.setdefault("GPUSWARM_ROOT", ROOTG)
os.environ.setdefault("ALLCPU_TARGETS", os.path.join(G_DIR, "targets_gpuall.json"))
sys.path.insert(0, os.path.join(ROOTG, "AllCPU", "common"))
sys.path.insert(0, G_DIR)
import numpy as np
import torch
import allcpu_pdns as R
import gpuall_eval as G
from gpuall_eval import DEV, CD, sync

ap = argparse.ArgumentParser()
ap.add_argument("--pdns", default=",".join(R.PDN_KEYS))
ap.add_argument("--bands")
ap.add_argument("--refine", type=int, default=2)
ap.add_argument("--max-caps", type=int, default=0, help="0 = pad budget")
ap.add_argument("--model-chunk", type=int, default=int(os.environ.get("MODEL_CHUNK", "2048")))
ap.add_argument("--gpuall-out", default=os.path.join(G_DIR, "out_" + R.CFG))
ap.add_argument("--out", default=os.path.join(H, "out_" + R.CFG))
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)


def log(*x):
    print(time.strftime("[%H:%M:%S]"), *x, flush=True)


class Oracle:
    def __init__(self, ev, band):
        self.ev, self.core = ev, ev.core
        self.fb = torch.from_numpy(np.nonzero(self.core.band_mask(band))[0]).to(DEV)
        self.Yb = ev.Y.index_select(0, self.fb)
        self.Db = ev.D.index_select(1, self.fb)                     # (C,Fb)
        self.DIb = ev.DINV.index_select(1, self.fb)
        self.no = ev.n_obs
        w = ev.weights()
        self.w = None if w is None else w
        self.pads = torch.as_tensor(ev.prob.pads, device=DEV)

    def zc(self, placement):
        A = self.Yb.clone()
        for c, p in placement:
            A[:, p, p] += self.Db[c]
        return torch.linalg.inv(A)                                   # (Fb,N,N)

    def cost_of_Z(self, Z):
        a = torch.diagonal(Z[:, :self.no, :self.no], dim1=1, dim2=2).abs()
        a = a if self.w is None else a * self.w
        return float(a.max())

    def best_add(self, Z, free_pads, model_chunk):
        """exact cost of every (model, pad in free_pads) added to Z: returns (cost, model, pad)."""
        no = self.no
        zoo = torch.diagonal(Z[:, :no, :no], dim1=1, dim2=2)          # (Fb,no)
        best = (np.inf, -1, -1)
        for p in free_pads:
            u = Z[:, :no, p]                                          # (Fb,no)   Zc[o,p]
            v = Z[:, p, :no]                                          # (Fb,no)   Zc[p,o]
            zpp = Z[:, p, p]                                          # (Fb,)
            uv = u * v
            for s in range(0, self.DIb.shape[0], model_chunk):
                den = zpp.unsqueeze(0) + self.DIb[s:s + model_chunk]  # (c,Fb)
                val = zoo.unsqueeze(0) - uv.unsqueeze(0) / den.unsqueeze(-1)    # (c,Fb,no)
                a = val.abs()
                if self.w is not None:
                    a = a * self.w
                cst = a.amax(dim=(1, 2))
                cst = torch.where(torch.isfinite(cst), cst, torch.full_like(cst, 1e200))
                i = int(cst.argmin())
                v_ = float(cst[i])
                if v_ < best[0]:
                    best = (v_, s + i, int(p))
        return best


def run_cell(cell, ev, band, levels, max_caps):
    core, prob = cell.core, cell.prob
    orc = Oracle(ev, band)
    sync()
    t0 = time.perf_counter()
    placement = []
    Z = orc.zc(placement)
    bare = orc.cost_of_Z(Z)
    curve = [(0, bare, 0.0)]
    hit = {}
    cost = bare
    for n in range(1, max_caps + 1):
        used = {p for _, p in placement}
        free = [int(p) for p in prob.pads if int(p) not in used]
        if not free:
            break
        c_new, m, p = orc.best_add(Z, free, a.model_chunk)
        if not c_new < cost * (1 - 1e-12):
            log(f"   {n} caps: no single addition improves {cost:.6g} -- stop")
            break
        placement.append((m, p))
        Z = orc.zc(placement)
        cost = orc.cost_of_Z(Z)
        sync()
        curve.append((n, cost, time.perf_counter() - t0))
        for lv, thr in levels.items():
            if lv not in hit and cost <= thr:
                hit[lv] = {"caps": n, "time_s": time.perf_counter() - t0}
        if levels and all(lv in hit for lv in levels):
            break
    t_fwd = time.perf_counter() - t0
    # ---- exact 1-swap refinement at fixed count
    swaps = 0
    for _ in range(a.refine):
        improved = False
        for j in range(len(placement)):
            rest = placement[:j] + placement[j + 1:]
            Zr = orc.zc(rest)
            used = {p for _, p in rest}
            free = [int(p) for p in prob.pads if int(p) not in used]
            c_new, m, p = orc.best_add(Zr, free, a.model_chunk)
            if c_new < cost * (1 - 1e-12):
                placement = rest[:j] + [(m, p)] + rest[j:]
                cost, improved, swaps = c_new, True, swaps + 1
        if not improved:
            break
    sync()
    total = time.perf_counter() - t0
    for lv, thr in levels.items():                                 # a refined placement may meet a level the forward pass missed
        if lv not in hit and cost <= thr:
            hit[lv] = {"caps": len(placement), "time_s": total, "after_refine": True}
    # exact re-check against AllCPU's own evaluation
    pk = core._orig_peaks_all_bands(prob, placement, "numpy") if len(prob.pads) else None
    check = core.cost_of(pk, band)
    return dict(placement=placement, cost=cost, cost_numpy_check=check, bare=bare, curve=curve, hit=hit,
                time_forward_s=t_fwd, time_total_s=total, swaps=swaps)


def pso_reference(pdn, band):
    """best cost and capacitors at target of the GPUAll arms, if present."""
    out = {}
    for f in glob.glob(os.path.join(a.gpuall_out, pdn, "*", "*", "results_long.jsonl")):
        sw, arm = f.split(os.sep)[-3:-1]
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r["rank_band"] != band:
                    continue
                k = (arm, sw, r["threshold_level"])
                o = out.setdefault(k, {"runs": 0, "succ": 0, "caps": [], "best": [], "time": []})
                o["runs"] += 1
                o["succ"] += int(r["success"])
                o["best"].append(r["best_cost"])
                o["time"].append(r["total_wall_time_s"])
                if r["success"]:
                    o["caps"].append(r["caps_at_target"])
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res_path = os.path.join(a.out, "greedy.jsonl")
    lines = [f"# Greedy oracle ({R.CFG})\n"]
    for key in a.pdns.split(","):
        cell = R.Cell(key)
        core, prob = cell.core, cell.prob
        core._mask_cache.clear()
        ev = G.register(core, prob)
        bands = a.bands.split(",") if a.bands else R.SPEC[key]["opt_bands"]
        for b in bands:
            try:
                levels, info = cell.select_band(b)
            except RuntimeError as e:                               # floors not derived yet: run without levels
                log(f"{key} {b}: {e} -- running without target levels")
                core.RANK_BAND, levels = b, {}
                if R.SPEC[key]["ratio"]:
                    core.OBJ_WEIGHTS = None
            mc = a.max_caps or prob.max_caps
            log(f"{key} {b}: N={prob.N} obs={prob.n_obs} pads={len(prob.pads)} models={ev.C} F_band={int(core.band_mask(b).sum())} "
                f"candidates/step={ev.C * len(prob.pads)} levels={ {k: round(v, 6) for k, v in levels.items()} }")
            r = run_cell(cell, ev, b, levels, mc)
            rec = {"pdn": key, "band": b, "cfg": R.CFG, "objective": "ratio" if core.OBJ_WEIGHTS is not None else "ohm",
                   "levels": levels, **{k: v for k, v in r.items() if k != "placement"},
                   "placement": core.placement_records(r["placement"])}
            with open(res_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"   best {r['cost']:.6g} with {len(r['placement'])} caps (numpy check {r['cost_numpy_check']:.6g}), forward "
                f"{r['time_forward_s']:.1f}s, total {r['time_total_s']:.1f}s, swaps {r['swaps']}, levels met: {r['hit']}")
            lines.append(f"\n## {key} {b}\n\nbare {r['bare']:.6g}; greedy best {r['cost']:.6g} with {len(r['placement'])} caps "
                         f"(forward {r['time_forward_s']:.1f}s, with refine {r['time_total_s']:.1f}s, {r['swaps']} swaps); "
                         f"exact numpy re-check {r['cost_numpy_check']:.6g}\n\n")
            lines.append("| level | threshold | greedy caps | greedy time (s) |\n|---|---|---|---|\n")
            for lv, thr in levels.items():
                h = r["hit"].get(lv)
                lines.append(f"| {lv} | {thr:.6g} | {h['caps'] if h else 'not met'} | {h['time_s']:.2f} |\n" if h else f"| {lv} | {thr:.6g} | not met | - |\n")
            ref = pso_reference(key, b)
            if ref:
                lines.append("\nGPUAll arms on the same band (success / runs, mean caps at target, mean best, mean wall time):\n\n"
                             "| arm | swarm | level | success | caps | best | wall (s) |\n|---|---|---|---|---|---|---|\n")
                for (arm, sw, lv), o in sorted(ref.items()):
                    lines.append(f"| {arm} | {sw} | {lv} | {o['succ']}/{o['runs']} | {np.mean(o['caps']) if o['caps'] else float('nan'):.2f} | "
                                 f"{np.mean(o['best']):.6g} | {np.mean(o['time']):.2f} |\n")
            # plots
            cv = np.array(r["curve"])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(cv[:, 0], cv[:, 1], "o-", label="greedy oracle")
            for lv, thr in levels.items():
                ax.axhline(thr, ls="--", lw=.8, color="crimson")
                ax.text(0, thr, lv, fontsize=7, color="crimson", va="bottom")
            ax.set_xlabel("capacitors")
            ax.set_ylabel(f"cost ({rec['objective']})")
            ax.set_title(f"{key} {b}: greedy forward selection")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "plots", f"{key}_{b}_curve.png"), dpi=150)
            plt.close(fig)
            d, f = core.library()["d"], core.library()["freq"]
            A = prob.y.copy()
            for c, p in r["placement"]:
                A[:, p, p] += d[c]
            zb = np.abs(np.einsum("fkk->fk", np.linalg.inv(prob.y)[:, :prob.n_obs, :prob.n_obs]))
            zo = np.abs(np.einsum("fkk->fk", np.linalg.inv(A)[:, :prob.n_obs, :prob.n_obs]))
            fig, axs = plt.subplots(1, prob.n_obs, figsize=(5.5 * prob.n_obs, 4), squeeze=False)
            for k in range(prob.n_obs):
                axs[0][k].loglog(f, zb[:, k], color="0.4", label="bare")
                axs[0][k].loglog(f, zo[:, k], label=f"greedy ({len(r['placement'])} caps)")
                hi = core.band_edges(core.BANDS[b])[1]
                if hi:
                    axs[0][k].axvline(hi, color="crimson", ls=":", lw=1)
                axs[0][k].set_title(f"{key} obs {k + 1}")
                axs[0][k].grid(True, which="both", alpha=.3)
                axs[0][k].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "plots", f"{key}_{b}_impedance.png"), dpi=130)
            plt.close(fig)
    open(os.path.join(a.out, "summary.md"), "w").write("".join(lines))
    log("written", a.out)


if __name__ == "__main__":
    main()
