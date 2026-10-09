"""
Experiment 3 -- incremental (rank-1) updates of the stored Woodbury inverse.

Woodbury: Z11 = Z0[t,t] - Z0[t,P] Minv Z0[P,t],  M = diag(1/d_P) + Z0[P,P]   (one k x k system per frequency).
If a particle keeps its pads (same port vector, slot by slot) and only swaps the capacitor MODEL on s pads, then M changes only on
s diagonal entries, so the stored inverse follows from s Sherman-Morrison steps
    Minv <- Minv - delta/(1 + delta*Minv[j,j]) * Minv[:,j] Minv[j,:],   delta = 1/d_new - 1/d_old
at O(s k^2) per frequency instead of the O(k^3) inverse. Per-particle state (ports, models, Minv (F,k,k)) is kept for the whole
stage. A particle is ELIGIBLE when its ports are unchanged and s <= max(1, S_MAX_FRAC*k) and fewer than R_MAX update steps have
accumulated since its last full inverse (drift control). Everything else is recomputed (batched inverse). A tiny denominator
forces a recompute too.

Tie-safety: the updated cost of a configuration differs from the full-solve cost by ~1e-15, which would flip `cost < pbest` for
a REPEATED configuration and change pbest's position. Like GPUBench2's cache, a configuration's cost is therefore computed once per
run and reused (a deterministic function of the configuration, as in the baseline). The cache-only arm (wb_cache) is the fair
reference for the speed-up; the uncached woodbury_batch arm is shown too.

Arms: woodbury_batch, wb_cache (Woodbury + cache), incremental (rank-1 updates + cache, timed), incr_check (same algorithm,
additionally compares every computed cost with the exact Woodbury cost: max deviation; NOT used for speed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE

EXPERIMENT = "incremental"
S_MAX_FRAC = float(os.environ.get("S_MAX_FRAC", "0.5"))
R_MAX = int(os.environ.get("R_MAX", "8"))
DEN_TOL = 1e-8


class IncrementalEvaluator:
    sequential = False

    def __init__(self, prob, check=False, name="incremental"):
        self.prob, self.check, self.name = prob, check, name
        self.reset()

    def reset(self):
        self.state = None
        self.cache = {}
        self.n_hits = 0
        self.n_computed = 0
        self.n_evals = self.n_elig = self.n_zero = self.n_steps = self.n_full = self.n_forced = 0
        self.n_evals_later = self.n_elig_later = 0
        self.max_dev = 0.0

    def begin_stage(self, n_caps):
        self.state = None

    def _full_inverse(self, mt, pt):
        """Minv for configurations (g,k) -> (g,F,k,k), singular flag (g,)"""
        prob = self.prob
        g, k = pt.shape
        out, sing = [], []
        ch = prob.chunk(prob.F, k)
        for s in range(0, g, ch):
            M, _, _, _ = prob.gather(mt[s:s + ch], pt[s:s + ch])
            inv, info = torch.linalg.inv_ex(M)
            out.append(inv)
            sing.append((info != 0).any(dim=1))
        return torch.cat(out), torch.cat(sing)

    def evaluate(self, models, ports, pbest_val):
        prob = self.prob
        P, k = models.shape
        mt = torch.from_numpy(models).to(DEVICE)
        pt = torch.from_numpy(ports).to(DEVICE)
        cost_np = np.empty(P)
        keys = [tuple(sorted(zip(ports[i].tolist(), models[i].tolist()))) for i in range(P)]
        todo = np.ones(P, bool)
        for i in range(P):
            h = self.cache.get(keys[i])
            if h is not None:
                cost_np[i] = h
                todo[i] = False
        self.n_hits += int((~todo).sum())
        elig = np.zeros(P, bool)
        diff = None
        if self.state is not None:
            st = self.state
            diff = models != st["models"]
            s = diff.sum(1)
            smax = max(1, int(S_MAX_FRAC * k))
            elig = (ports == st["ports"]).all(1) & (s <= smax) & (st["nupd"] + s <= R_MAX) & st["valid"] & todo
        else:
            self.state = {"Minv": torch.empty(P, prob.F, k, k, dtype=torch.complex128, device=DEVICE),
                          "models": models.copy(), "ports": ports.copy(), "nupd": np.zeros(P, int), "valid": np.ones(P, bool)}
            st = self.state
        full = todo & ~elig
        # ---- rank-1 steps on the eligible particles
        forced = np.zeros(P, bool)
        if elig.any():
            ie = np.nonzero(elig)[0]
            pos = {int(i): np.nonzero(diff[i])[0] for i in ie}
            smax_i = max(len(v) for v in pos.values())
            for t in range(smax_i):
                pi = np.array([i for i in ie if len(pos[int(i)]) > t])
                if len(pi) == 0:
                    continue
                jj = np.array([pos[int(i)][t] for i in pi])
                pit = torch.from_numpy(pi).to(DEVICE)
                jt = torch.from_numpy(jj).to(DEVICE)
                old = torch.from_numpy(st["models"][pi, jj]).to(DEVICE)
                new = torch.from_numpy(models[pi, jj]).to(DEVICE)
                delta = prob.DINV[new] - prob.DINV[old]                                  # (g,F)
                Mg = st["Minv"][pit]                                                     # (g,F,k,k)
                ar = torch.arange(len(pi), device=DEVICE)
                u = Mg[ar, :, :, jt]                                                      # (g,F,k)  column j
                v = Mg[ar, :, jt, :]                                                      # (g,F,k)  row j
                den = 1.0 + delta * u[ar, :, jt]                                          # (g,F)
                small = (den.abs() < DEN_TOL).any(dim=1) | ~torch.isfinite(den).all(dim=1)
                den = torch.where(den.abs() < DEN_TOL, torch.ones_like(den), den)
                Mg = Mg - (delta / den).unsqueeze(-1).unsqueeze(-1) * (u.unsqueeze(-1) * v.unsqueeze(-2))
                st["Minv"][pit] = Mg
                if small.any():
                    forced[pi[small.cpu().numpy()]] = True
                st["models"][pi, jj] = models[pi, jj]
                self.n_steps += len(pi)
            st["nupd"][ie] += np.array([len(pos[int(i)]) for i in ie])
            self.n_zero += int(sum(1 for i in ie if len(pos[int(i)]) == 0))
        self.n_forced += int(forced.sum())
        full = full | forced
        # ---- full recompute for the rest
        if full.any():
            fi = np.nonzero(full)[0]
            fit = torch.from_numpy(fi).to(DEVICE)
            inv, sing = self._full_inverse(mt[fit], pt[fit])
            st["Minv"][fit] = inv
            st["models"][fi] = models[fi]
            st["ports"][fi] = ports[fi]
            st["nupd"][fi] = 0
            st["valid"][fi] = ~sing.cpu().numpy()
            self.n_full += len(fi)
        # ---- cost from the stored inverse (only the configurations not found in the cache)
        ti = np.nonzero(todo)[0]
        if len(ti):
            costs = []
            ch = prob.chunk(prob.F, k)
            for s0 in range(0, len(ti), ch):
                sub = ti[s0:s0 + ch]
                subt = torch.from_numpy(sub).to(DEVICE)
                _, zc, zr, z00 = prob.gather(mt[subt], pt[subt])
                y = st["Minv"][subt] @ zc
                val = z00.unsqueeze(0) - (zr @ y)[..., 0, 0]
                pk = val.abs().max(dim=1)[0]
                costs.append(torch.where(torch.isfinite(pk), pk, torch.full_like(pk, 1e200)))
            cost = torch.cat(costs)
            tit = torch.from_numpy(ti).to(DEVICE)
            cost[torch.from_numpy(~st["valid"][ti]).to(DEVICE)] = 1e200
            if self.check:
                ref, _ = prob.wb_peak(mt[tit], pt[tit])
                ok = torch.from_numpy(st["valid"][ti]).to(DEVICE) & (ref < 1e199)
                if ok.any():
                    self.max_dev = max(self.max_dev, ((cost - ref).abs() / ref.abs().clamp_min(1e-300))[ok].max().item())
            cn = cost.cpu().numpy()
            cost_np[ti] = cn
            for i, v in zip(ti, cn):
                self.cache[keys[i]] = float(v)
        # models/ports of the state must follow this call even for ineligible-but-valid ones (already set above)
        n_e = int(elig.sum())
        self.n_evals += P
        self.n_computed += len(ti)
        self.n_elig += n_e
        if diff is not None:
            self.n_evals_later += P
            self.n_elig_later += n_e
        return cost_np

    def stats(self):
        return {"gpu_full_evals": self.n_computed, "gpu_screen_evals": 0, "cache_hits": self.n_hits, "pruned": 0, "subset_size": 0,
                "elig_evals": self.n_elig, "elig_fraction": self.n_elig / max(1, self.n_evals), "elig_fraction_of_computed": self.n_elig / max(1, self.n_computed),
                "elig_fraction_after_first_iter": self.n_elig_later / max(1, self.n_evals_later),
                "zero_change_evals": self.n_zero, "rank1_steps": self.n_steps, "full_recomputes": self.n_full,
                "forced_recomputes": self.n_forced, "max_rel_cost_dev": self.max_dev if self.check else None}


def main():
    run = C.Run(EXPERIMENT, __file__, {"S_MAX_FRAC": S_MAX_FRAC, "R_MAX": R_MAX, "DEN_TOL": DEN_TOL})
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    arms = {"woodbury_batch": C.SwarmEvaluator(prob, False, False), "wb_cache": C.SwarmEvaluator(prob, True, False, name="wb_cache"), "incremental": IncrementalEvaluator(prob),
            "incr_check": IncrementalEvaluator(prob, check=True, name="incr_check")}
    C.warmup(prob, list(arms.values()))
    base = C.load_baseline()
    for thr in S["TARGETS"]:
        for name, ev in arms.items():
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", name, thr, r)
                try:
                    rec = C.run_pso(prob, ev, [thr], r, label=name)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(name, thr, r, e)
                    continue
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, base.get((round(thr, 6), r)))
                run.add(rec)
                run.result_line(rec, match=rec["traj_match"], elig=f"{rec.get('elig_fraction', 0):.3f}")
    run.summarise(extra_cols=("elig_fraction", "elig_fraction_after_first_iter", "rank1_steps", "full_recomputes", "forced_recomputes",
                              "max_rel_cost_dev", "traj_match", "traj_match_strict"))
    import pandas as pd
    df = pd.DataFrame(run.records)
    t = df.groupby("method").total_time_s.sum()
    lines = []
    for ref in ("woodbury_batch", "wb_cache"):
        if ref in t and "incremental" in t:
            lines.append(f"total_time {ref} = {t[ref]:.2f}s incremental = {t['incremental']:.2f}s "
                         f"speedup_incremental_over_{ref} = {t[ref] / t['incremental']:.3f}")
    ic = df[df.method == "incr_check"]
    if len(ic):
        lines.append(f"max_rel_cost_dev_incremental_vs_exact_woodbury = {pd.to_numeric(ic.max_rel_cost_dev).max():.3e}")
        lines.append(f"eligible_fraction_of_evaluations = {ic.elig_evals.sum() / ic.n_evals.sum():.4f} (of all swarm evaluations incl. cache hits)")
        lines.append(f"eligible_fraction_of_computed = {pd.to_numeric(ic.elig_fraction_of_computed).mean():.4f} (mean over runs; configurations not served by the cache)")
        lines.append(f"eligible_fraction_after_first_iteration = {pd.to_numeric(ic.elig_fraction_after_first_iter).mean():.4f} (mean over runs)")
    inc = df[df.method == "incremental"]
    if len(inc):
        a = df[df.method == "woodbury_batch"].set_index(["threshold", "run"])
        b = inc.set_index(["threshold", "run"])
        j = a.join(b, lsuffix="_w", rsuffix="_i", how="inner")
        if len(j):
            lines.append(f"incremental_vs_woodbury_batch: max_rel_best_minz_dev = {((j.best_minz_w - j.best_minz_i).abs() / j.best_minz_w.abs()).max():.3e}"
                         f" | identical_evals_and_caps_rate = {((j.n_evals_w == j.n_evals_i) & (j.caps_at_target_w.fillna(-1) == j.caps_at_target_i.fillna(-1))).mean():.4f}")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
    plots(run)
    run.finish()


def plots(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(run.records)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df[df.method != "incr_check"].groupby("method"):
        gg = g.groupby("threshold").total_time_s
        ax[0].errorbar(gg.mean().index, gg.mean().values, yerr=[C.ci95(x) for _, x in gg], marker="o", capsize=3, label=m)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("target |Z11| (ohm)")
    ax[0].set_ylabel("wall time per run (s)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=.3)
    ic = df[df.method == "incremental"]
    if len(ic):
        g = ic.groupby("threshold").elig_fraction.mean()
        ax[1].bar([str(x) for x in g.index], g.values)
    ax[1].set_xlabel("target |Z11| (ohm)")
    ax[1].set_ylabel("fraction of evaluations eligible for rank-1 update")
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "time_and_eligibility.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
