"""
Search quality Q2 -- search a smaller capacitor library.

The PSO picks a model index out of the whole library (3348 parts on the reference set). Many parts are near-duplicates or are
beaten everywhere by another part, so a smaller library could make the search converge faster. The candidate libraries are real
SUBSETS of the full library, so every placement found is still a real part list and its score is exact.

    full        the whole library (reference)
    dominance   drop part a if some part b has |Z_b(f)| <= |Z_a(f)| at every frequency (strictly lower somewhere).
                HEURISTIC: the phase also matters inside a network, so a 'dominated' part can still be useful -- that is
                what this arm measures.
    clusterK    k-means (K = each value in CLUSTER_K) on log|Z(f)| sampled at 64 frequencies; the member nearest to each
                centroid represents the cluster.
All arms run 'fast' with the same paired seeds. Note the discretisation changes with the library size, so trajectories differ.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE

EXPERIMENT = "q2_library_reduction"
DATASET = os.environ.get("DATASET", "ref21")
CLUSTER_K = R.env_list("CLUSTER_K", "50,200" if not S["QUICK"] else "20,40", int)


def dominance_keep(absz):
    """absz (C,F) torch on device. Returns indices of non-dominated parts."""
    Cn = absz.shape[0]
    dominated = torch.zeros(Cn, dtype=torch.bool, device=absz.device)
    ch = max(1, int(2e8 // (Cn * absz.shape[1])))
    for s in range(0, Cn, ch):
        a = absz[s:s + ch]                                         # (c,F)
        le = (absz.unsqueeze(0) <= a.unsqueeze(1)).all(-1)        # (c,C): b <= a everywhere
        lt = (absz.unsqueeze(0) < a.unsqueeze(1)).any(-1)
        d = le & lt
        # exact duplicates: keep the lowest index only
        eq = (absz.unsqueeze(0) == a.unsqueeze(1)).all(-1)
        idx = torch.arange(s, s + a.shape[0], device=absz.device).unsqueeze(1)
        dup = eq & (torch.arange(Cn, device=absz.device).unsqueeze(0) < idx)
        dominated[s:s + a.shape[0]] = (d | dup).any(1)
    return torch.nonzero(~dominated).flatten().cpu().numpy()


def kmeans_keep(feat, K, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.tensor(feat, device=DEVICE)
    K = min(K, X.shape[0])
    cent = X[torch.tensor(rng.choice(X.shape[0], K, replace=False), device=DEVICE)]
    for _ in range(iters):
        lab = torch.cdist(X, cent).argmin(1)
        for j in range(K):
            m = lab == j
            if m.any():
                cent[j] = X[m].mean(0)
    lab = torch.cdist(X, cent).argmin(1)
    keep = []
    for j in range(K):
        idx = torch.nonzero(lab == j).flatten()
        if len(idx):
            keep.append(int(idx[torch.cdist(X[idx], cent[j:j + 1]).argmin()]))
    return np.array(sorted(set(keep)))


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "CLUSTER_K": CLUSTER_K})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    absz = (1.0 / prob.D.abs())                                      # (C,F)
    libs = {"full": np.arange(prob.C), "dominance": dominance_keep(absz)}
    sub = np.linspace(0, prob.F - 1, min(64, prob.F)).astype(int)
    feat = np.log(absz[:, sub].cpu().numpy())
    for K in CLUSTER_K:
        libs[f"cluster{K}"] = kmeans_keep(feat, K)
    lines = [f"library sizes: " + ", ".join(f"{k}={len(v)}" for k, v in libs.items())]
    recs = []
    targets = R.targets_for(info)
    for name, keep in libs.items():
        p2 = C.Problem(y=prob.y_np, d=prob.d_np[keep], freq=prob.freq, label=f"{prob.label}/{name}")
        ev = C.SwarmEvaluator(p2, True, True)
        C.warmup(p2, [ev])
        rs = R.loop_methods(run, p2, {name: ev}, targets, S["NUM_RUNS"], reference="__none__",
                            extra={"library": name, "library_size": int(len(keep))})
        for r in rs:                                                  # library row ids of the placed parts
            if r.get("best_models"):
                r["best_models_full"] = [int(keep[m]) for m in r["best_models"]]
        recs += rs
        R.plot_best_runs(p2, rs, os.path.join(run.run_dir, "plots"), tag=f"{name}_")
    np.savez(os.path.join(run.run_dir, "output", "libraries.npz"), **{k: v for k, v in libs.items()})
    import pandas as pd
    df = pd.DataFrame(recs)
    for thr, g in df.groupby("threshold"):
        f = g[g.method == "full"]
        for name, a in g.groupby("method"):
            p = R.fisher_success(int(a.success.sum()), len(a), int(f.success.sum()), len(f)) if name != "full" else float("nan")
            lines.append(f"target {thr:g} {name:12s} ({int(a.library_size.iloc[0])} parts): success {int(a.success.sum())}/{len(a)} | "
                         f"mean evals {a.n_evals.mean():.0f} | mean time {a.total_time_s.mean():.2f}s | best {a.best_minz.mean():.5f} | "
                         f"caps {a.caps_at_target.dropna().mean():.2f} | Fisher p vs full {p:.3g}")
    run.summarise(extra_cols=("library_size", "cache_hits", "pruned"))
    R.write_lines(run, "library_summary.txt", lines)
    run.finish()


if __name__ == "__main__":
    main()
