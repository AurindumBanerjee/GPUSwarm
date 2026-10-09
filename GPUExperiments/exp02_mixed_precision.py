"""
Experiment 2 -- mixed precision.

The k x k Woodbury systems are factorised/solved in complex64 (LU on the GPU's fast FP32 units), then refined in complex128:
    x0 = solve64(M, z);  repeat REFINE_STEPS:  r = z - M x (complex128);  x += solve64(M, r)
Entries whose final relative residual exceeds RES_TOL (ill-conditioned frequencies) are re-solved in complex128, so the cost
keeps double-precision accuracy. Measured: max relative cost deviation vs the complex128 baseline (random configurations of
every size AND along the PSO trajectories), fraction of fallbacks, trajectory match vs the numpy baseline, time vs complex128.

Arms: c128_batch (woodbury_batch), mixed_batch, c128_fast (cache+prune), mixed_fast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE

EXPERIMENT = "mixed_precision"
REFINE_STEPS = int(os.environ.get("REFINE_STEPS", "2"))
RES_TOL = float(os.environ.get("RES_TOL", "1e-12"))


class MixedPeak:
    """peak_fn with complex64 LU + complex128 iterative refinement and a complex128 fallback for badly conditioned entries."""

    def __init__(self, prob):
        self.prob = prob
        self.n_sys = self.n_fallback = 0

    def reset_counts(self):
        self.n_sys = self.n_fallback = 0

    def __call__(self, models_t, ports_t, fidx=None):
        prob = self.prob
        B, k = ports_t.shape
        Fp = prob.F if fidx is None else len(fidx)
        ch = prob.chunk(Fp, k)
        peaks, argm = [], []
        for s in range(0, B, ch):
            M, zc, zr, z00 = prob.gather(models_t[s:s + ch], ports_t[s:s + ch], fidx)
            LU, piv, info = torch.linalg.lu_factor_ex(M.to(torch.complex64))
            x = torch.linalg.lu_solve(LU, piv, zc.to(torch.complex64)).to(torch.complex128)
            for _ in range(REFINE_STEPS):
                r = zc - M @ x
                x = x + torch.linalg.lu_solve(LU, piv, r.to(torch.complex64)).to(torch.complex128)
            r = zc - M @ x
            res = r.flatten(2).norm(dim=2) / zc.flatten(2).norm(dim=2).clamp_min(1e-300)       # (b,F')
            bad = (res > RES_TOL) | ~torch.isfinite(res) | (info != 0)
            self.n_sys += bad.numel()
            sing = torch.zeros(bad.shape[0], dtype=torch.bool, device=DEVICE)
            bi = bad.nonzero()
            if len(bi):
                self.n_fallback += len(bi)
                x2, info2 = torch.linalg.solve_ex(M[bi[:, 0], bi[:, 1]], zc[bi[:, 0], bi[:, 1]])
                x = x.clone()
                x[bi[:, 0], bi[:, 1]] = x2
                sing[bi[:, 0][info2 != 0]] = True
            val = z00.unsqueeze(0) - (zr @ x)[..., 0, 0]
            pk, am = val.abs().max(dim=1)
            peaks.append(torch.where(sing | ~torch.isfinite(pk), torch.full_like(pk, 1e200), pk))
            argm.append(am)
        peak, am = torch.cat(peaks), torch.cat(argm)
        if fidx is not None:
            am = fidx[am]
        return peak, am


def accuracy_test(prob, mixed, n_per_k=40):
    """Max relative deviation of the mixed peak from the complex128 Woodbury peak on random configurations of size 1..MAX_CAPS."""
    rng = np.random.default_rng(2024)
    worst, rows = 0.0, []
    for k in range(1, min(S["MAX_CAPS"], len(prob.valid_ports)) + 1):
        Ms, Ps = [], []
        for _ in range(n_per_k):
            m, p = prob.decode_particle(rng.random(2 * k), k)
            m, p = prob.resolve_adjacent_ports(m, p)
            Ms.append(m)
            Ps.append(p)
        mt = torch.from_numpy(np.array(Ms)).to(DEVICE)
        pt = torch.from_numpy(np.array(Ps)).to(DEVICE)
        a, _ = prob.wb_peak(mt, pt)
        b, _ = mixed(mt, pt)
        dev = ((a - b).abs() / a.abs().clamp_min(1e-300)).max().item()
        rows.append((k, dev))
        worst = max(worst, dev)
    return worst, rows


def main():
    run = C.Run(EXPERIMENT, __file__, {"REFINE_STEPS": REFINE_STEPS, "RES_TOL": RES_TOL})
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    mp = MixedPeak(prob)
    arms = {"c128_batch": C.SwarmEvaluator(prob, False, False, name="c128_batch"),
            "mixed_batch": C.SwarmEvaluator(prob, False, False, peak_fn=mp, name="mixed_batch"),
            "c128_fast": C.SwarmEvaluator(prob, True, True, name="c128_fast"),
            "mixed_fast": C.SwarmEvaluator(prob, True, True, peak_fn=mp, name="mixed_fast")}
    worst, rows = accuracy_test(prob, mp)
    fb_random = mp.n_fallback / max(1, mp.n_sys)
    run.lg.info("STAGE | accuracy (random configs) | max_rel_dev=%.3e | fallback_frac=%.3e", worst, fb_random)
    with open(os.path.join(run.run_dir, "summary", "accuracy.txt"), "w") as fh:
        fh.write(f"random-configuration test (40 per size), complex64 solve + {REFINE_STEPS} refinement steps vs complex128\n")
        for k, d in rows:
            fh.write(f"k={k} | max_rel_cost_dev={d:.3e}\n")
        fh.write(f"max_rel_cost_dev = {worst:.3e}\nfallback_fraction_of_systems = {fb_random:.3e}\n")
    C.warmup(prob, list(arms.values()))
    base = C.load_baseline()
    for thr in S["TARGETS"]:
        for name, ev in arms.items():
            for r in range(1, S["NUM_RUNS"] + 1):
                mp.reset_counts()
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", name, thr, r)
                try:
                    rec = C.run_pso(prob, ev, [thr], r, label=name)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(name, thr, r, e)
                    continue
                rec["fallback_fraction"] = mp.n_fallback / max(1, mp.n_sys) if name.startswith("mixed") else 0.0
                ref = base.get((round(thr, 6), r))
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, ref)
                if ref is not None:
                    rec["cost_dev_vs_baseline"] = abs(rec["best_minz"] - ref["best_minz"]) / max(abs(ref["best_minz"]), 1e-30)
                run.add(rec)
                run.result_line(rec, match=rec["traj_match"], fb=f"{rec['fallback_fraction']:.2e}")
    run.summarise(extra_cols=("fallback_fraction", "cost_dev_vs_baseline", "traj_match", "traj_match_strict", "gpu_full_evals"))
    import pandas as pd
    df = pd.DataFrame(run.records)
    lines = []
    for tw in ("batch", "fast"):
        a = df[df.method == f"c128_{tw}"].set_index(["threshold", "run"])
        b = df[df.method == f"mixed_{tw}"].set_index(["threshold", "run"])
        j = a.join(b, lsuffix="_c", rsuffix="_m", how="inner")
        if len(j):
            dev = ((j.best_minz_c - j.best_minz_m).abs() / j.best_minz_c.abs()).max()
            same = ((j.n_evals_c == j.n_evals_m) & (j.caps_at_target_c.fillna(-1) == j.caps_at_target_m.fillna(-1))).mean()
            lines.append(f"{tw}: max_rel_best_minz_dev_mixed_vs_c128 = {dev:.3e} | identical_caps_and_evals_rate = {same:.4f}")
    t = df.groupby("method").total_time_s.sum()
    for tw in ("batch", "fast"):
        if f"c128_{tw}" in t and f"mixed_{tw}" in t:
            lines.append(f"{tw}: total_time c128 = {t[f'c128_{tw}']:.2f}s mixed = {t[f'mixed_{tw}']:.2f}s "
                         f"speedup_mixed_over_c128 = {t[f'c128_{tw}'] / t[f'mixed_{tw}']:.3f}")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write(f"max_rel_cost_dev_random_configs = {worst:.3e}\nfallback_fraction_random = {fb_random:.3e}\n" + "\n".join(lines) + "\n")
    plots(run, rows)
    run.finish(notes={"max_rel_cost_dev": worst})


def plots(run, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(run.records)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df.groupby("method"):
        gg = g.groupby("threshold").total_time_s
        ax[0].errorbar(gg.mean().index, gg.mean().values, yerr=[C.ci95(x) for _, x in gg], marker="o", capsize=3, label=m)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("target |Z11| (ohm)")
    ax[0].set_ylabel("wall time per run (s)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=.3)
    ax[1].semilogy([k for k, _ in rows], [max(d, 1e-18) for _, d in rows], "o-")
    ax[1].axhline(1e-10, color="r", ls="--", label="target 1e-10")
    ax[1].set_xlabel("capacitors k")
    ax[1].set_ylabel("max relative cost deviation vs complex128")
    ax[1].legend()
    ax[1].grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "time_and_accuracy.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
