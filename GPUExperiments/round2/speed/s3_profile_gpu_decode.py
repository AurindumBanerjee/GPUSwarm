"""
Speed S3 -- where does the time of 'fast' go now, and does moving decoding to the GPU help (EXACTLY)?

Profile: every PSO iteration of 'fast' is split into
    decode     particle -> (models, ports) including the Python port-collision loop (FIX 12)
    evaluate   cache lookup + pruning screen + Woodbury solve + host copies
    other      velocity/position update, bookkeeping, NumPy RNG
New arm 'fast_gpudecode': identical NumPy RNG and PSO, but decoding is vectorised on the GPU:
    floor(frac*K) on tensors, and port resolution done for all particles at once, looping only over the k slots with a candidate
    table [p, p+1, p-1, p+2, p-2, ...] -- exactly the FIX 12 search order.
The decoder is checked against the reference on random particles at start-up (must be 0 mismatches) and the trajectories of
both arms must match exactly.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync

EXPERIMENT = "s3_profile_gpu_decode"
DATASET = os.environ.get("DATASET", "ref21")


class GpuDecoder:
    def __init__(self, prob):
        self.prob = prob
        n = prob.n
        cand = np.full((n, 2 * n - 1), -1, np.int64)
        for p0 in range(n):
            seq = [p0]
            for off in range(1, n):
                seq += [p0 + off, p0 - off]
            cand[p0] = [c if 0 <= c < n else -1 for c in seq]
        self.cand = torch.tensor(cand, device=DEVICE)
        self.valid = torch.tensor(prob.valid_ports, device=DEVICE)

    def __call__(self, particles, n_caps):
        prob = self.prob
        x = torch.from_numpy(particles).to(DEVICE)
        mi = (x[:, :n_caps] * prob.C).floor().long().clamp(0, prob.C - 1)
        pi = (x[:, n_caps:] * len(prob.valid_ports)).floor().long().clamp(0, len(prob.valid_ports) - 1)
        ports = self.valid[pi]
        P = ports.shape[0]
        used = torch.zeros(P, prob.n, dtype=torch.bool, device=DEVICE)
        used[:, prob.target_port] = True
        ar = torch.arange(P, device=DEVICE)
        for i in range(n_caps):
            c = self.cand[ports[:, i]]
            free = (c >= 0) & ~used.gather(1, c.clamp(min=0))
            has = free.any(1)
            first = free.int().argmax(1)
            new = torch.where(has, c[ar, first], ports[:, i])
            ports[:, i] = new
            used[ar, new] = True
        return mi.cpu().numpy(), ports.cpu().numpy()


class Profiled:
    """Wraps an evaluator and the problem's decode_swarm to accumulate time per phase (synchronised)."""

    def __init__(self, prob, ev, decoder=None):
        self.prob, self.ev, self.decoder = prob, ev, decoder
        self.sequential, self.name = False, ev.name
        self.orig = prob.decode_swarm

    def reset(self):
        self.ev.reset()
        self.t_dec = self.t_eval = 0.0

    def begin_stage(self, n):
        self.ev.begin_stage(n)

    def install(self):
        def dec(particles, n_caps):
            cuda_sync()
            t0 = time.perf_counter()
            out = self.decoder(particles, n_caps) if self.decoder else self.orig(particles, n_caps)
            cuda_sync()
            self.t_dec += time.perf_counter() - t0
            return out
        self.prob.decode_swarm = dec

    def uninstall(self):
        self.prob.decode_swarm = self.orig

    def evaluate(self, M, Pp, pbest):
        cuda_sync()
        t0 = time.perf_counter()
        c = self.ev.evaluate(M, Pp, pbest)
        cuda_sync()
        self.t_eval += time.perf_counter() - t0
        return c

    def stats(self):
        s = self.ev.stats()
        s.update({"t_decode_s": self.t_dec, "t_evaluate_s": self.t_eval})
        return s


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    gdec = GpuDecoder(prob)
    rng = np.random.default_rng(4)
    mism = 0
    for _ in range(300):
        n = int(rng.integers(1, min(S["MAX_CAPS"], len(prob.valid_ports)) + 3))
        x = rng.random((6, 2 * n))
        a = prob.decode_swarm(x, n)
        b = gdec(x, n)
        mism += int(not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])))
    run.lg.info("STAGE | GPU decoder mismatches vs reference: %d / 300", mism)
    assert mism == 0, "GPU decoder differs from the reference"
    arms = {"fast": Profiled(prob, C.SwarmEvaluator(prob, True, True)),
            "fast_gpudecode": Profiled(prob, C.SwarmEvaluator(prob, True, True), gdec)}
    C.warmup(prob, [a.ev for a in arms.values()])
    targets = R.targets_for(info)
    recs = []
    for thr in targets:
        for name, a in arms.items():
            a.install()
            try:
                recs += R.loop_methods(run, prob, {name: a}, [thr], S["NUM_RUNS"], reference="fast")
            finally:
                a.uninstall()
    import pandas as pd
    df = pd.DataFrame(recs)
    df["t_other_s"] = df.total_time_s - df.t_decode_s - df.t_evaluate_s
    ref = df[df.method == "fast"].set_index(["threshold", "run"])
    df["match"] = [abs(r.best_minz - ref.best_minz[(r.threshold, r.run)]) <= 1e-9 * max(1, r.best_minz) and
                   r.n_evals == ref.n_evals[(r.threshold, r.run)] for r in df.itertuples()]
    lines = []
    for (thr, m), g in df.groupby(["threshold", "method"]):
        tot = g.total_time_s.sum()
        lines.append(f"target {thr:g} {m}: total {g.total_time_s.mean():.3f}s/run | decode {100 * g.t_decode_s.sum() / tot:.1f}% | "
                     f"evaluate {100 * g.t_evaluate_s.sum() / tot:.1f}% | other {100 * g.t_other_s.sum() / tot:.1f}% | "
                     f"exact match vs fast {g.match.mean():.3f}")
    for thr, g in df.groupby("threshold"):
        a, b = g[g.method == "fast"].total_time_s.mean(), g[g.method == "fast_gpudecode"].total_time_s.mean()
        lines.append(f"target {thr:g}: fast_gpudecode speed-up over fast = {a / b:.2f}x")
    df.drop(columns=["stage_best", "crossings"], errors="ignore").to_csv(os.path.join(run.run_dir, "summary", "profile.csv"), index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = df.groupby(["method", "threshold"])[["t_decode_s", "t_evaluate_s", "t_other_s"]].mean()
    agg.plot.bar(stacked=True, figsize=(9, 4)).set_ylabel("mean seconds per run")
    plt.tight_layout()
    plt.savefig(os.path.join(run.run_dir, "plots", "time_breakdown.png"), dpi=150)
    plt.close()
    run.summarise(extra_cols=("t_decode_s", "t_evaluate_s", "cache_hits", "pruned", "traj_match"))
    R.write_lines(run, "profile_summary.txt", lines)
    R.plot_best_runs(prob, [r for r in recs if r["method"] == "fast_gpudecode"], os.path.join(run.run_dir, "plots"))
    run.finish(notes={"decoder_mismatches": mism})


if __name__ == "__main__":
    main()
