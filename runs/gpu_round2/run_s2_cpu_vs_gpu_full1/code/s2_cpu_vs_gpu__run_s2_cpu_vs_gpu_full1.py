"""
Speed S2 -- separate the ALGORITHM gain from the HARDWARE gain.

Reviewers will ask how much of the 54-73x comes from the Woodbury reformulation and how much from the GPU. The same three
fitness implementations are run on the CPU (torch, CPU_THREADS threads) and on the GPU:

    numpy           full inverse of the assembled Y per particle
    woodbury_batch  exact batched Woodbury, whole swarm per call
    fast            woodbury_batch + cache + exact pruning

    algorithm gain  = cpu numpy / cpu woodbury_batch       (same hardware, different maths)
    hardware gain   = cpu woodbury_batch / gpu woodbury_batch (same maths, different hardware)
    total           = cpu numpy / gpu fast

Part A: micro-benchmark of one swarm evaluation per arm, for k in BENCH_K and every thread count in CPU_THREADS_LIST.
Part B: full PSO runs of all six arms on the dataset (DATASET, default ref21); cpu numpy is slow, so it gets NUM_RUNS_CPU_NUMPY runs.
All arms give identical trajectories (checked against gpu numpy).
"""
import copy
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

EXPERIMENT = "s2_cpu_vs_gpu"
DATASET = os.environ.get("DATASET", "ref21")
CPU_THREADS = int(os.environ.get("CPU_THREADS", "8"))
CPU_THREADS_LIST = R.env_list("CPU_THREADS_LIST", "1,4,8", int)
BENCH_K = R.env_list("BENCH_K", "5,10,20" if not S["QUICK"] else "3,5", int)
REPS = int(os.environ.get("REPS", "3" if S["QUICK"] else "5"))
NUM_RUNS_CPU_NUMPY = int(os.environ.get("NUM_RUNS_CPU_NUMPY", "2" if S["QUICK"] else "3"))
NUM_RUNS_CPU = int(os.environ.get("NUM_RUNS_CPU", "3" if S["QUICK"] else "5"))      # cpu_woodbury_batch: ~1 s per swarm at k=20 -> keep the run count small
ARMS = R.env_list("ARMS", "gpu_numpy,gpu_woodbury_batch,gpu_fast,cpu_numpy,cpu_woodbury_batch,cpu_fast")


def cpu_copy(prob):
    cp = copy.copy(prob)
    cp.Z0, cp.DINV, cp.D = prob.Z0.cpu(), prob.DINV.cpu(), prob.D.cpu()
    cp.Y = prob.Y.cpu() if prob.Y is not None else None
    return cp


def make_arms(prob, cprob):
    def cpu_peak(mt, pt, fidx=None):
        return cprob.wb_peak(mt.cpu(), pt.cpu(), None if fidx is None else fidx.cpu())
    arms = {"gpu_numpy": C.SeqNumpyEvaluator(prob),
            "gpu_woodbury_batch": C.SwarmEvaluator(prob, False, False, name="gpu_woodbury_batch"),
            "gpu_fast": C.SwarmEvaluator(prob, True, True, name="gpu_fast"),
            "cpu_numpy": C.SeqNumpyEvaluator(cprob),
            "cpu_woodbury_batch": C.SwarmEvaluator(prob, False, False, peak_fn=cpu_peak, name="cpu_woodbury_batch"),
            "cpu_fast": C.SwarmEvaluator(prob, True, True, peak_fn=cpu_peak, name="cpu_fast")}
    for k, v in arms.items():
        v.name = k
    return {k: arms[k] for k in ARMS}


def bench(prob, cprob, run):
    import pandas as pd
    rng = np.random.default_rng(0)
    P = S["N_PARTICLES"]
    rows = []
    for th in CPU_THREADS_LIST:
        torch.set_num_threads(th)
        for k in BENCH_K:
            if k > len(prob.valid_ports):
                continue
            m = rng.integers(0, prob.C, size=(P, k))
            p = np.stack([rng.choice(prob.valid_ports, size=k, replace=False) for _ in range(P)])
            for arm, (pr, kind) in {"gpu_numpy": (prob, "seq"), "gpu_woodbury_batch": (prob, "wb"),
                                    "cpu_numpy": (cprob, "seq"), "cpu_woodbury_batch": (cprob, "wb")}.items():
                if arm.startswith("gpu") and th != CPU_THREADS_LIST[0]:
                    continue                                   # GPU arms do not depend on CPU threads
                dev = pr.Z0.device
                mt, pt = torch.tensor(m, device=dev), torch.tensor(p, device=dev)
                ev = C.SeqNumpyEvaluator(pr)
                ts = []
                for rep in range(REPS + 1):
                    cuda_sync()
                    t0 = time.perf_counter()
                    if kind == "seq":
                        for i in range(P):
                            ev.cost_one(m[i], p[i])
                    else:
                        pr.wb_peak(mt, pt)[0].cpu()
                    cuda_sync()
                    if rep:
                        ts.append(time.perf_counter() - t0)
                rows.append({"threads": th if arm.startswith("cpu") else None, "k": k, "arm": arm, "swarm_eval_s": float(np.median(ts))})
                run.lg.info("BENCH | threads=%s k=%d %s %.5fs", th, k, arm, rows[-1]["swarm_eval_s"])
    torch.set_num_threads(CPU_THREADS)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run.run_dir, "summary", "bench.csv"), index=False)
    lines = []
    g0 = df[df.arm.str.startswith("gpu")].set_index(["k", "arm"]).swarm_eval_s
    for (th, k), g in df[df.arm.str.startswith("cpu")].groupby(["threads", "k"]):
        c = g.set_index("arm").swarm_eval_s
        alg = c["cpu_numpy"] / c["cpu_woodbury_batch"]
        hw = c["cpu_woodbury_batch"] / g0[(k, "gpu_woodbury_batch")]
        lines.append(f"threads={int(th)} k={k}: algorithm gain (cpu numpy/cpu woodbury) {alg:.1f}x | hardware gain "
                     f"(cpu woodbury/gpu woodbury) {hw:.1f}x | gpu numpy/gpu woodbury {g0[(k, 'gpu_numpy')] / g0[(k, 'gpu_woodbury_batch')]:.1f}x")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for (arm, th), g in df.groupby(["arm", df.threads.fillna(0)]):
        lab = arm + (f" ({int(th)} thr)" if th else "")
        ax.semilogy(g.k, 1e3 * g.swarm_eval_s / P, "o-", label=lab)
    ax.set_xlabel("capacitors k")
    ax.set_ylabel("time per particle evaluation (ms)")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "cpu_vs_gpu_bench.png"), dpi=150)
    plt.close(fig)
    return lines


def main():
    torch.set_num_threads(CPU_THREADS)
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "CPU_THREADS": CPU_THREADS, "CPU_THREADS_LIST": CPU_THREADS_LIST,
                                       "BENCH_K": BENCH_K, "NUM_RUNS_CPU_NUMPY": NUM_RUNS_CPU_NUMPY, "NUM_RUNS_CPU": NUM_RUNS_CPU, "ARMS": ARMS})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    cprob = cpu_copy(prob)
    run.lg.info("STAGE | %s", info)
    lines = ["Part A: one swarm evaluation (N_PARTICLES placements)"] + bench(prob, cprob, run) + [""]
    arms = make_arms(prob, cprob)
    rep = C.verify_against_cpu(prob, list(arms.values()))
    run.lg.info("STAGE | verify | %s", rep)
    C.warmup(prob, list(arms.values()))
    targets = R.targets_for(info)
    recs = []
    for name, ev in arms.items():
        n = NUM_RUNS_CPU_NUMPY if name == "cpu_numpy" else NUM_RUNS_CPU if name == "cpu_woodbury_batch" else S["NUM_RUNS"]
        recs += R.loop_methods(run, prob, {name: ev}, targets, n, reference="__none__")
    import pandas as pd
    df = pd.DataFrame(recs)
    ref = df[df.method == "gpu_numpy"].set_index(["threshold", "run"])
    df["match_gpu_numpy"] = [(abs(r.best_minz - ref.best_minz[(r.threshold, r.run)]) <= 1e-9 * max(1, r.best_minz)
                              and r.n_evals == ref.n_evals[(r.threshold, r.run)])      # same best |Z11| AND same logical evaluations
                             if (r.threshold, r.run) in ref.index else None for r in df.itertuples()]
    lines.append("Part B: full PSO, mean total time per run on the runs common to all arms (paired)")
    common = set(df[df.method == "cpu_numpy"].run) if "cpu_numpy" in arms else set(df.run)
    for thr, g in df.groupby("threshold"):
        gg = g[g.run.isin(common)].groupby("method").total_time_s.mean()
        base = gg.get("cpu_numpy", np.nan)
        lines.append(f"target {thr:g}: " + " | ".join(f"{m} {t:.3f}s ({base / t:.1f}x vs cpu numpy)" for m, t in gg.items()))
    lines.append(f"trajectory match vs gpu numpy: {df.match_gpu_numpy.dropna().mean():.3f}")
    run.summarise(extra_cols=("gpu_full_evals", "cache_hits", "pruned"))
    R.write_lines(run, "cpu_vs_gpu_summary.txt", lines)
    R.plot_best_runs(prob, [r for r in recs if r["method"] == "gpu_fast"], os.path.join(run.run_dir, "plots"))
    run.finish(notes={"verify_max_rel": rep})


if __name__ == "__main__":
    main()
