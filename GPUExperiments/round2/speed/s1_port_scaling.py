"""
Speed S1 -- how the cost scales with the number of PDN ports N.

Claim under test: full inversion costs O(N^3) per frequency, the Woodbury fitness O(k^3 + k^2) with k = number of capacitors,
independent of N. So the speed-up should GROW with the port count.

Part A (micro-benchmark, no PSO): the 79-port DDR3 PDN is reduced (open-circuit, Z0 sub-block) to N = SCALING_N ports
(observation port + the first N-1 pads). For every N and k in SCALING_K, the mean time of one swarm evaluation (N_PARTICLES random
placements) is measured for
    numpy           full inverse of the assembled Y, one particle at a time (GPUBench2 'numpy')
    woodbury_batch  exact batched Woodbury, the whole swarm in one call
Part B (full PSO): numpy vs fast on ddr21 and ddr79 at targets relative to the bare peak (same fractions for both).
numpy runs are expensive at 79 ports: NUM_RUNS_NUMPY (default 5) runs, fast uses NUM_RUNS.

Outputs: summary/scaling.csv, plots/scaling.png, plots/before_after_*.png, the usual summary files.
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

EXPERIMENT = "s1_port_scaling"
SCALING_N = R.env_list("SCALING_N", "6,11,21,41,79" if not S["QUICK"] else "6,11", int)
SCALING_K = R.env_list("SCALING_K", "5,10,20" if not S["QUICK"] else "3,5", int)
REPS = int(os.environ.get("REPS", "3" if S["QUICK"] else "5"))
DATASETS_B = R.env_list("DATASETS_B", "ddr21,ddr79")
NUM_RUNS_NUMPY = int(os.environ.get("NUM_RUNS_NUMPY", "2" if S["QUICK"] else "5"))


def time_eval(prob, k, method, rng):
    P = S["N_PARTICLES"]
    m = rng.integers(0, prob.C, size=(P, k))
    p = np.stack([rng.choice(prob.valid_ports, size=k, replace=False) for _ in range(P)])
    mt, pt = torch.tensor(m, device=DEVICE), torch.tensor(p, device=DEVICE)
    ev = C.SeqNumpyEvaluator(prob)
    ts = []
    for rep in range(REPS + 1):                                  # first repetition = warm-up
        cuda_sync()
        t0 = time.perf_counter()
        if method == "numpy":
            for i in range(P):
                ev.cost_one(m[i], p[i])
        else:
            prob.wb_peak(mt, pt)[0].cpu()
        cuda_sync()
        if rep:
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def part_a(run):
    full, info = R.load_problem("ddr79")
    Z0 = full.Z0.cpu().numpy()
    rows = []
    rng = np.random.default_rng(0)
    for N in SCALING_N:
        N = min(N, full.n)
        idx = np.arange(N)
        z = Z0[:, idx][:, :, idx]
        prob = C.Problem(y=np.linalg.inv(z), d=full.d_np, freq=full.freq, label=f"ddr_N{N}")
        for k in SCALING_K:
            if k > N - 1:
                continue
            for meth in ("numpy", "woodbury_batch"):
                t = time_eval(prob, k, meth, rng)
                rows.append({"N": N, "k": k, "method": meth, "swarm_eval_s": t, "per_particle_ms": 1e3 * t / S["N_PARTICLES"]})
                run.lg.info("SCALING | N=%d k=%d %s swarm_eval=%.5fs", N, k, meth, t)
    import pandas as pd
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["N", "k"], columns="method", values="swarm_eval_s").reset_index()
    piv["speedup"] = piv["numpy"] / piv["woodbury_batch"]
    piv.to_csv(os.path.join(run.run_dir, "summary", "scaling.csv"), index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (meth, k), g in df.groupby(["method", "k"]):
        ax[0].loglog(g.N, g.per_particle_ms, "o-" if meth == "numpy" else "s--", label=f"{meth}, k={k}")
    for k, g in piv.groupby("k"):
        ax[1].semilogx(g.N, g.speedup, "o-", label=f"k={k}")
    ax[0].set_ylabel("time per particle evaluation (ms)")
    ax[1].set_ylabel("speed-up woodbury_batch / numpy")
    for a in ax:
        a.set_xlabel("PDN ports N")
        a.grid(True, which="both", alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "scaling.png"), dpi=150)
    plt.close(fig)
    return piv


def main():
    run = C.Run(EXPERIMENT, __file__, {"SCALING_N": SCALING_N, "SCALING_K": SCALING_K, "REPS": REPS, "DATASETS_B": DATASETS_B,
                                       "NUM_RUNS_NUMPY": NUM_RUNS_NUMPY, "SP_FMAX": R.SP_FMAX, "LIB_SOURCE": R.LIB_SOURCE})
    C.init_device_log(run)
    piv = part_a(run)
    lines = ["Part A: swarm-evaluation time vs ports (speed-up = numpy / woodbury_batch)", piv.to_string(index=False), ""]
    for name in DATASETS_B:
        prob, info = R.load_problem(name)
        run.lg.info("STAGE | %s | %s", name, info)
        targets = R.targets_for(info)
        arms = {"numpy": C.SeqNumpyEvaluator(prob), "fast": C.SwarmEvaluator(prob, True, True)}
        C.warmup(prob, list(arms.values()))
        recs = R.loop_methods(run, prob, {"numpy": arms["numpy"]}, targets, NUM_RUNS_NUMPY, reference="numpy",
                              label_prefix=f"{name}/", extra={"dataset": name, "ports": prob.n})
        recs += R.loop_methods(run, prob, {"fast": arms["fast"]}, targets, S["NUM_RUNS"], reference="none",
                               label_prefix=f"{name}/", extra={"dataset": name, "ports": prob.n})
        import pandas as pd
        df = pd.DataFrame(recs)
        for thr, g in df.groupby("threshold"):
            a = g[g.method == f"{name}/numpy"]
            b = g[g.method == f"{name}/fast"]
            # paired runs 1..NUM_RUNS_NUMPY: identical trajectories -> compare the same runs
            bb = b[b.run.isin(a.run)]
            sp = a.total_time_s.mean() / bb.total_time_s.mean() if len(a) and len(bb) else float("nan")
            same = int(sum(abs(x - y) < 1e-9 * max(1, x) for x, y in zip(a.sort_values("run").best_minz, bb.sort_values("run").best_minz)))
            lines.append(f"Part B {name} (N={prob.n}) target {thr:.5g}: numpy {a.total_time_s.mean():.3f}s, fast {bb.total_time_s.mean():.3f}s "
                         f"on the same {len(a)} runs -> speed-up {sp:.1f}x; identical best |Z11| in {same}/{len(a)}")
        R.plot_best_runs(prob, recs, os.path.join(run.run_dir, "plots"), tag="")
    run.summarise(extra_cols=("gpu_full_evals", "cache_hits", "pruned"))
    R.write_lines(run, "scaling_summary.txt", lines)
    run.finish()


if __name__ == "__main__":
    main()
