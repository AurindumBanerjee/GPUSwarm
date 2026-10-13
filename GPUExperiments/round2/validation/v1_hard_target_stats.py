"""
Validation V1 -- make the hard-target (0.04) claims statistically solid.

With 20 runs, 5/20 vs 13/20 (polish) is significant but 0/20 vs 3-4/20 is not. Here the hard target(s) get many more runs:
    fast     NUM_RUNS runs (default 100). Its trajectories are identical to GPU numpy, so success counts ARE numpy's.
    polish   NUM_RUNS runs with the experiment-5 polish (exp05_gradient_polish defaults).
    numpy    NUM_RUNS_NUMPY runs (default 20) -- only to measure its time per run; identity with fast is checked on these runs.
Reported per target: successes with Wilson 95% intervals, Fisher exact tests (polish vs fast), mean caps, and
time per success = total wall time / successes (numpy: its mean time per run x NUM_RUNS / fast successes, valid because the
trajectories are identical).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import gpuexp_common as C
from gpuexp_common import S
import exp05_gradient_polish as P5

EXPERIMENT = "v1_hard_target_stats"
DATASET = os.environ.get("DATASET", "ref21")
HARD_TARGETS = R.env_list("HARD_TARGETS", "0.04")
NUM_RUNS = int(os.environ.get("NUM_RUNS_HARD", "6" if S["QUICK"] else "100"))
NUM_RUNS_NUMPY = int(os.environ.get("NUM_RUNS_NUMPY", "2" if S["QUICK"] else "20"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "HARD_TARGETS": HARD_TARGETS, "NUM_RUNS_HARD": NUM_RUNS,
                                       "NUM_RUNS_NUMPY": NUM_RUNS_NUMPY})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    targets = [float(t) for t in HARD_TARGETS] if info["dataset"] == "ref21" or os.environ.get("TARGET_MODE") == "abs" \
        else [round(float(t) * info["bare_peak"], 6) for t in HARD_TARGETS]
    pol = P5.Polisher(prob)
    fast, npy = C.SwarmEvaluator(prob, True, True), C.SeqNumpyEvaluator(prob)
    C.warmup(prob, [fast, npy])
    pol.polish([0, 1], [int(prob.valid_ports[0]), int(prob.valid_ports[1])])
    recs = R.loop_methods(run, prob, {"numpy": npy}, targets, NUM_RUNS_NUMPY, reference="numpy")
    recs += R.loop_methods(run, prob, {"fast": fast}, targets, NUM_RUNS, reference="__none__")
    recs += R.loop_methods(run, prob, {"polish": fast}, targets, NUM_RUNS, reference="__none__",
                           on_stage_end=lambda name, thr: P5.make_hook(prob, pol, thr))
    import pandas as pd
    df = pd.DataFrame(recs)
    lines = []
    for thr, g in df.groupby("threshold"):
        f, p, n = g[g.method == "fast"], g[g.method == "polish"], g[g.method == "numpy"]
        fn = f[f.run.isin(n.run)].set_index("run").best_minz
        same = sum(abs(fn[r] - b) <= 1e-9 * max(1, b) for r, b in n.set_index("run").best_minz.items())
        lines.append(f"target {thr:g}: numpy vs fast identical best |Z11| on {same}/{len(n)} paired runs")
        for name, a in (("fast", f), ("polish", p)):
            k = int(a.success.sum())
            lo, hi = wilson(k, len(a))
            lines.append(f"  {name:6s}: success {k}/{len(a)} = {k / max(1, len(a)):.3f} (Wilson 95% {lo:.3f}-{hi:.3f}) | "
                         f"caps {a.caps_at_target.dropna().mean():.2f} | mean time {a.total_time_s.mean():.2f}s | "
                         f"time per success {R.time_per_success(a):.2f}s")
        ks = int(f.success.sum())
        tps_np = n.total_time_s.mean() * len(f) / ks if ks else float("inf")
        lines.append(f"  numpy : mean time {n.total_time_s.mean():.2f}s per run -> time per success {tps_np:.2f}s "
                     f"(success count taken from fast, identical trajectories)")
        lines.append(f"  Fisher exact p (polish vs fast) = {R.fisher_success(int(p.success.sum()), len(p), ks, len(f)):.4g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, g) in enumerate(df[df.method != "numpy"].groupby("method")):
        for thr, a in g.groupby("threshold"):
            k = int(a.success.sum())
            lo, hi = wilson(k, len(a))
            ax.errorbar(i + 0.1 * list(sorted(df.threshold.unique())).index(thr), k / len(a),
                        yerr=[[k / len(a) - lo], [hi - k / len(a)]], fmt="o", capsize=4, label=f"{name} T={thr:g}")
    ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_xticks([])
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "success_ci.png"), dpi=150)
    plt.close(fig)
    run.summarise(extra_cols=("cache_hits", "pruned"))
    R.write_lines(run, "hard_target_summary.txt", lines)
    R.plot_best_runs(prob, [r for r in recs if r["method"] != "numpy"], os.path.join(run.run_dir, "plots"))
    run.finish()


if __name__ == "__main__":
    main()
