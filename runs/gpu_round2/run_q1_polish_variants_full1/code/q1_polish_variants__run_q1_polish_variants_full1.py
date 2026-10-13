"""
Search quality Q1 -- make gradient polish cheaper without losing its success gain.

Experiment 5 polish raised 0.04 successes from 5/20 to 13/20 but cost ~6x the time of 'fast'. The polish (exp05_gradient_polish:
R-L-C relaxation -> Adam through the exact Woodbury cost -> snap to real parts -> exact local sweep) is reused unchanged; only
WHEN and HOW LONG it runs is varied:

    fast            no polish (reference)
    polish_default  POLISH_RATIO 1.3, POLISH_STEPS 180 (experiment 5 settings)
    polish_short    POLISH_STEPS 45                     -> cheaper descent
    polish_tight    POLISH_RATIO 1.1                    -> only stages already within 10% of the target
    polish_nosweep  SWEEPS 0                            -> snap only, no exact local sweep
    polish_every    POLISH_RATIO 1e9                    -> every unfinished stage

Every polished candidate is scored exactly on the tabulated library, so success is genuine.
Main metric: time per success = total wall time of all runs / number of successes. Fisher exact test vs fast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import gpuexp_common as C
from gpuexp_common import S
import exp05_gradient_polish as P5

EXPERIMENT = "q1_polish_variants"
DATASET = os.environ.get("DATASET", "ref21")
VARIANTS = {
    "polish_default": {},
    "polish_short": {"POLISH_STEPS": 45},
    "polish_tight": {"POLISH_RATIO": 1.1},
    "polish_nosweep": {"SWEEPS": 0},
    "polish_every": {"POLISH_RATIO": 1e9},
}
ARMS = R.env_list("ARMS", "fast," + ",".join(VARIANTS))
DEFAULTS = {k: getattr(P5, k) for k in ("POLISH_RATIO", "POLISH_STEPS", "SWEEPS", "SNAP_CAND", "POLISH_LR")}


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "ARMS": ARMS, "VARIANTS": VARIANTS, "DEFAULTS": DEFAULTS})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    pol = P5.Polisher(prob)
    ev = C.SwarmEvaluator(prob, True, True)
    C.warmup(prob, [ev])
    pol.polish([0, 1], [int(prob.valid_ports[0]), int(prob.valid_ports[1])])
    targets = R.targets_for(info)
    recs = []
    for arm in ARMS:
        for k, v in DEFAULTS.items():
            setattr(P5, k, v)
        for k, v in VARIANTS.get(arm, {}).items():
            setattr(P5, k, v)
        hook = None if arm == "fast" else (lambda name, thr: P5.make_hook(prob, pol, thr))
        pol.log.clear()
        rs = R.loop_methods(run, prob, {arm: ev}, targets, S["NUM_RUNS"], reference="__none__", on_stage_end=hook)
        recs += rs
    import pandas as pd
    df = pd.DataFrame(recs)
    lines = []
    for thr, g in df.groupby("threshold"):
        f = g[g.method == "fast"]
        for arm, a in g.groupby("method"):
            p = R.fisher_success(int(a.success.sum()), len(a), int(f.success.sum()), len(f)) if len(f) and arm != "fast" else float("nan")
            lines.append(f"target {thr:g} {arm:15s}: success {int(a.success.sum())}/{len(a)} | mean time {a.total_time_s.mean():.2f}s | "
                         f"time per success {R.time_per_success(a):.2f}s | caps {a.caps_at_target.dropna().mean():.2f} | "
                         f"Fisher p vs fast {p:.3g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for arm, g in df.groupby("method"):
        s = g.groupby("threshold").success.mean()
        tps = g.groupby("threshold").apply(R.time_per_success)
        ax[0].plot(s.index, s.values, "o-", label=arm)
        ax[1].semilogy(tps.index, tps.replace(np.inf, np.nan).values, "o-", label=arm)
    ax[0].set_ylabel("success rate")
    ax[1].set_ylabel("time per success (s)")
    for a in ax:
        a.set_xlabel("target |Z11| (ohm)")
        a.grid(True, which="both", alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "polish_variants.png"), dpi=150)
    plt.close(fig)
    run.summarise(extra_cols=("cache_hits", "pruned"))
    R.write_lines(run, "polish_variants_summary.txt", lines)
    R.plot_best_runs(prob, recs, os.path.join(run.run_dir, "plots"))
    run.finish()


if __name__ == "__main__":
    main()
