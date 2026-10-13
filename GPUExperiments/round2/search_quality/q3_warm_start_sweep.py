"""
Search quality Q3 -- how much of the swarm should be warm-started from the previous stage?

At stage n+1 a fraction of the particles starts from the stage-n best (its first 2n genes copied, optionally jittered); the rest
is uniformly random. The re-runs showed 40% warm -> 5/20 successes at 0.04, 100% warm -> 0/20: too much warm start collapses
exploration. This sweep maps the whole range with 'fast' (the exact evaluator; only the search changes):

    WARM_FRACS  default 0, 0.2, 0.4, 0.6, 0.8, 1.0
    JITTERS     default 0, 0.02   (Gaussian jitter on the carried genes of warm particles 1..)
Paired seeds across all cells. Outputs: success / caps / time / evaluations per cell, heat maps, before/after plots of the best
cell.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import gpuexp_common as C
from gpuexp_common import S

EXPERIMENT = "q3_warm_start_sweep"
DATASET = os.environ.get("DATASET", "ref21")
WARM_FRACS = R.env_list("WARM_FRACS", "0,0.2,0.4,0.6,0.8,1.0", float)
JITTERS = R.env_list("JITTERS", "0,0.02", float)


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "WARM_FRACS": WARM_FRACS, "JITTERS": JITTERS})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    ev = C.SwarmEvaluator(prob, True, True)
    C.warmup(prob, [ev])
    targets = R.targets_for(info)
    recs = []
    orig = (S["WARM_START_FRACTION"], S["WARM_START_JITTER"])
    try:
        for fr in WARM_FRACS:
            for j in JITTERS:
                S["WARM_START_FRACTION"], S["WARM_START_JITTER"] = fr, j
                name = f"warm{fr:g}_jit{j:g}"
                recs += R.loop_methods(run, prob, {name: ev}, targets, S["NUM_RUNS"], reference="__none__",
                                       extra={"warm_fraction": fr, "jitter": j})
    finally:
        S["WARM_START_FRACTION"], S["WARM_START_JITTER"] = orig
    import pandas as pd
    df = pd.DataFrame(recs)
    tab = df.groupby(["threshold", "warm_fraction", "jitter"]).agg(
        success=("success", "mean"), caps=("caps_at_target", "mean"), time=("total_time_s", "mean"),
        evals=("n_evals", "mean"), best=("best_minz", "mean")).reset_index()
    tab.to_csv(os.path.join(run.run_dir, "summary", "warm_start_grid.csv"), index=False)
    lines = [tab.to_string(index=False), ""]
    for thr, g in tab.groupby("threshold"):
        b = g.sort_values(["success", "time"], ascending=[False, True]).iloc[0]
        lines.append(f"target {thr:g}: best cell warm={b.warm_fraction:g} jitter={b.jitter:g} -> success {b.success:.2f}, "
                     f"mean time {b.time:.2f}s, caps {b.caps:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ths = sorted(tab.threshold.unique())
    fig, ax = plt.subplots(1, len(ths), figsize=(4.2 * len(ths), 3.6), squeeze=False)
    for a, thr in zip(ax[0], ths):
        g = tab[tab.threshold == thr].pivot(index="jitter", columns="warm_fraction", values="success")
        im = a.imshow(g.values, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
        a.set_xticks(range(len(g.columns)), [f"{c:g}" for c in g.columns])
        a.set_yticks(range(len(g.index)), [f"{c:g}" for c in g.index])
        for (i, jj), v in np.ndenumerate(g.values):
            a.text(jj, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
        a.set_xlabel("warm-start fraction")
        a.set_ylabel("jitter")
        a.set_title(f"success rate, target {thr:g}")
    fig.colorbar(im, ax=ax.ravel().tolist())
    fig.savefig(os.path.join(run.run_dir, "plots", "warm_start_success.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    run.summarise(extra_cols=("warm_fraction", "jitter", "cache_hits", "pruned"))
    R.write_lines(run, "warm_start_summary.txt", lines)
    R.plot_best_runs(prob, recs, os.path.join(run.run_dir, "plots"))
    run.finish()


if __name__ == "__main__":
    main()
