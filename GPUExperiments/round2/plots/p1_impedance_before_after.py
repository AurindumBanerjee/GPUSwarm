"""
Plots P1 -- impedance with and without decaps on the same axes, from the results of ANY finished run.

Every run record (round 1 GPUExperiments and round 2) stores the final placement (best_models, best_ports). This script reads
one or more results_long.jsonl files and, for the PDN the run used, draws |Z11|(f) of the bare PDN and of the placed decaps:

    plots/<method>_T<target>_best.png       best run (successful first) per method and target
    plots/compare_T<target>.png             best run of every method on one plot per target
    plots/<method>_T<target>_allruns.png    every run of a method (thin lines) + bare PDN, if ALL_RUNS=1

Inputs (environment):
    RESULTS    comma-separated results_long.jsonl files or run directories (their output/results_long.jsonl is used)
    DATASET    dataset the runs used (ref21, ddr21, ddr79, mer1, mer2, pll1v0, pll1v8, mphyvdd); default ref21
    SP_FMAX / LIB_SOURCE   must match the run (defaults 50e6 / all for SP sets)
    METHODS    optional filter, ALL_RUNS=1 for the all-runs plots
Model indices refer to the library as the run loaded it (round-1 runs use the reference set with the full library).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import gpuexp_common as C

EXPERIMENT = "p1_impedance_before_after"
DATASET = os.environ.get("DATASET", "ref21")
RESULTS = R.env_list("RESULTS", "")
METHODS = R.env_list("METHODS", "")
ALL_RUNS = os.environ.get("ALL_RUNS", "0") == "1"


def load(paths):
    rows = []
    for p in paths:
        f = os.path.join(p, "output", "results_long.jsonl") if os.path.isdir(p) else p
        for line in open(f):
            r = json.loads(line)
            if r.get("best_models") and (not METHODS or r["method"] in METHODS):
                r["source"] = f
                rows.append(r)
    return rows


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "RESULTS": RESULTS, "METHODS": METHODS, "ALL_RUNS": ALL_RUNS})
    prob, info = R.load_problem(DATASET)
    rows = load(RESULTS)
    if not rows:
        run.lg.info("STAGE | no records with placements in %s", RESULTS)
        run.finish()
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(run.run_dir, "plots")
    lines = []
    for thr, g in df.groupby("threshold"):
        picks, labels = [], []
        for m, a in g.groupby("method"):
            ok = a[a.success == True]                                          # noqa: E712
            r = (ok if len(ok) else a).sort_values("best_minz").iloc[0]
            z = R.z11_curve(prob, r.best_models, r.best_ports)
            lines.append(f"T={thr:g} {m}: run {int(r.run)} recorded best {r.best_minz:.5g} | recomputed peak {z.max():.5g} | "
                         f"{len(r.best_models)} caps | bare peak {info['bare_peak']:.5g}")
            safe = str(m).replace("/", "_")
            R.impedance_before_after(prob, [(r.best_models, r.best_ports)], thr, os.path.join(out, f"{safe}_T{thr:g}_best.png"),
                                     title=f"{info['dataset']}: {m}, target {thr:g} ohm, run {int(r.run)}")
            picks.append((r.best_models, r.best_ports))
            labels.append(str(m))
            if ALL_RUNS:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4.8))
                ax.loglog(prob.freq, R.z11_curve(prob, None, None), color="0.3", lw=1.8, label="bare PDN")
                for rr in a.itertuples():
                    ax.loglog(prob.freq, R.z11_curve(prob, rr.best_models, rr.best_ports), lw=.7, alpha=.6,
                              color="tab:green" if rr.success else "tab:orange")
                ax.axhline(thr, color="crimson", ls="--", lw=1, label="target")
                ax.set_title(f"{m}: all {len(a)} runs (green = met target)")
                ax.set_xlabel("frequency (Hz)")
                ax.set_ylabel("|Z11| (ohm)")
                ax.grid(True, which="both", alpha=.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{safe}_T{thr:g}_allruns.png"), dpi=150)
                plt.close(fig)
        R.impedance_before_after(prob, picks, thr, os.path.join(out, f"compare_T{thr:g}.png"),
                                 title=f"{info['dataset']}: best run per method, target {thr:g} ohm", labels=labels)
    R.write_lines(run, "plots_summary.txt", lines)
    run.finish()


if __name__ == "__main__":
    main()
