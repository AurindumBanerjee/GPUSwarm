"""
Final compare run: cross-experiment tables and plots from the finished runs of one run_all.sh invocation.
Inputs (env): GROUP_DIRS (comma separated run directories), CPU_SUMMARY (Swarming run08 summary.csv).
Baseline (a) = the 'numpy' arm of the 'baseline' run (same GPU, full inverse per particle, same settings and paired seeds).
Baseline (b) = CPU ScratchBench run08: 50 particles x 15 iterations, 5 runs -- NOT the same search budget as the GPU runs
(50 x 40, 20 runs); shown as separate rows and labelled.
speedup_vs_gpu_numpy = mean total wall time of baseline (a) at the same target / mean total wall time of the method.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import gpuexp_common as C

EXPERIMENT = "compare"


def main():
    run = C.Run(EXPERIMENT, __file__)
    dirs = [d for d in os.environ.get("GROUP_DIRS", "").split(",") if d]
    rows = []
    for d in dirs:
        jl = os.path.join(d, "output", "results_long.jsonl")
        if not os.path.exists(jl):
            run.lg.info("STAGE | missing results in %s", d)
            continue
        exp = json.load(open(os.path.join(d, "manifest.json"))).get("experiment", os.path.basename(d))
        for line in open(jl):
            r = json.loads(line)
            r["experiment"] = exp
            rows.append(r)
    df = pd.DataFrame(rows)
    base = df[(df.experiment == "baseline") & (df.method == "numpy")].groupby("threshold").total_time_s.mean()
    out = []
    for (exp, m, thr), g in df.groupby(["experiment", "method", "threshold"]):
        if m == "shared_run":
            continue
        ok = g[g.success == True]                                              # noqa: E712
        tt = g[g.total_time_s >= 0].total_time_s
        row = {"experiment": exp, "method": m, "threshold": thr, "runs": len(g), "success_rate": float(g.success.mean()),
               "time_to_target_mean_s": float(ok.time_s.mean()) if len(ok) else None,
               "total_time_mean_s": float(tt.mean()) if len(tt) else None,
               "speedup_vs_gpu_numpy": float(base[thr] / tt.mean()) if (thr in base.index and len(tt) and tt.mean() > 0) else None,
               "caps_mean": float(ok.caps_at_target.mean()) if len(ok) else None, "n_evals_mean": float(g.n_evals.mean()),
               "best_minz_mean": float(g.best_minz.mean())}
        for c in ("traj_match", "traj_match_strict"):
            if c in g and g[c].notna().any():
                row[c + "_rate"] = float(pd.to_numeric(g[c]).mean())
        out.append(row)
    tab = pd.DataFrame(out)
    cpu = os.environ.get("CPU_SUMMARY", "")
    if cpu and os.path.exists(cpu):
        c = pd.read_csv(cpu)
        c = c[c.method == "numpy"]
        cpu_rows = [{"experiment": "cpu_scratchbench_run08", "method": "cpu numpy (50 particles x 15 iterations, 5 runs)",
                     "threshold": r.threshold, "runs": r.runs_found, "success_rate": r.success_rate,
                     "time_to_target_mean_s": r.time_mean_s, "caps_mean": r.caps_mean, "best_minz_mean": r.best_minz_mean}
                    for _, r in c.iterrows()]
        cp = {r["threshold"]: r["time_to_target_mean_s"] for r in cpu_rows if r["time_to_target_mean_s"] == r["time_to_target_mean_s"]}
        tab["speedup_vs_cpu_numpy_time_to_target (budget differs)"] = tab.apply(
            lambda r: (cp[r.threshold] / r.time_to_target_mean_s) if (r.threshold in cp and r.time_to_target_mean_s
                                                                      and r.time_to_target_mean_s == r.time_to_target_mean_s) else None, axis=1)
        tab = pd.concat([tab, pd.DataFrame(cpu_rows)], ignore_index=True)
    tab = tab.sort_values(["threshold", "experiment", "method"], ascending=[False, True, True])
    tab.to_csv(os.path.join(run.run_dir, "summary", "summary.csv"), index=False)
    with open(os.path.join(run.run_dir, "summary", "summary.txt"), "w") as fh:
        for _, r in tab.iterrows():
            fh.write(" | ".join(f"{k}={'-' if (v is None or v != v) else (format(v, '.6g') if isinstance(v, float) else v)}" for k, v in r.items()) + "\n")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "w") as fh:
        fh.write(f"records = {len(df)}\nsuccesses = {int(df.success.sum())}\nfailures_or_timeouts = {int((~df.success).sum())}\n"
                 f"best_overall = {df.best_minz.min():.6g}\nruns_compared = {len(dirs)}\n")
        fh.write("baseline (a) = GPU numpy arm of the baseline run; baseline (b) = CPU ScratchBench run08 (50 x 15, 5 runs, different budget)\n")
        for d in dirs:
            st = os.path.join(d, "summary", "statistics.txt")
            if os.path.exists(st):
                fh.write(f"\n--- {os.path.basename(d)} ---\n" + open(st).read())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tab.dropna(subset=["speedup_vs_gpu_numpy"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for (exp, m), g in t.groupby(["experiment", "method"]):
        g = g.sort_values("threshold")
        ax[0].plot(g.threshold, g.speedup_vs_gpu_numpy, "o-", label=f"{exp}:{m}")
        ax[1].plot(g.threshold, g.success_rate, "o-", label=f"{exp}:{m}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("target |Z11| (ohm)")
    ax[0].set_ylabel("speed-up over GPU numpy baseline (total time)")
    ax[1].set_xlabel("target |Z11| (ohm)")
    ax[1].set_ylabel("success rate")
    ax[0].legend(fontsize=5)
    ax[0].grid(True, which="both", alpha=.3)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "speedup_and_success.png"), dpi=150)
    plt.close(fig)
    run.finish()


if __name__ == "__main__":
    main()
