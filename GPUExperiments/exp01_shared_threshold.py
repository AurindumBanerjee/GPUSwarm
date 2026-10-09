"""
Experiment 1 -- shared-threshold runs.

The PSO trajectory does not depend on the target until the stop test fires (prefix property), so ONE run at the lowest target
(0.04) can report, for every higher target, the moment it is first crossed: (stage/caps, iteration, evaluations, time, cost).
This script runs, per paired seed, (a) separate runs at every target and (b) one shared run at the lowest target, with the same
engine ('fast' = batched Woodbury + cache + exact pruning; complex128), and checks that the crossing records equal the separate
runs EXACTLY (stage caps, iteration, logical evaluations, cost). It then measures the total benchmark time shared vs separate.

Rows in the summary: method=separate | shared, one per target; plus 'benchmark' totals in statistics.txt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gpuexp_common as C
from gpuexp_common import S

EXPERIMENT = "shared_threshold"


def main():
    run = C.Run(EXPERIMENT, __file__)
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    ev = C.SwarmEvaluator(prob, True, True)
    C.warmup(prob, [ev])
    targets = sorted(S["TARGETS"], reverse=True)
    lo = targets[-1]
    base = C.load_baseline()
    tot = {"separate": 0.0, "shared": 0.0}
    mism = []
    n_cmp = n_ok = 0
    for r in range(1, S["NUM_RUNS"] + 1):
        sep = {}
        for thr in targets:
            run.lg.info("RUN_START | method=separate | threshold=%.6f | run=%d", thr, r)
            rec = C.run_pso(prob, ev, [thr], r, label="separate")
            sep[thr] = rec
            tot["separate"] += rec["total_time_s"]
            rec["traj_match"], rec["traj_match_strict"] = C.match(rec, base.get((round(thr, 6), r)))
            run.add(rec)
            run.result_line(rec)
        run.lg.info("RUN_START | method=shared_run | threshold=%.6f | run=%d | reports=%s", lo, r, targets)
        sh = C.run_pso(prob, ev, targets, r, label="shared_run")
        tot["shared"] += sh["total_time_s"]
        sh["traj_match"], sh["traj_match_strict"] = C.match(sh, sep[lo])
        run.add(sh)
        run.result_line(sh)
        for thr in targets:                                     # derived record: what the shared run says about target thr
            cr = sh["crossings"].get(str(thr))
            s = sep[thr]
            d = {"method": "shared", "run": r, "threshold": thr, "seed": sh["seed"], "success": cr is not None,
                 "time_s": cr["time_s"] if cr else -1.0, "total_time_s": cr["time_s"] if cr else sh["total_time_s"],
                 "caps_at_target": cr["stage_caps"] if cr else None, "final_n_caps": cr["stage_caps"] if cr else sh["final_n_caps"],
                 "best_minz": cr["cost"] if cr else sh["best_minz"], "n_evals": cr["n_evals"] if cr else sh["n_evals"],
                 "iteration": cr["iteration"] if cr else None}
            sc = s["crossings"].get(str(thr))
            same = (cr is None and sc is None) or (cr is not None and sc is not None and cr["stage_caps"] == sc["stage_caps"]
                    and cr["iteration"] == sc["iteration"] and cr["n_evals"] == sc["n_evals"] and cr["cost"] == sc["cost"])
            if cr is None and sc is None:                       # neither reached it: full-budget runs must coincide
                same = s["n_evals"] == sh["n_evals"] and s["best_minz"] == sh["best_minz"]
            d["crossing_identical_to_separate"] = bool(same)
            d["traj_match"], d["traj_match_strict"] = C.match(d, s)
            d["shared_total_time_s"] = sh["total_time_s"]
            n_cmp += 1
            n_ok += int(same)
            if not same:
                mism.append((r, thr, cr, sc))
            run.add(d)
            run.lg.info("RESULT | method=shared | threshold=%s | run=%d | time_s=%.4f | caps=%s | n_evals=%s | success=%s | identical=%s",
                        thr, r, d["time_s"], d["caps_at_target"], d["n_evals"], d["success"], same)
    rows = run.summarise(group=("method", "threshold"), extra_cols=("iteration", "traj_match", "traj_match_strict",
                                                                    "crossing_identical_to_separate"))
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write(f"crossing_records_compared = {n_cmp}\ncrossing_records_identical = {n_ok}\n"
                 f"crossing_identical_rate = {n_ok / max(1, n_cmp):.4f}\n"
                 f"benchmark_time_separate_s = {tot['separate']:.3f}\nbenchmark_time_shared_s = {tot['shared']:.3f}\n"
                 f"benchmark_speedup_shared_over_separate = {tot['separate'] / max(tot['shared'], 1e-9):.3f}\n"
                 f"targets = {targets}\nruns = {S['NUM_RUNS']}\nmismatches = {len(mism)}\n")
        for m in mism:
            fh.write(f"MISMATCH run={m[0]} target={m[1]} shared={m[2]} separate={m[3]}\n")
    plots(run, tot)
    run.finish(notes={"crossing_identical_rate": n_ok / max(1, n_cmp), "benchmark_speedup": tot["separate"] / max(tot["shared"], 1e-9)})


def plots(run, tot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame([r for r in run.records if r["method"] in ("separate", "shared")])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df.groupby("method"):
        gg = g.groupby("threshold").time_s.apply(lambda x: x[x >= 0].mean())
        ax[0].plot(gg.index, gg.values, "o-", label=m)
    ax[0].set_xlabel("target |Z11| (ohm)"); ax[0].set_ylabel("mean time to target (s, successful runs)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar(["separate\n(all targets)", "shared\n(one run at lowest)"], [tot["separate"], tot["shared"]], color=["#4C72B0", "#55A868"])
    ax[1].set_ylabel("total benchmark time (s)")
    ax[1].set_title(f"speed-up {tot['separate'] / max(tot['shared'], 1e-9):.2f}x")
    fig.tight_layout(); fig.savefig(os.path.join(run.run_dir, "plots", "shared_vs_separate.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
