"""
Experiment 0 -- baselines on this GPU, inside our own run (same settings as experiments 1-6; run07 used 0.4/0.02 warm start and
ran concurrently with CPU jobs, so it is not reusable).

Arms (all complex128, same PSO, same paired seeds):
    numpy           GPUBench2 'numpy' method: one particle at a time, full inverse of the assembled Y at every frequency  <- BASELINE (a)
    woodbury_batch  exact Woodbury, whole swarm in one batched solve
    fast            woodbury_batch + configuration cache + exact lower-bound pruning
Trajectories of the three arms must coincide (best_minz, caps_at_target, n_evals): the match rate against 'numpy' is reported.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gpuexp_common as C
from gpuexp_common import S

EXPERIMENT = "baseline"
METHODS = os.environ.get("METHODS", "numpy,woodbury_batch,fast").split(",")


def main():
    run = C.Run(EXPERIMENT, __file__, {"METHODS": METHODS})
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    run.lg.info("STAGE | data loaded | F=%d n=%d models=%d", prob.F, prob.n, prob.C)
    evs = {"numpy": C.SeqNumpyEvaluator(prob), "woodbury_batch": C.SwarmEvaluator(prob, False, False),
           "fast": C.SwarmEvaluator(prob, True, True)}
    evs = {m: evs[m] for m in METHODS}
    rep = C.verify_against_cpu(prob, list(evs.values()))
    run.lg.info("STAGE | verify vs CPU full inverse | max_rel=%s", rep)
    C.warmup(prob, list(evs.values()))
    base = {}
    for thr in S["TARGETS"]:
        for m, ev in evs.items():
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d | seed=%d", m, thr, r, S["BASE_SEED"] + r * 10000)
                try:
                    rec = C.run_pso(prob, ev, [thr], r)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(m, thr, r, e)
                    continue
                if m == "numpy":
                    base[(thr, r)] = rec
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, base.get((thr, r)))
                run.add(rec)
                run.result_line(rec, match=rec["traj_match"], strict=rec["traj_match_strict"])
    run.summarise(extra_cols=("gpu_full_evals", "cache_hits", "pruned", "traj_match", "traj_match_strict"))
    plots(run)
    run.finish(notes={"verify_max_rel": rep})


def plots(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(run.records)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df.groupby("method"):
        gg = g.groupby("threshold").total_time_s
        ax[0].errorbar(gg.mean().index, gg.mean().values, yerr=[C.ci95(x) for _, x in gg], marker="o", capsize=3, label=m)
    b = df[df.method == "numpy"].groupby("threshold").total_time_s.mean() if (df.method == "numpy").any() else None
    if b is not None:
        for m, g in df.groupby("method"):
            gm = g.groupby("threshold").total_time_s.mean()
            ax[1].plot(gm.index, b.reindex(gm.index) / gm.values, marker="o", label=m)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_ylabel("wall time per run (s)"); ax[1].set_ylabel("speed-up vs GPU numpy (mean time ratio)")
    for a in ax:
        a.set_xlabel("target |Z11| (ohm)"); a.grid(True, which="both", alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run.run_dir, "plots", "time_and_speedup.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
