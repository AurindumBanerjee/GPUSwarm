# Experiment 01 -- one PSO run serves every threshold.
#
# Seeding depends only on (run_id, n_caps), never on the threshold, so a run
# aimed at the lowest target passes through exactly the states the runs for
# the higher targets would have visited, up to the moment each of them
# stops. Recording the first crossing of every threshold inside one run
# gives the same time-to-target / caps / best |Z11| per threshold for a
# fraction of the total benchmark time.
#
# This script runs each method twice:
#   separate : the GPUBench2 protocol (one run per threshold)
#   shared   : one run per (method, run) at min(TARGETS), all crossings logged
# and reports (a) per-threshold agreement between the two protocols and
# (b) total benchmark wall time saved.
#
# nohup python bench_shared_thresholds.py > exp01.log 2>&1 &

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import gbcore as gb  # noqa: E402

p = gb.common_parser("Exp01 shared-threshold runs", "numpy,fast", "MinTime/Experiments/01_shared_threshold_runs")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
fac = gb.standard_evaluators(prob)
methods = gb.select_methods(args, fac)
gb.verify(prob, {m: fac[m] for m in methods}, args.out)
gb.warmup(prob, {m: fac[m] for m in methods})

t0 = time.perf_counter()
rec_sep = gb.run_benchmark(prob, fac, methods, args.out, shared=False, tag="_separate")
t_sep = time.perf_counter() - t0

t0 = time.perf_counter()
rec_sh = gb.run_benchmark(prob, fac, methods, args.out, shared=True, tag="_shared")
t_sh = time.perf_counter() - t0

gb.analyse(rec_sep, args.out, args.baseline, tag="_separate")
gb.analyse(rec_sh, args.out, args.baseline, tag="_shared")

# per-threshold agreement between the protocols
key = lambda r: (r["method"], r["threshold"], r["run_id"])
sep = {key(r): r for r in rec_sep}
agree, n = 0, 0
mismatch = []
for r in rec_sh:
    s = sep.get(key(r))
    if s is None:
        continue
    n += 1
    ok = (s["caps_at_target"] == r["caps_at_target"] and s["success"] == r["success"]
          and abs(s["best_minz_ohm"] - r["best_minz_ohm"]) <= 1e-9 * max(1, abs(s["best_minz_ohm"]))
          and s["n_evals"] == r["n_evals"])
    agree += ok
    if not ok:
        mismatch.append({"key": key(r), "separate": [s["caps_at_target"], s["best_minz_ohm"], s["n_evals"]],
                         "shared": [r["caps_at_target"], r["best_minz_ohm"], r["n_evals"]]})

summary = {
    "benchmark_wall_time_separate_s": t_sep,
    "benchmark_wall_time_shared_s": t_sh,
    "benchmark_speedup": t_sep / t_sh if t_sh > 0 else None,
    "per_threshold_records_compared": n,
    "records_identical": agree,
    "agreement_rate": agree / n if n else None,
    "mismatches": mismatch[:50],
}
with open(os.path.join(args.out, "shared_vs_separate.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}, indent=2))
print("Done.")
