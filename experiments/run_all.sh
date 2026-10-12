#!/usr/bin/env bash
# Usage: bash run_all.sh [extra args passed to every experiment]
# e.g.   bash run_all.sh --y /DATA/.../y2.mat --d /DATA/.../decaps.mat --freq /DATA/.../freq2.mat
set -e
cd "$(dirname "$0")"
for s in 01_shared_threshold_runs/bench_shared_thresholds.py \
         02_mixed_precision_refinement/bench_mixed_precision.py \
         03_incremental_woodbury_updates/bench_incremental.py \
         04_macromodel_rational_peak/bench_macromodel.py \
         05_continuous_relaxation_polish/bench_relaxation_polish.py \
         06_gpu_resident_pso/bench_gpu_resident.py; do
  d=$(dirname "$s")
  echo "=== $d"
  python "$s" "$@" > "$d/$d.log" 2>&1 || echo "  FAILED (see $d/$d.log)"
done
