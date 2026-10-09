#!/bin/bash
# Re-run the whole GPU experiment set (baseline, exp 1-6, compare) with a NAIVE warm start (jitter 0), twice, one after the other
# (GPU timings are only valid with the GPU to themselves):  40% warm  and  100% warm.  Each config gets its own run folders:
#   runs/gpu_experiments/run_<name>_naive40/   and   runs/gpu_experiments/run_<name>_naive100/
# Exp 4 thresholds are raised to 0.9/0.85/0.8 ohm (the first run's 0.6/0.5/0.45 were below the 0.76 ohm floor, so nothing could succeed).
cd /DATA/Aurindum/GPUSwarm || exit 1
for cfg in "naive40 0.4" "naive100 1.0"; do
  set -- $cfg
  bash GPUExperiments/run_all.sh --tag $1 --warm $2 --jitter 0.0 --t4 0.9,0.85,0.8 > GPUExperiments/run_all_$1.log 2>&1
  echo "[variants] $1 finished $(date) (exit $?)" >> GPUExperiments/run_naive_variants.log
done
echo "[variants] all done $(date)" >> GPUExperiments/run_naive_variants.log
