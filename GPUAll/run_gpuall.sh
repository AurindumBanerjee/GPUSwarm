#!/bin/bash
# GPUAll launcher (server): floors -> study (all arms) -> greedy oracle -> cost model -> comparison. Each step is resumable / re-runnable.
#   bash GPUAll/run_gpuall.sh [--quick] [--steps floor,study,greedy,cost,compare] [extra args for run_gpuall.py ...]
# Detached:  nohup setsid bash GPUAll/run_gpuall.sh > GPUAll/run_gpuall.out 2>&1 < /dev/null &
# Data config as AllCPU: ALLCPU_GRID=5g ALLCPU_LIB=murata (defaults). PDN 3/4 need AllCPU/data_built (AllCPU/run_prep.sh builds it).
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export GPUSWARM_ROOT="$(dirname "$G")"
STEPS=floor,study,greedy,cost,compare; QUICK=0; EXTRA=()
while [ $# -gt 0 ]; do case $1 in --steps) STEPS=$2; shift 2;; --quick) QUICK=1; shift;; *) EXTRA+=("$1"); shift;; esac; done
source /DATA/Aurindum/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate gpuwork
export ALLCPU_GRID=${ALLCPU_GRID:-5g} ALLCPU_LIB=${ALLCPU_LIB:-murata}
if [ $QUICK = 1 ]; then Q=(--swarms 50x15 --runs 1 --levels T3 --numpy-swarms 50x15); CQ=--quick; else Q=(); CQ=; fi
echo "GPUAll start $(date) steps=$STEPS cfg=${ALLCPU_GRID}_${ALLCPU_LIB} quick=$QUICK"; nvidia-smi --query-gpu=name,memory.used --format=csv
has() { [[ ",$STEPS," == *",$1,"* ]]; }
cd "$G"
has floor   && python -u run_gpuall.py --floor "${EXTRA[@]}"
has study   && python -u run_gpuall.py "${Q[@]}" "${EXTRA[@]}"
has greedy  && (cd greedy_oracle && python -u greedy_oracle.py)
has cost    && (cd cost_model && python -u cost_model.py $CQ)
has compare && python -u compare_gpuall.py --overlay --cpu "$GPUSWARM_ROOT/AllCPU/CPUParallel/out_${ALLCPU_GRID}_${ALLCPU_LIB}"
echo "GPUAll done $(date)"
