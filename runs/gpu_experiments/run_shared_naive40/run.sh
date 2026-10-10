#!/bin/bash
# Run gpu_experiments/run_shared_naive40. Launch from anywhere; the script runs from the repository root.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN/../../.." || exit 1
source $(conda info --base 2>/dev/null || echo /DATA/Aurindum/conda)/etc/profile.d/conda.sh
conda activate gpuwork
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUN_DIR="$RUN" RUN_ID="run_shared_naive40" QUICK=0 Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl WARM_START_FRACTION=0.4 WARM_START_JITTER=0.0
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "START: $(stamp)  host: $(hostname)  load: $(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)" >> "$RUN/README.txt"
taskset -c 24-31 python -u "$RUN/code/exp01_shared_threshold__run_shared_naive40.py"  > "$RUN/logs/console.log" 2>&1
echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
