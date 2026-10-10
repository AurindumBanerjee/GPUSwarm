#!/bin/bash
# Run gpu_experiments/run_compare_naive40. Launch from anywhere; the script runs from the repository root.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN/../../.." || exit 1
source $(conda info --base 2>/dev/null || echo /DATA/Aurindum/conda)/etc/profile.d/conda.sh
conda activate gpuwork
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUN_DIR="$RUN" RUN_ID="run_compare_naive40" QUICK=0 GROUP_DIRS=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_shared_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_mixed_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_incremental_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_gpupso_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_polish_naive40,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_macro_naive40 CPU_SUMMARY=/DATA/Aurindum/Swarming/runs/scratchbench/run08_warm100_5runs/summary/summary.csv BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "START: $(stamp)  host: $(hostname)  load: $(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)" >> "$RUN/README.txt"
python -u "$RUN/code/exp07_compare__run_compare_naive40.py"  > "$RUN/logs/console.log" 2>&1
echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
