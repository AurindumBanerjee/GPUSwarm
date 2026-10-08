#!/bin/bash
# Run reference/run01_template_sample. Launch from anywhere; the script runs from the repository root.
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN/../../.." || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUN_DIR="$RUN" RUN_ID="run01_template_sample" NUM_RUNS=3
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "START: $(stamp)  host: $(hostname)  load: $(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)" >> "$RUN/README.txt"
python -u "$RUN/code/template_experiment__run01_template_sample.py"  > "$RUN/logs/console.log" 2>&1
echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"
