#!/bin/bash
# Detached:  nohup setsid bash time_study/run_time_study.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")" && mkdir -p logs results
PY=${PY:-python}
$PY pso_pcb28_time.py --validate > logs/validate.log 2>&1 || { echo "validate failed, see logs/validate.log"; exit 1; }
NUM_RUNS=${NUM_RUNS:-10} $PY pso_pcb28_time.py > logs/time_study.log 2>&1
$PY summarize_time.py > logs/summary.txt 2>&1
