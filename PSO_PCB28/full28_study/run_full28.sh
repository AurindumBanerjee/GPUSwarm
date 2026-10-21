#!/bin/bash
cd "$(dirname "$0")" && mkdir -p logs results; PY=${PY:-python}
$PY pso_full28.py --validate > logs/validate.log 2>&1 || { echo "validate failed, see logs/validate.log"; exit 1; }
NUM_RUNS=${NUM_RUNS:-10} $PY pso_full28.py > logs/study.log 2>&1
$PY summarize_methods.py > logs/summary.txt 2>&1
