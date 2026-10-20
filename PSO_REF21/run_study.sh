#!/bin/bash
# Detached: nohup setsid bash run_study.sh > /dev/null 2>&1 < /dev/null &      (PY=/path/to/python to override)
cd "$(dirname "$0")" && mkdir -p logs; PY=${PY:-python}
$PY pso_ref21.py --validate > logs/validate.log 2>&1 || { echo "validate failed, see logs/validate.log"; exit 1; }
NUM_RUNS=${NUM_RUNS:-10} FLOOR_RUNS=${FLOOR_RUNS:-3} $PY pso_ref21.py > logs/study.log 2>&1
$PY summarize.py > logs/summary.txt 2>&1
