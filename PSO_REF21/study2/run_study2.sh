#!/bin/bash
# 21-port reference PDN, method time-to-target study, targets 60 / 50 / 45 mOhm, 50 particles x 15 iterations, legacy library
# (LIB=murata or LIB=both to include the Murata library). Output: study2/results/<lib>/..., logs/
cd "$(dirname "$0")" && mkdir -p logs; PY=${PY:-python}
LIB=${LIB:-ref} METHODS=${METHODS:-numpy,solve,sm,iterative} TARGETS=0.06,0.05,0.045 FLOOR_RUNS=0 OUT=$PWD/results $PY ../pso_ref21.py --validate > logs/validate.log 2>&1 || { echo "validate failed"; exit 1; }
LIB=${LIB:-ref} METHODS=${METHODS:-numpy,solve,sm,iterative} TARGETS=0.06,0.05,0.045 FLOOR_RUNS=0 NUM_RUNS=${NUM_RUNS:-5} OUT=$PWD/results $PY ../pso_ref21.py > logs/study.log 2>&1
$PY summarize_methods.py "results/*/results_long.jsonl" > logs/summary.txt 2>&1
