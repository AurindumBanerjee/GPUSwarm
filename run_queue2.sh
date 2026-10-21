#!/bin/bash
# Queue 2 (new targets). Stop all earlier jobs first. One job at a time so timings stay clean.
#   PY=/DATA/Aurindum/conda/envs/work/bin/python nohup setsid bash run_queue2.sh > queue2.log 2>&1 < /dev/null &
# 0 validate all | 1 28-port methods on the 12x12 reduction (5 runs) | 2 21-port methods, targets 60/50/45 mOhm (5 runs)
# | 3 28-port methods on the FULL 28x28 network (5 runs) | 4 pure_python baseline (4 runs, round-robin over the same 4 targets, 12x12)
cd "$(dirname "$0")"; PY=${PY:-python}; export PY; STAMP=$(date +%Y%m%d_%H%M); P=PSO_PCB28; say(){ echo "[$(date '+%F %T')] $*"; }
for d in $P/full28_study/results_full $P/full28_study/results_netA PSO_REF21/study2/results $P/baseline_pure_python/results; do
  [ -d "$d" ] && mv "$d" "${d}_old_$STAMP" && say "moved aside $d"; done
mkdir -p $P/full28_study/logs PSO_REF21/study2/logs
say "STEP 0 validate"
( cd $P/full28_study && NET=full $PY pso_full28.py --validate && NET=netA $PY pso_full28.py --validate && cd .. && $PY baseline_pure_python/pso_pcb28_pure.py --validate ) > queue2_validate_28.log 2>&1 || { say "28-port validation FAILED, see queue2_validate_28.log"; exit 1; }
( cd PSO_REF21/study2 && LIB=ref TARGETS=0.06,0.05,0.045 FLOOR_RUNS=0 OUT=$PWD/results $PY ../pso_ref21.py --validate ) > queue2_validate_21.log 2>&1 || { say "21-port validation FAILED, see queue2_validate_21.log"; exit 1; }
say "STEP 1 28-port, 12x12 reduction, numpy/solve/sm/iterative x 5 runs"
( cd $P/full28_study && NET=netA METHODS=numpy,solve,sm,iterative NUM_RUNS=5 TIME_LIMIT_S=7200 $PY pso_full28.py > logs/study_netA.log 2>&1; $PY summarize_methods.py results_netA/results_long.jsonl > logs/summary_netA.txt 2>&1 )
say "STEP 2 21-port, 60/50/45 mOhm, 50 particles x 15 iterations"
( cd PSO_REF21/study2 && PY=$PY NUM_RUNS=5 bash run_study2.sh )
say "STEP 3 28-port FULL 28x28, numpy/solve/sm/iterative x 5 runs"
( cd $P/full28_study && NET=full METHODS=numpy,solve,sm,iterative NUM_RUNS=5 TIME_LIMIT_S=14400 $PY pso_full28.py > logs/study_full.log 2>&1; $PY summarize_methods.py results_full/results_long.jsonl > logs/summary_full.txt 2>&1 )
say "STEP 4 pure_python baseline (12x12, round-robin over targets)"
( cd $P && NUM_RUNS=4 TIME_LIMIT_S=604800 $PY baseline_pure_python/pso_pcb28_pure.py > baseline_pure_python/run.log 2>&1 )
say "ALL DONE"
