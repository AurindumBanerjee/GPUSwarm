#!/bin/bash
# Sequential queue for the 28-port + 21-port studies. Strictly one job at a time so timings are not disturbed.
#   PY=/DATA/Aurindum/conda/envs/work/bin/python nohup setsid bash run_queue.sh > queue.log 2>&1 < /dev/null &
# Order: 0 validate everything -> 1 time_study (4 methods x 4 targets x 10 runs) -> 2 PSO_REF21 (ref vs murata, 4 targets x 10 runs + 3 floor runs)
#        -> 3 ordering_study (srf, cap, part, random x 10 runs) -> 4 plots -> 5 pure_python baseline (4 runs, round-robin over the 4 targets; longest, so last)
cd "$(dirname "$0")"; PY=${PY:-python}; export PY; STAMP=$(date +%Y%m%d_%H%M); P28=PSO_PCB28
say(){ echo "[$(date '+%F %T')] $*"; }
for d in $P28/time_study/results $P28/baseline_pure_python/results $P28/ordering_study/results_srf $P28/ordering_study/results_cap \
         $P28/ordering_study/results_part $P28/ordering_study/results_random PSO_REF21/results; do
  [ -d "$d" ] && mv "$d" "${d}_old_$STAMP" && say "moved aside $d"; done
say "STEP 0 validate"
( cd $P28 && $PY time_study/pso_pcb28_time.py --validate && ORDER=srf $PY ordering_study/pso_pcb28_order.py --validate && \
  $PY baseline_pure_python/pso_pcb28_pure.py --validate ) > queue_validate_28.log 2>&1 || { say "28-port validation FAILED, see queue_validate_28.log"; exit 1; }
( cd PSO_REF21 && $PY pso_ref21.py --validate ) > queue_validate_21.log 2>&1 || { say "21-port validation FAILED, see queue_validate_21.log"; exit 1; }
say "STEP 1 time_study";   PY=$PY NUM_RUNS=10 bash $P28/time_study/run_time_study.sh
say "STEP 2 PSO_REF21";    PY=$PY bash PSO_REF21/run_study.sh
say "STEP 3 ordering_study"
( cd $P28 && $PY ordering_study/make_orders.py && for o in srf cap part random; do
    ORDER=$o NUM_RUNS=10 STUDY_THREADS=4 $PY ordering_study/pso_pcb28_order.py > ordering_study/run_$o.log 2>&1; done
  $PY ordering_study/compare_orders.py > ordering_study/compare.txt 2>&1 )
say "STEP 4 plots (one PNG per run)"
( cd $P28 && $PY plot_placements.py --jsonl time_study/results/results_long.jsonl --out time_study/pdn
  $PY plot_placements.py --jsonl ordering_study/results_srf/results_long.jsonl --out ordering_study/pdn_srf ) > queue_plots.log 2>&1
say "STEP 5 pure_python baseline (round-robin over targets)"
( cd $P28 && NUM_RUNS=4 $PY baseline_pure_python/pso_pcb28_pure.py > baseline_pure_python/run.log 2>&1 )
say "ALL DONE"
