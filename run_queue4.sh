#!/bin/bash
# Queue 4: 100 particles x 40 iterations, numpy/solve/sm only (no iterative), 4 runs per target, one job at a time.
#   PY=/DATA/Aurindum/conda/envs/work/bin/python nohup setsid bash run_queue4.sh > queue4.log 2>&1 < /dev/null &
# A validate | B 28-port FULL (28x28) methods | C 21-port methods | D post_run.py on the two new result files
# Outputs go to NEW folders (the codes' default folders are untouched):
#   PSO_PCB28/full28_study/results_full_100x40        PSO_REF21/study_100x40/results/ref
# The RUN_START line of every job's first log must show particles=100 iters=40, otherwise the queue stops.
cd "$(dirname "$0")"; PY=${PY:-python}; export PY
P=PSO_PCB28; F=$P/full28_study; R=PSO_REF21; S=$R/study_100x40
export N_PARTICLES=100 N_ITERATIONS=40 NUM_RUNS=4 METHODS=numpy,solve,sm OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say(){ echo "[$(date '+%F %T')] $*"; }
for d in $F/results_full_100x40 $S/results; do [ -d "$d" ] && { mv "$d" "${d}_old_$(date +%Y%m%d_%H%M)"; say "moved aside $d"; }; done
mkdir -p $F/logs $S/logs

guard(){   # label results-root pid
  local label=$1 root=$2 pid=$3 f="" line=""
  for i in $(seq 1 360); do
    f=$(find "$root" -name 'run_1.log' 2>/dev/null | head -1)
    [ -n "$f" ] && grep -q RUN_START "$f" 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  line=$(grep -m1 RUN_START "$f" 2>/dev/null)
  say "$label first log: $line"
  case "$line" in *"particles=100 iters=40"*) return 0;; esac
  say "$label WRONG OR MISSING SWARM SETTINGS (need particles=100 iters=40) - stopping queue"; kill $pid 2>/dev/null; return 1
}
job(){     # label results-root workdir logfile cmd...
  local label=$1 root=$2 dir=$3 logf=$4; shift 4
  ( cd "$dir" && exec "$@" > "$logf" 2>&1 ) &
  local pid=$!
  guard "$label" "$root" $pid || { wait $pid 2>/dev/null; exit 1; }
  wait $pid; say "$label finished (exit $?)"
}
fail(){ say "$1 FAILED, see $2"; exit 1; }

say "STEP A validate (particles=$N_PARTICLES iters=$N_ITERATIONS runs=$NUM_RUNS methods=$METHODS)"
( cd $F && NET=full $PY pso_full28.py --validate ) > queue4_validate_28.log 2>&1 || fail "28-port validation" queue4_validate_28.log
( cd $R && LIB=ref TARGETS=0.06,0.05,0.045 FLOOR_RUNS=0 $PY pso_ref21.py --validate ) > queue4_validate_21.log 2>&1 || fail "21-port validation" queue4_validate_21.log
say "validate OK"

say "STEP B 28x28, 100x40, 4 runs"
job "STEP B" $F/results_full_100x40 $F logs/study_full_100x40.log env NET=full TIME_LIMIT_S=14400 OUT=$PWD/$F/results_full_100x40 $PY pso_full28.py
( cd $F && $PY summarize_methods.py results_full_100x40/results_long.jsonl > logs/summary_full_100x40.txt 2>&1 )
say "STEP C 21x21, 100x40, 4 runs"
job "STEP C" $S/results $R study_100x40/logs/study.log env LIB=ref TARGETS=0.06,0.05,0.045 FLOOR_RUNS=0 OUT=$PWD/$S/results $PY pso_ref21.py
( cd $R && $PY study2/summarize_methods.py "study_100x40/results/*/results_long.jsonl" > study_100x40/logs/summary.txt 2>&1 )
say "STEP D post_run"
$PY post_run.py $F/results_full_100x40/results_long.jsonl $S/results/ref/results_long.jsonl > post_run_100x40.log 2>&1
say "ALL DONE"
