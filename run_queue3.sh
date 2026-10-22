#!/bin/bash
# Queue 3: every method run at 50 particles x 15 iterations (same as both pure_python baselines). One job at a time (timings stay clean).
#   PY=/DATA/Aurindum/conda/envs/work/bin/python nohup setsid bash run_queue3.sh > queue3.log 2>&1 < /dev/null &
# A validate | B 28-port FULL numpy/solve/sm/iterative x5 | C 28-port pure_python x5 | D 21-port pure_python x5 | E 12x12 reduction methods x5 (last) | F post_run.py
# Before a job's result is trusted the RUN_START line of its first log must show particles=50 iters=15, otherwise the queue stops.
cd "$(dirname "$0")"; PY=${PY:-python}; export PY; STAMP=$(date +%Y%m%d_%H%M)
P=PSO_PCB28; F=$P/full28_study; B=$P/baseline_pure_python; R=PSO_REF21/baseline_pure_python
export N_PARTICLES=50 N_ITERATIONS=15 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say(){ echo "[$(date '+%F %T')] $*"; }
for d in $F/results_full $F/results_netA $B/results $R/results; do [ -d "$d" ] && mv "$d" "${d}_old_$STAMP" && say "moved aside $d"; done
for f in $B/run.log $R/run.log; do [ -f "$f" ] && mv "$f" "${f%.log}_old_$STAMP.log" && say "moved aside $f"; done
mkdir -p $F/logs

guard(){   # label results-root pid: wait for the first run log, require particles=50 iters=15
  local label=$1 root=$2 pid=$3 f="" line=""
  for i in $(seq 1 360); do
    f=$(find "$root" -name 'run_1.log' 2>/dev/null | head -1)
    [ -n "$f" ] && grep -q RUN_START "$f" 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  line=$(grep -m1 RUN_START "$f" 2>/dev/null)
  say "$label first log: $line"
  case "$line" in *"particles=50 iters=15"*) return 0;; esac
  say "$label WRONG OR MISSING SWARM SETTINGS (need particles=50 iters=15) - stopping queue"; kill $pid 2>/dev/null; return 1
}
job(){     # label results-root workdir logfile cmd...
  local label=$1 root=$2 dir=$3 logf=$4; shift 4
  ( cd "$dir" && exec "$@" > "$logf" 2>&1 ) &
  local pid=$!
  guard "$label" "$root" $pid || { wait $pid 2>/dev/null; exit 1; }
  wait $pid; say "$label finished (exit $?)"
}
fail(){ say "$1 FAILED, see $2"; exit 1; }

say "STEP A validate (N_PARTICLES=$N_PARTICLES N_ITERATIONS=$N_ITERATIONS)"
( cd $F && NET=full $PY pso_full28.py --validate && NET=netA $PY pso_full28.py --validate ) > queue3_validate_methods.log 2>&1 || fail "28-port methods validation" queue3_validate_methods.log
( cd $P && NET=full $PY baseline_pure_python/pso_pcb28_pure.py --validate ) > queue3_validate_pure28.log 2>&1 || fail "28-port pure_python validation" queue3_validate_pure28.log
( cd $R && LIB=ref $PY pso_ref21_pure.py --validate ) > queue3_validate_pure21.log 2>&1 || fail "21-port pure_python validation" queue3_validate_pure21.log
say "validate OK:"; grep -h "ETA\|one evaluation" queue3_validate_pure28.log queue3_validate_pure21.log | sed 's/^/    /'

say "STEP B 28-port FULL methods"
job "STEP B" $F/results_full $F logs/study_full.log env NET=full METHODS=numpy,solve,sm,iterative NUM_RUNS=5 TIME_LIMIT_S=14400 $PY pso_full28.py
( cd $F && $PY summarize_methods.py results_full/results_long.jsonl > logs/summary_full.txt 2>&1 )
say "STEP C 28-port pure_python baseline"
job "STEP C" $B/results $P baseline_pure_python/run.log env NET=full NUM_RUNS=5 TIME_LIMIT_S=604800 $PY baseline_pure_python/pso_pcb28_pure.py
say "STEP D 21-port pure_python baseline"
job "STEP D" $R/results $R run.log env LIB=ref NUM_RUNS=5 TIME_LIMIT_S=604800 $PY pso_ref21_pure.py
say "STEP E 12x12 reduction methods at 50x15"
job "STEP E" $F/results_netA $F logs/study_netA.log env NET=netA METHODS=numpy,solve,sm,iterative NUM_RUNS=5 TIME_LIMIT_S=7200 $PY pso_full28.py
( cd $F && $PY summarize_methods.py results_netA/results_long.jsonl > logs/summary_netA.txt 2>&1 )
say "STEP F post_run"
$PY post_run.py > post_run.log 2>&1
say "ALL DONE"
