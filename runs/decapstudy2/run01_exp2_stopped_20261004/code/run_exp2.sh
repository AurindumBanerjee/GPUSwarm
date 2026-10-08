#!/usr/bin/env bash
# Exp 2 launcher.
#
# Pins every BLAS / OpenMP layer to ONE thread before python starts (parallelism
# comes from STUDY_THREADS particle threads instead; stacking both oversubscribes
# the cores), then runs run_all.py detached under nohup so it survives the SSH
# session and a closed laptop. Everything it writes stays inside DecapStudy2/.
#
#   ./run_exp2.sh                 start the full study, detached
#   ./run_exp2.sh --validate      foreground dry check: data, port rules, targets, plan (no PSO)
#   ./run_exp2.sh --pause-after-stage1     extra args are passed to run_all.py
#
# Env overrides: PYTHON (default: conda env "work"), PDN_DATA, STUDY_THREADS.
# Refuses to start while targets.json still has nulls (run derive_targets.py first).
# A reboot kills nohup jobs: just run this script again, finished runs are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PDN_DATA="${PDN_DATA:-/DATA/Aurindum/Swarming/Dataset}"
export STUDY_THREADS="${STUDY_THREADS:-16}"
PYTHON="${PYTHON:-/DATA/Aurindum/conda/envs/work/bin/python}"

mkdir -p logs

if [ "${1:-}" = "--validate" ]; then
    exec "$PYTHON" -u run_all.py --validate
fi

PIDFILE="logs/run_exp2.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Exp 2 is already running (PID $(cat "$PIDFILE")); not starting a second copy." >&2
    exit 1
fi

nohup setsid "$PYTHON" -u run_all.py "$@" > logs/run_exp2.out 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown || true

echo "Exp 2 started, PID $(cat "$PIDFILE")"
echo "  python:  $PYTHON"
echo "  threads: STUDY_THREADS=$STUDY_THREADS  OMP/OPENBLAS/MKL/NUMEXPR=1"
echo "  log:     $HERE/logs/run_exp2.out   $HERE/logs/study.log"
echo "  results: $HERE/results/results_long.jsonl"
