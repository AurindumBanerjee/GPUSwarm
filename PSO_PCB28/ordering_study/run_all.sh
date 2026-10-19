#!/bin/bash
# Paired ordering study. Same seeds for every order. ~ NUM_RUNS x 4 PSO runs; use nohup/setsid on the server.
cd "$(dirname "$0")" && python make_orders.py && for o in srf cap part random; do
  ORDER=$o NUM_RUNS=${NUM_RUNS:-10} STUDY_THREADS=${STUDY_THREADS:-4} python pso_pcb28_order.py > run_$o.log 2>&1
done; python compare_orders.py
