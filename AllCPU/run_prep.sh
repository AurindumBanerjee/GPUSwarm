#!/bin/bash
# AllCPU preparation + tests on the server.   nohup ./run_prep.sh > prep.out 2>&1 &
# Edit EXCLUDE to the logical CPUs your own jobs sit on ("ps -eo pid,psr,pcpu,cmd --sort=-pcpu | head" shows them).
cd "$(dirname "$0")"
export GPUSWARM_ROOT=${GPUSWARM_ROOT:-/DATA/Aurindum/GPUSwarm}
export ALLCPU_DATASET=${ALLCPU_DATASET:-/DATA/Aurindum/Swarming/Dataset}
export EXCLUDE=${EXCLUDE:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 common/prep_all.py "$@"
