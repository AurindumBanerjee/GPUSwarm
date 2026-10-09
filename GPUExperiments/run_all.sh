#!/bin/bash
# GPUExperiments driver: baseline, experiments 1-6, compare -- one run folder each (reference/new_run.py), executed one after the other.
#   Partial re-run: --only incremental,macro --reuse <dirs of other finished runs> --baseline-tag <tag of the baseline run>
#   bash GPUExperiments/run_all.sh [--y PDN.mat] [--d decaps.mat] [--freq freq.mat] [--quick] [--tag NAME] [--t4 0.6,0.5,0.45]
# Launch detached (survives logout):
#   ssh -f EEDept 'cd /DATA/Aurindum/GPUSwarm && nohup setsid bash GPUExperiments/run_all.sh > GPUExperiments/run_all.log 2>&1 < /dev/null &'
# Results: runs/gpu_experiments/<id>/{output,summary,plots,...}. GPU work is pinned with taskset (GPU_CPUS, default 24-31);
# run it only when the GPU is idle (timings are not valid otherwise).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 1
Y=$ROOT/Data/Reference/y2.mat; D=$ROOT/Data/Reference/decaps.mat; F=$ROOT/Data/Reference/freq2.mat
WARM=${WARM:-1.0}; JIT=${JIT:-0.0}   # warm-start fraction / jitter (--warm / --jitter); defaults unchanged
ONLY=""; REUSE=""; BTAG=""; VF=""; QUICK=0; TAG=$(date +%m%d%H%M); T4=0.6,0.5,0.45; CPUS=${GPU_CPUS:-24-31}
CPU_SUMMARY=${CPU_SUMMARY:-/DATA/Aurindum/Swarming/runs/scratchbench/run08_warm100_5runs/summary/summary.csv}
while [ $# -gt 0 ]; do case $1 in
  --y) Y=$2; shift 2;; --d) D=$2; shift 2;; --freq) F=$2; shift 2;; --quick) QUICK=1; shift;; --tag) TAG=$2; shift 2;; --t4) T4=$2; shift 2;; --only) ONLY=$2; shift 2;; --reuse) REUSE=$2; shift 2;; --baseline-tag) BTAG=$2; shift 2;; --vf) VF=$2; shift 2;; --warm) WARM=$2; shift 2;; --jitter) JIT=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ $QUICK = 1 ] && TAG=${TAG}_quick
[ -z "$BTAG" ] && BTAG=$TAG
source /DATA/Aurindum/conda/etc/profile.d/conda.sh; conda activate gpuwork
echo "run_all: warm=$WARM jitter=$JIT tag=$TAG quick=$QUICK y=$Y d=$D freq=$F start=$(date)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
EXP=gpu_experiments; RUNS=$ROOT/runs/$EXP; DIRS=""
mk() {  # mk <name> <script> <desc> [extra --env args...]
  local name=$1 id="run_${1}_${TAG}"; local src=$2 desc=$3; shift 3
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then return 0; fi
  python reference/new_run.py --experiment $EXP --id "$id" --desc "$desc" --source GPUExperiments/$src \
     --env QUICK=$QUICK --env Y_PATH=$Y --env D_PATH=$D --env FREQ_PATH=$F \
     --env BASELINE_JSONL=$RUNS/run_baseline_${BTAG}/output/results_long.jsonl \
     --env WARM_START_FRACTION=$WARM --env WARM_START_JITTER=$JIT "$@" --taskset $CPUS --conda-env gpuwork > /dev/null || return 1
  cp GPUExperiments/gpuexp_common.py "$RUNS/$id/code/"
  echo "== $id start $(date)"; bash "$RUNS/$id/run.sh"; echo "== $id end $(date)"
  DIRS="$DIRS,$RUNS/$id"
}
mk baseline exp00_baseline.py "GPU baselines: numpy (full inverse, BASELINE a), woodbury_batch, fast. 50x40, warm $WARM/$JIT, complex128."
mk shared exp01_shared_threshold.py "Exp 1: shared-threshold run at the lowest target vs separate runs (fast engine)."
mk mixed exp02_mixed_precision.py "Exp 2: complex64 batched solve + complex128 refinement vs complex128."
mk incremental exp03_incremental.py "Exp 3: rank-1 incremental inverse updates for model-swap-only moves."
mk gpupso exp06_gpu_resident.py "Exp 6: GPU-resident PSO (GPU RNG) + numpy-RNG debug mode."
mk polish exp05_gradient_polish.py "Exp 5: gradient polish of the stage best through the exact Woodbury cost."
mk macro exp04_rational_macromodel.py "Exp 4: vector-fit macromodel on a dense grid, VDDQ_DDR3 ddr21 up to 300 MHz, 6695-model library." --env EXP4_TARGETS=$T4 ${VF:+--env VF_ORDERS=$VF}
id="run_compare_${TAG}"
python reference/new_run.py --experiment $EXP --id "$id" --desc "Cross-experiment comparison tables and plots (GPU numpy baseline a, CPU ScratchBench baseline b)." \
   --source GPUExperiments/exp07_compare.py --env QUICK=$QUICK --env GROUP_DIRS=${REUSE:+$REUSE,}${DIRS#,} --env CPU_SUMMARY=$CPU_SUMMARY \
   --env BASELINE_JSONL=$RUNS/run_baseline_${BTAG}/output/results_long.jsonl --conda-env gpuwork > /dev/null
cp GPUExperiments/gpuexp_common.py "$RUNS/$id/code/"
bash "$RUNS/$id/run.sh"
echo "run_all done $(date)"
