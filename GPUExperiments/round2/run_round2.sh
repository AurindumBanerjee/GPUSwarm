#!/bin/bash
# Round-2 GPU experiments driver: one isolated run folder per experiment (reference/new_run.py), executed one after the other.
#   bash GPUExperiments/round2/run_round2.sh [--only s1,s2,...] [--quick] [--tag NAME] [--warm 0.4] [--jitter 0.02]
# Experiments (ids for --only):
#   speed:           s1 port scaling | s2 CPU vs GPU | s3 profile + GPU decode | s4 batched seeds
#   search_quality:  q1 polish variants | q2 library reduction | q3 warm-start sweep | q4 optimiser comparison
#   validation:      v1 hard-target statistics | v2 more PDNs + MPHY combined
#   floors:          v3 full-band floors per SP PDN -> round2/floors_full_band.json (then TARGET_MODE=floor uses it)
#   plots:           p1 before/after impedance from finished runs (needs RESULTS=..., see plots/README.md; not in the default list)
# Launch detached:
#   ssh -f EEDept 'cd /DATA/Aurindum/GPUSwarm && nohup setsid bash GPUExperiments/round2/run_round2.sh > GPUExperiments/round2/run_round2.log 2>&1 < /dev/null &'
# Results: runs/gpu_round2/<run id>/{summary,plots,output,logs}. Run only when the GPU is idle (timings).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
ONLY="s1,s2,s3,s4,q1,q2,q3,q4,v1,v2"        # v3 is not in the default list (long; run it first when floor targets are wanted); QUICK=0; TAG=$(date +%m%d%H%M); CPUS=${GPU_CPUS:-24-31}
WARM=${WARM:-0.4}; JIT=${JIT:-0.0}
while [ $# -gt 0 ]; do case $1 in
  --only) ONLY=$2; shift 2;; --quick) QUICK=1; shift;; --tag) TAG=$2; shift 2;; --warm) WARM=$2; shift 2;; --jitter) JIT=$2; shift 2;;
  *=*) export "$1"; shift;;                                 # KEY=VALUE -> environment (e.g. DATASETS=mer2,ddr21)
  *) echo "unknown arg $1"; exit 2;; esac; done
[ $QUICK = 1 ] && TAG=${TAG}_quick
source /DATA/Aurindum/conda/etc/profile.d/conda.sh; conda activate gpuwork
echo "run_round2: only=$ONLY warm=$WARM jitter=$JIT tag=$TAG quick=$QUICK start=$(date)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
EXP=gpu_round2; RUNS=$ROOT/runs/$EXP; R2=GPUExperiments/round2
PASS=(); for v in SP_FMAX TARGET_MODE DATASETS DATASET FLOOR_LEVELS FLOORS_JSON FLOOR_MULTS NUM_RUNS NUM_RUNS_NUMPY NUM_RUNS_HARD LIB_SOURCE BANDS V3_RUNS V3_ITERS V3_CAPS_DDR79; do
  [ -n "${!v}" ] && PASS+=(--env "$v=${!v}"); done
IDSUF=""
mk() {  # mk <id> <script> <desc> [extra --env args...]   (IDSUF appended to the run id, e.g. the dataset)
  local key=$1 src=$2 desc=$3; shift 3
  [[ ",$ONLY," != *",$key,"* ]] && return 0
  local id="run_$(basename "${src%.py}")${IDSUF}_${TAG}"
  python reference/new_run.py --experiment $EXP --id "$id" --desc "$desc" --source $R2/$src      --env QUICK=$QUICK --env GPUSWARM_ROOT=$ROOT --env WARM_START_FRACTION=$WARM --env WARM_START_JITTER=$JIT      --env Y_PATH=$ROOT/Data/Reference/y2.mat --env D_PATH=$ROOT/Data/Reference/decaps.mat --env FREQ_PATH=$ROOT/Data/Reference/freq2.mat      "${PASS[@]}" "$@" --taskset $CPUS --conda-env gpuwork > /dev/null || { echo "!! could not create $id"; return 1; }
  cp GPUExperiments/gpuexp_common.py GPUExperiments/exp05_gradient_polish.py $R2/r2_common.py "$RUNS/$id/code/"
  echo "== $id start $(date)"; bash "$RUNS/$id/run.sh"; echo "== $id end $(date) ($(tail -1 $RUNS/$id/README.txt))"
}
# per-dataset runs: when DATASETS is given, q1/q2/q4 run once per dataset (they take a single DATASET)
mkds() {  # mkds <key> <script> <desc>
  if [ -n "$DATASETS" ]; then
    for ds in ${DATASETS//,/ }; do IDSUF="_$ds" mk "$1" "$2" "$3" --env DATASET=$ds --env MAX_CAPS=$(pads $ds); done; IDSUF=""
  else mk "$1" "$2" "$3"; fi
}
pads() { case $1 in mer1|mer2|pll1v8) echo 15;; pll1v0) echo 14;; mphyvdd) echo 9;; ddr21) echo 20;; ddr79) echo 40;; *) echo 20;; esac; }
mk s1 speed/s1_port_scaling.py        "Speed S1: per-evaluation time vs PDN ports (DDR3 reduced to 6..79 ports) + numpy vs fast PSO on ddr21/ddr79." --env NUM_RUNS_NUMPY=${S1_NUMPY_RUNS:-2}
mk s2 speed/s2_cpu_vs_gpu.py          "Speed S2: algorithm vs hardware gain -- numpy / woodbury_batch / fast on CPU and GPU." --env CPU_THREADS=8
mk s3 speed/s3_profile_gpu_decode.py  "Speed S3: time breakdown of fast + exact GPU-vectorised decoding."
mk s4 speed/s4_batched_seeds.py       "Speed S4: all runs of a threshold in lock-step, one GPU call per capacitor-count group."
mkds q1 search_quality/q1_polish_variants.py      "Search Q1: cheaper gradient-polish variants, time per success."
mkds q2 search_quality/q2_library_reduction.py    "Search Q2: dominance-pruned and clustered capacitor libraries."
mk q3 search_quality/q3_warm_start_sweep.py     "Search Q3: warm-start fraction x jitter sweep."
mkds q4 search_quality/q4_optimizer_comparison.py "Search Q4: PSO vs DE vs GA vs SA on the same exact fitness and budget."
mk v1 validation/v1_hard_target_stats.py        "Validation V1: 100 runs at the hard target, Wilson CIs, Fisher tests, time per success."
mk v3 validation/v3_full_band_floors.py        "Validation V3: full-band (and <=50 MHz) floors per SP PDN with GA / PSO+polish on the full and dominance-pruned library; writes floors_full_band.json."
mk v2 validation/v2_more_pdns.py                "Validation V2: MPHY rails + DDR3 at relative targets (<= 50 MHz) + all MPHY rails combined."
if [[ ",$ONLY," == *",p1,"* ]]; then
  mk p1 plots/p1_impedance_before_after.py "Plots P1: bare vs decap |Z11| from finished runs." --env RESULTS=$RESULTS --env DATASET=${DATASET:-ref21} --env ALL_RUNS=${ALL_RUNS:-0}
fi
echo "run_round2 done $(date)"
