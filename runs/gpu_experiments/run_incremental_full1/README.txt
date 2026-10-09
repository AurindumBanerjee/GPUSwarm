RUN gpu_experiments/run_incremental_full1
Exp 3: rank-1 incremental inverse updates for model-swap-only moves.

Created: 2026-10-09T02:04:27+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp03_incremental.py  (git None, md5 a02f30a6ac1601736c1c8730a8d24c5e)
As run:  code/exp03_incremental__run_incremental_full1.py  (md5 a02f30a6ac1601736c1c8730a8d24c5e)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_incremental_full1/run.sh > runs/gpu_experiments/run_incremental_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 02:04:28 IST  host: eeiitj123.iitj.ac.in  load: 1.05 1.07 1.05
END:   2026-10-09 02:18:06 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
SUPERSEDED by run_incremental_full2 (which adds the exact-recheck arm). Kept as the record of the first attempt.
Plain rank-1 + cache arm: trajectory identity only 76.7 % (max rel. best_minz deviation 4.5 %), see full2 for the explanation.
