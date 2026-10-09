RUN gpu_experiments/run_incremental_full2
Exp 3: rank-1 incremental inverse updates for model-swap-only moves.

Created: 2026-10-09T09:28:29+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp03_incremental.py  (git None, md5 89726e1cb9da9f0c064e51b102306c86)
As run:  code/exp03_incremental__run_incremental_full2.py  (md5 89726e1cb9da9f0c064e51b102306c86)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_incremental_full2/run.sh > runs/gpu_experiments/run_incremental_full2/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 09:28:30 IST  host: eeiitj123.iitj.ac.in  load: 0.10 0.08 0.06
END:   2026-10-09 09:45:24 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 3 (code with the added incr_safe arm; baseline trajectories come from run_baseline_full1). Eligible for a rank-1 update (same ports, <= k/2 model
swaps, <= 8 steps since the last inverse): 17.9 % of all evaluations (30.5 % of those not served by the cache). Max cost deviation of the
updated cost vs the exact Woodbury cost: 2.9e-13.
EXACTNESS: plain 'incremental' (rank-1 + cache) does NOT reproduce the baseline: identical caps/evals rate 0.767, max rel. best_minz deviation
4.5 % (one run even reached 0.04). Cause (measured: 1760 of 3348 library models are exact duplicates of another model): distinct
configurations with identical true cost tie bit-for-bit in the baseline (no update), while the rank-1 values differ by ~1e-13, which flips
cost < pbest and changes the trajectory. 'incr_safe' re-scores exactly every candidate that could beat/tie its pbest: identical rate 1.0,
max deviation 0. Speed: incremental 2.62x over uncached woodbury_batch but 0.95x vs the cached Woodbury (wb_cache); incr_safe 1.73x /
0.625x. Conclusion: rank-1 updates give no speed-up over a batched, cached Woodbury solve at k <= 20 and cannot be exact without exact re-scoring.
