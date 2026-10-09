RUN gpu_experiments/run_gpupso_full1
Exp 6: GPU-resident PSO (GPU RNG) + numpy-RNG debug mode.

Created: 2026-10-09T02:18:06+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp06_gpu_resident.py  (git None, md5 05530fe36eac599b0d266f90022e2b52)
As run:  code/exp06_gpu_resident__run_gpupso_full1.py  (md5 05530fe36eac599b0d266f90022e2b52)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_gpupso_full1/run.sh > runs/gpu_experiments/run_gpupso_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 02:18:07 IST  host: eeiitj123.iitj.ac.in  load: 1.00 1.01 1.05
END:   2026-10-09 02:36:56 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 6. Device decode + collision resolution vs CPU on 200 swarms (half heavily colliding): 0 differing entries of 216000.
gpu_debug (numpy draws fed to the device): trajectory match 1.0, strict (n_evals, final stage) 1.0 vs the numpy baseline and vs woodbury_batch.
gpu_rng (GPU generator, outcome-judged): success 0.05 / 0.045: 20/20 (caps 4.8 / 6.65 vs 4.75 / 6.8), 0.04: 3/20 (baseline 0/20; caps 10),
so quality is equivalent (random-seed variation), not identical. Speed: gpu_rng 1.09x over host-driven woodbury_batch, 0.30x vs 'fast'
(which skips most solves via cache+pruning); gpu_debug 0.95x / 0.26x. The cost is dominated by the batched solve, not by host logic.
