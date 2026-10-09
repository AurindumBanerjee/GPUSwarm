RUN gpu_experiments/run_mixed_full1
Exp 2: complex64 batched solve + complex128 refinement vs complex128.

Created: 2026-10-09T01:52:06+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp02_mixed_precision.py  (git None, md5 06b71270e67a116acd996c88cf0b05dd)
As run:  code/exp02_mixed_precision__run_mixed_full1.py  (md5 06b71270e67a116acd996c88cf0b05dd)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_mixed_full1/run.sh > runs/gpu_experiments/run_mixed_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 01:52:07 IST  host: eeiitj123.iitj.ac.in  load: 1.00 1.01 1.05
END:   2026-10-09 02:04:27 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 2. complex64 LU + 2 refinement steps in complex128 (+complex128 fallback if residual > 1e-12). Max relative cost deviation vs
complex128 on random configurations (k=1..20): 6.3e-16 (target <= 1e-10); fallback fraction 0; along the PSO trajectories max rel. best_minz
deviation 3.4e-16 and identical caps/evals rate 1.0 (trajectory match 1.0). Speed: plain batch arm 1.76x faster in total (0.04: 16.5 -> 8.6 s)
but SLOWER for the easy targets (0.05: 0.306 -> 0.468 s; small k, launch/refinement overhead). With cache+pruning ('fast') the net effect is
0.996x (no gain: the full solves that remain are few).
