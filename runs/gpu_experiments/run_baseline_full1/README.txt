RUN gpu_experiments/run_baseline_full1
GPU baselines: numpy (full inverse, BASELINE a), woodbury_batch, fast. 50x40, warm 1.0/0.0, complex128.

Created: 2026-10-09T00:45:45+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp00_baseline.py  (git None, md5 23d83a82845889fe4e215b834c5ee23a)
As run:  code/exp00_baseline__run_baseline_full1.py  (md5 23d83a82845889fe4e215b834c5ee23a)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_baseline_full1/run.sh > runs/gpu_experiments/run_baseline_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 00:45:46 IST  host: eeiitj123.iitj.ac.in  load: 1.57 1.25 0.76
END:   2026-10-09 01:49:10 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Baseline (a) = arm 'numpy' (full inverse per particle on this GPU). Mean total time per run: 0.05: 20.7 s; 0.045: 31.8 s;
0.04: 115.5 s. woodbury_batch: 0.30 / 0.59 / 16.2 s (68x / 54x / 7.1x). fast (cache+pruning): 0.38 / 0.58 / 3.75 s (55x / 55x / 31x).
All three arms follow the same trajectory (trajectory match 1.0, strict incl. n_evals 1.0; verification vs CPU full inverse 5e-15).
