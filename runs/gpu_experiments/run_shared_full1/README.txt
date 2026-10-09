RUN gpu_experiments/run_shared_full1
Exp 1: shared-threshold run at the lowest target vs separate runs (fast engine).

Created: 2026-10-09T01:49:10+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp01_shared_threshold.py  (git None, md5 b4bdaf954a93547a766176550fe8039d)
As run:  code/exp01_shared_threshold__run_shared_full1.py  (md5 b4bdaf954a93547a766176550fe8039d)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_shared_full1/run.sh > runs/gpu_experiments/run_shared_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 01:49:11 IST  host: eeiitj123.iitj.ac.in  load: 1.06 1.03 1.05
END:   2026-10-09 01:52:06 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 1. Crossing records of the shared run (stage caps, iteration, logical evaluations, cost) vs separate runs: 60/60 identical
(rate 1.0). Total benchmark time separate 94.98 s vs shared 75.62 s = 1.256x, NOT ~4x: the 0.04 run never succeeds and always costs the whole
budget, so the two shorter targets only add ~20 %. The ~4x hypothesis does not hold for this setting; it would apply if all targets were
reached at comparable cost.
