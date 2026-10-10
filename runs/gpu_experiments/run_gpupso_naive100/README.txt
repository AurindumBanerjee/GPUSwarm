RUN gpu_experiments/run_gpupso_naive100
Exp 6: GPU-resident PSO (GPU RNG) + numpy-RNG debug mode.

Created: 2026-10-09T13:46:14+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp06_gpu_resident.py  (git None, md5 05530fe36eac599b0d266f90022e2b52)
As run:  code/exp06_gpu_resident__run_gpupso_naive100.py  (md5 05530fe36eac599b0d266f90022e2b52)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive100/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_gpupso_naive100/run.sh > runs/gpu_experiments/run_gpupso_naive100/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 13:46:15 IST  host: eeiitj123.iitj.ac.in  load: 1.02 1.05 1.05
END:   2026-10-09 14:05:03 IST  exit code 0
