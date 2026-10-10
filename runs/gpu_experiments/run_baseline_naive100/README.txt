RUN gpu_experiments/run_baseline_naive100
GPU baselines: numpy (full inverse, BASELINE a), woodbury_batch, fast. 50x40, warm 1.0/0.0, complex128.

Created: 2026-10-09T12:10:45+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp00_baseline.py  (git None, md5 23d83a82845889fe4e215b834c5ee23a)
As run:  code/exp00_baseline__run_baseline_naive100.py  (md5 23d83a82845889fe4e215b834c5ee23a)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive100/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_baseline_naive100/run.sh > runs/gpu_experiments/run_baseline_naive100/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 12:10:45 IST  host: eeiitj123.iitj.ac.in  load: 0.97 1.02 1.04
END:   2026-10-09 13:14:11 IST  exit code 0
