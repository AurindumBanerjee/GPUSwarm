RUN gpu_experiments/run_baseline_naive40
GPU baselines: numpy (full inverse, BASELINE a), woodbury_batch, fast. 50x40, warm 0.4/0.0, complex128.

Created: 2026-10-09T09:51:39+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp00_baseline.py  (git None, md5 23d83a82845889fe4e215b834c5ee23a)
As run:  code/exp00_baseline__run_baseline_naive40.py  (md5 23d83a82845889fe4e215b834c5ee23a)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_baseline_naive40/run.sh > runs/gpu_experiments/run_baseline_naive40/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 09:51:40 IST  host: eeiitj123.iitj.ac.in  load: 0.10 0.46 0.59
END:   2026-10-09 10:49:41 IST  exit code 0
