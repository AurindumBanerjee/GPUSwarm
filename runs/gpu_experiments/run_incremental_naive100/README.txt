RUN gpu_experiments/run_incremental_naive100
Exp 3: rank-1 incremental inverse updates for model-swap-only moves.

Created: 2026-10-09T13:29:19+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp03_incremental.py  (git None, md5 89726e1cb9da9f0c064e51b102306c86)
As run:  code/exp03_incremental__run_incremental_naive100.py  (md5 89726e1cb9da9f0c064e51b102306c86)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive100/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_incremental_naive100/run.sh > runs/gpu_experiments/run_incremental_naive100/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 13:29:19 IST  host: eeiitj123.iitj.ac.in  load: 1.01 1.03 1.05
END:   2026-10-09 13:46:14 IST  exit code 0
