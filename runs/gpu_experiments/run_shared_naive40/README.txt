RUN gpu_experiments/run_shared_naive40
Exp 1: shared-threshold run at the lowest target vs separate runs (fast engine).

Created: 2026-10-09T10:49:41+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp01_shared_threshold.py  (git None, md5 b4bdaf954a93547a766176550fe8039d)
As run:  code/exp01_shared_threshold__run_shared_naive40.py  (md5 b4bdaf954a93547a766176550fe8039d)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_shared_naive40/run.sh > runs/gpu_experiments/run_shared_naive40/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 10:49:42 IST  host: eeiitj123.iitj.ac.in  load: 1.06 1.09 1.06
END:   2026-10-09 10:53:01 IST  exit code 0
