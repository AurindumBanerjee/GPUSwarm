RUN gpu_experiments/run_polish_naive40
Exp 5: gradient polish of the stage best through the exact Woodbury cost.

Created: 2026-10-09T11:50:27+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp05_gradient_polish.py  (git None, md5 baf2ad0e88491aec83a3332bf4f055be)
As run:  code/exp05_gradient_polish__run_polish_naive40.py  (md5 baf2ad0e88491aec83a3332bf4f055be)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_polish_naive40/run.sh > runs/gpu_experiments/run_polish_naive40/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 11:50:28 IST  host: eeiitj123.iitj.ac.in  load: 1.01 1.05 1.07
END:   2026-10-09 11:57:43 IST  exit code 0
