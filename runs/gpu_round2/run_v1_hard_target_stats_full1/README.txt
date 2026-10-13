RUN gpu_round2/run_v1_hard_target_stats_full1
Validation V1: 100 runs at the hard target, Wilson CIs, Fisher tests, time per success.

Created: 2026-10-13T00:02:58+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/validation/v1_hard_target_stats.py  (git None, md5 736be2d34a44c16ea04f5ef6457f0d64)
As run:  code/v1_hard_target_stats__run_v1_hard_target_stats_full1.py  (md5 736be2d34a44c16ea04f5ef6457f0d64)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_v1_hard_target_stats_full1/run.sh > runs/gpu_round2/run_v1_hard_target_stats_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-13 00:02:59 IST  host: eeiitj123.iitj.ac.in  load: 1.05 1.06 1.05
END:   2026-10-13 01:04:39 IST  exit code 0
