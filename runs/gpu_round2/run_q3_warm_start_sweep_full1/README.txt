RUN gpu_round2/run_q3_warm_start_sweep_full1
Search Q3: warm-start fraction x jitter sweep.

Created: 2026-10-12T23:05:01+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/search_quality/q3_warm_start_sweep.py  (git None, md5 a63484037347a30587afbbfcbe25d42f)
As run:  code/q3_warm_start_sweep__run_q3_warm_start_sweep_full1.py  (md5 a63484037347a30587afbbfcbe25d42f)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_q3_warm_start_sweep_full1/run.sh > runs/gpu_round2/run_q3_warm_start_sweep_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 23:05:02 IST  host: eeiitj123.iitj.ac.in  load: 1.01 1.02 1.24
END:   2026-10-12 23:44:41 IST  exit code 0
