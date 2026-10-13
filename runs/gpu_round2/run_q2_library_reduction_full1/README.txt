RUN gpu_round2/run_q2_library_reduction_full1
Search Q2: dominance-pruned and clustered capacitor libraries.

Created: 2026-10-12T22:56:35+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/search_quality/q2_library_reduction.py  (git None, md5 3f2e9a4d43b1940d9c5a65a325f90c90)
As run:  code/q2_library_reduction__run_q2_library_reduction_full1.py  (md5 3f2e9a4d43b1940d9c5a65a325f90c90)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_q2_library_reduction_full1/run.sh > runs/gpu_round2/run_q2_library_reduction_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 22:56:36 IST  host: eeiitj123.iitj.ac.in  load: 0.85 0.96 1.38
END:   2026-10-12 23:05:01 IST  exit code 0
