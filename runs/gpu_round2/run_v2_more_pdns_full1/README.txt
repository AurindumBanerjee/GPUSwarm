RUN gpu_round2/run_v2_more_pdns_full1
Validation V2: MPHY rails + DDR3 at relative targets (<= 50 MHz) + all MPHY rails combined.

Created: 2026-10-13T01:04:40+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/validation/v2_more_pdns.py  (git None, md5 c74fe65cc663eb3d1746e2739d1072bf)
As run:  code/v2_more_pdns__run_v2_more_pdns_full1.py  (md5 c74fe65cc663eb3d1746e2739d1072bf)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_v2_more_pdns_full1/run.sh > runs/gpu_round2/run_v2_more_pdns_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-13 01:04:40 IST  host: eeiitj123.iitj.ac.in  load: 0.97 0.94 1.03
END:   2026-10-13 02:29:40 IST  exit code 0
