RUN gpu_round2/run_q4_optimizer_comparison_full1
Search Q4: PSO vs DE vs GA vs SA on the same exact fitness and budget.

Created: 2026-10-12T23:44:41+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/search_quality/q4_optimizer_comparison.py  (git None, md5 2bb12860b52a194249825bad077b9690)
As run:  code/q4_optimizer_comparison__run_q4_optimizer_comparison_full1.py  (md5 2bb12860b52a194249825bad077b9690)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_q4_optimizer_comparison_full1/run.sh > runs/gpu_round2/run_q4_optimizer_comparison_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 23:44:42 IST  host: eeiitj123.iitj.ac.in  load: 1.07 1.03 1.05
END:   2026-10-13 00:02:58 IST  exit code 0
