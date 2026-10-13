RUN gpu_round2/run_s1_port_scaling_full1
Speed S1: per-evaluation time vs PDN ports (DDR3 reduced to 6..79 ports) + numpy vs fast PSO on ddr21/ddr79.

Created: 2026-10-12T19:46:19+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/speed/s1_port_scaling.py  (git None, md5 62186e0b132567b16e64d73f627260a5)
As run:  code/s1_port_scaling__run_s1_port_scaling_full1.py  (md5 62186e0b132567b16e64d73f627260a5)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'NUM_RUNS_NUMPY=2']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_s1_port_scaling_full1/run.sh > runs/gpu_round2/run_s1_port_scaling_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 19:46:20 IST  host: eeiitj123.iitj.ac.in  load: 0.64 1.67 1.37
END:   2026-10-12 20:26:13 IST  exit code 0
