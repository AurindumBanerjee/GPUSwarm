RUN gpu_round2/run_s4_batched_seeds_full1
Speed S4: all runs of a threshold in lock-step, one GPU call per capacitor-count group.

Created: 2026-10-12T22:17:39+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/speed/s4_batched_seeds.py  (git None, md5 3d8da07540a549fc0c4cae69c7cbcdb9)
As run:  code/s4_batched_seeds__run_s4_batched_seeds_full1.py  (md5 3d8da07540a549fc0c4cae69c7cbcdb9)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_s4_batched_seeds_full1/run.sh > runs/gpu_round2/run_s4_batched_seeds_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 22:17:40 IST  host: eeiitj123.iitj.ac.in  load: 1.10 3.52 5.42
END:   2026-10-12 22:26:56 IST  exit code 0
