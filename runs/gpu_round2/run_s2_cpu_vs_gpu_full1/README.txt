RUN gpu_round2/run_s2_cpu_vs_gpu_full1
Speed S2: algorithm vs hardware gain -- numpy / woodbury_batch / fast on CPU and GPU.

Created: 2026-10-12T20:26:13+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/speed/s2_cpu_vs_gpu.py  (git None, md5 25f96fac917a5bcbde9ab569cb420952)
As run:  code/s2_cpu_vs_gpu__run_s2_cpu_vs_gpu_full1.py  (md5 25f96fac917a5bcbde9ab569cb420952)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'CPU_THREADS=8']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_s2_cpu_vs_gpu_full1/run.sh > runs/gpu_round2/run_s2_cpu_vs_gpu_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 20:26:14 IST  host: eeiitj123.iitj.ac.in  load: 1.01 1.04 1.08
END:   2026-10-12 22:13:37 IST  exit code 0
