RUN gpu_round2/run_s3_profile_gpu_decode_full1
Speed S3: time breakdown of fast + exact GPU-vectorised decoding.

Created: 2026-10-12T22:13:37+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/speed/s3_profile_gpu_decode.py  (git None, md5 2223c67ebac483b6671df792a7d1ea00)
As run:  code/s3_profile_gpu_decode__run_s3_profile_gpu_decode_full1.py  (md5 2223c67ebac483b6671df792a7d1ea00)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_s3_profile_gpu_decode_full1/run.sh > runs/gpu_round2/run_s3_profile_gpu_decode_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 22:13:38 IST  host: eeiitj123.iitj.ac.in  load: 6.39 6.61 6.72
END:   2026-10-12 22:17:39 IST  exit code 0
