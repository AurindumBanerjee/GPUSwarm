RUN gpu_experiments/run_mixed_naive40
Exp 2: complex64 batched solve + complex128 refinement vs complex128.

Created: 2026-10-09T10:53:01+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp02_mixed_precision.py  (git None, md5 06b71270e67a116acd996c88cf0b05dd)
As run:  code/exp02_mixed_precision__run_mixed_naive40.py  (md5 06b71270e67a116acd996c88cf0b05dd)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_mixed_naive40/run.sh > runs/gpu_experiments/run_mixed_naive40/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 10:53:02 IST  host: eeiitj123.iitj.ac.in  load: 0.92 1.03 1.04
END:   2026-10-09 11:03:53 IST  exit code 0
