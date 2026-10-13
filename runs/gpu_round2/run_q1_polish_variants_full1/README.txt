RUN gpu_round2/run_q1_polish_variants_full1
Search Q1: cheaper gradient-polish variants, time per success.

Created: 2026-10-12T22:26:56+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/round2/search_quality/q1_polish_variants.py  (git None, md5 15dccf744ba197381d28052a6ddda337)
As run:  code/q1_polish_variants__run_q1_polish_variants_full1.py  (md5 15dccf744ba197381d28052a6ddda337)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_round2/run_q1_polish_variants_full1/run.sh > runs/gpu_round2/run_q1_polish_variants_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-12 22:26:57 IST  host: eeiitj123.iitj.ac.in  load: 1.04 1.45 3.45
END:   2026-10-12 22:56:35 IST  exit code 0
