RUN gpu_experiments/run_macro_naive40
Exp 4: vector-fit macromodel on a dense grid, VDDQ_DDR3 ddr21 up to 300 MHz, 6695-model library.

Created: 2026-10-09T11:57:44+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp04_rational_macromodel.py  (git None, md5 912f8e754569e4881529dd44c670398d)
As run:  code/exp04_rational_macromodel__run_macro_naive40.py  (md5 912f8e754569e4881529dd44c670398d)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive40/output/results_long.jsonl', 'WARM_START_FRACTION=0.4', 'WARM_START_JITTER=0.0', 'EXP4_TARGETS=0.9,0.85,0.8']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_macro_naive40/run.sh > runs/gpu_experiments/run_macro_naive40/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 11:57:44 IST  host: eeiitj123.iitj.ac.in  load: 0.94 0.99 1.04
END:   2026-10-09 12:10:39 IST  exit code 0
