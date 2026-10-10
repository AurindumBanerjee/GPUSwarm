RUN gpu_experiments/run_compare_naive100
Cross-experiment comparison tables and plots (GPU numpy baseline a, CPU ScratchBench baseline b).

Created: 2026-10-09T14:26:37+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp07_compare.py  (git None, md5 16f5c2cbd2883d7e1e309336cb286cb3)
As run:  code/exp07_compare__run_compare_naive100.py  (md5 16f5c2cbd2883d7e1e309336cb286cb3)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GROUP_DIRS=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_shared_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_mixed_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_incremental_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_gpupso_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_polish_naive100,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_macro_naive100', 'CPU_SUMMARY=/DATA/Aurindum/Swarming/runs/scratchbench/run08_warm100_5runs/summary/summary.csv', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_naive100/output/results_long.jsonl']; taskset none; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_compare_naive100/run.sh > runs/gpu_experiments/run_compare_naive100/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 14:26:38 IST  host: eeiitj123.iitj.ac.in  load: 1.02 1.12 1.11
END:   2026-10-09 14:26:42 IST  exit code 0
