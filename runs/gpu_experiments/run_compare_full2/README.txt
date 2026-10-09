RUN gpu_experiments/run_compare_full2
Cross-experiment comparison tables and plots (GPU numpy baseline a, CPU ScratchBench baseline b).

Created: 2026-10-09T09:47:18+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp07_compare.py  (git None, md5 16f5c2cbd2883d7e1e309336cb286cb3)
As run:  code/exp07_compare__run_compare_full2.py  (md5 16f5c2cbd2883d7e1e309336cb286cb3)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GROUP_DIRS=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_shared_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_mixed_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_gpupso_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_polish_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_incremental_full2,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_macro_full2', 'CPU_SUMMARY=/DATA/Aurindum/Swarming/runs/scratchbench/run08_warm100_5runs/summary/summary.csv', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl']; taskset none; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_compare_full2/run.sh > runs/gpu_experiments/run_compare_full2/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 09:47:19 IST  host: eeiitj123.iitj.ac.in  load: 1.00 1.03 0.77
END:   2026-10-09 09:47:23 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Final comparison. Baseline (a) = GPU numpy arm of run_baseline_full1; baseline (b) = CPU ScratchBench run08 (50 x 15 iterations, 5 runs;
different budget, shown as separate rows; its speed-up column is time-to-target ratio only where both succeed). Reuses full1 runs of
baseline/shared/mixed/gpupso/polish and full2 of incremental/macro.
