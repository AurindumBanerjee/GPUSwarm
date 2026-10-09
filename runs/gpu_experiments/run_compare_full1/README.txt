RUN gpu_experiments/run_compare_full1
Cross-experiment comparison tables and plots (GPU numpy baseline a, CPU ScratchBench baseline b).

Created: 2026-10-09T03:14:23+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp07_compare.py  (git None, md5 16f5c2cbd2883d7e1e309336cb286cb3)
As run:  code/exp07_compare__run_compare_full1.py  (md5 16f5c2cbd2883d7e1e309336cb286cb3)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'GROUP_DIRS=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_shared_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_mixed_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_incremental_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_gpupso_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_polish_full1,/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_macro_full1', 'CPU_SUMMARY=/DATA/Aurindum/Swarming/runs/scratchbench/run08_warm100_5runs/summary/summary.csv', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl']; taskset none; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_compare_full1/run.sh > runs/gpu_experiments/run_compare_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 03:14:23 IST  host: eeiitj123.iitj.ac.in  load: 1.00 1.01 1.05
END:   2026-10-09 03:14:27 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Compare run built from full1 (includes the SUPERSEDED incremental/macro full1). Use run_compare_full2.
