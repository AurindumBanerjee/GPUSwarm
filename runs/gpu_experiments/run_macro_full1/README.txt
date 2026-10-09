RUN gpu_experiments/run_macro_full1
Exp 4: vector-fit macromodel on a dense grid, VDDQ_DDR3 ddr21 up to 300 MHz, 6695-model library.

Created: 2026-10-09T02:45:13+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp04_rational_macromodel.py  (git None, md5 912f8e754569e4881529dd44c670398d)
As run:  code/exp04_rational_macromodel__run_macro_full1.py  (md5 912f8e754569e4881529dd44c670398d)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0', 'EXP4_TARGETS=0.6,0.5,0.45']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_macro_full1/run.sh > runs/gpu_experiments/run_macro_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 02:45:14 IST  host: eeiitj123.iitj.ac.in  load: 0.97 1.02 1.11
END:   2026-10-09 03:14:23 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
SUPERSEDED by run_macro_full2: targets 0.6/0.5/0.45 were all below the achievable floor (~0.763 ohm) -> 0 successes (uninformative). It also
scanned up to 160 poles (fit time 750 s) without improvement.
