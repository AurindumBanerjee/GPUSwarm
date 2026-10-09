RUN gpu_experiments/run_macro_full2
Exp 4: vector-fit macromodel on a dense grid, VDDQ_DDR3 ddr21 up to 300 MHz, 6695-model library.

Created: 2026-10-09T09:45:24+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp04_rational_macromodel.py  (git None, md5 912f8e754569e4881529dd44c670398d)
As run:  code/exp04_rational_macromodel__run_macro_full2.py  (md5 912f8e754569e4881529dd44c670398d)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0', 'EXP4_TARGETS=1.144,0.953,0.839', 'VF_ORDERS=24']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_macro_full2/run.sh > runs/gpu_experiments/run_macro_full2/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 09:45:25 IST  host: eeiitj123.iitj.ac.in  load: 1.01 1.04 0.74
END:   2026-10-09 09:47:18 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 4. Dataset: VDDQ_DDR3 ddr21 (21 ports) + combined library (6695 models), band 0-300 MHz (2327 grid pts; caps extrapolated above 99.8 MHz),
dense grid x4 (9305 pts). Targets = 1.1x/1.25x/1.5x the 20-cap floor 0.763 ohm: 0.839/0.953/1.144 (settings: --t4, --vf 24 = 48 poles).
PDN vector fit (48 poles): max error 0.7 % of each element's peak, Z11 error 0.1 % of its peak but ~16 % relative to local |Z11| at deep
nulls; unstable poles are flipped (11 per relocation iteration; the iteration does not settle -> NOT a clean stable fit); tolerance 2e-3 never met.
Capacitors: measured data + R-L-C tail above 99.8 MHz (pure R-L-C fit: 10 % median error, rejected). Macro vs exact cost on random configs:
median 0.34 %, max 3.4 %; at the PSO optima the macro cost is 1.9-2.4 % above the exact one. Outcome (20 runs): every run succeeded on
its own model AND on the exact re-score (original model, original grid). Caps tab/macro: 2.6/3.05 (0.839), 2.0/2.0, 1.0/1.0; the macromodel
is conservative so it needs slightly more caps at the tightest target. Dense-grid peak / grid peak = 1.000 for all placements: the peak is at
the 300 MHz band edge, and the grid (0.385 MHz step) resolves the interior resonances, so this dataset shows NO between-grid-point effect.
Extended range beyond 300 MHz (cavity resonances to 1 GHz) was not optimised: README of Data/MPHY says capacitors cannot move them.
