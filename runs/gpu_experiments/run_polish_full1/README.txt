RUN gpu_experiments/run_polish_full1
Exp 5: gradient polish of the stage best through the exact Woodbury cost.

Created: 2026-10-09T02:36:56+0530  on eeiitj123.iitj.ac.in
Source: GPUExperiments/exp05_gradient_polish.py  (git None, md5 baf2ad0e88491aec83a3332bf4f055be)
As run:  code/exp05_gradient_polish__run_polish_full1.py  (md5 baf2ad0e88491aec83a3332bf4f055be)

Changes vs source (only these lines differ):
(identical to source)
Execution: args ''; env ['QUICK=0', 'Y_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat', 'D_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat', 'FREQ_PATH=/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat', 'BASELINE_JSONL=/DATA/Aurindum/GPUSwarm/runs/gpu_experiments/run_baseline_full1/output/results_long.jsonl', 'WARM_START_FRACTION=1.0', 'WARM_START_JITTER=0.0']; taskset 24-31; conda env gpuwork

Launch: nohup setsid bash runs/gpu_experiments/run_polish_full1/run.sh > runs/gpu_experiments/run_polish_full1/logs/driver.log 2>&1 < /dev/null &

---- timeline ----
START: 2026-10-09 02:36:57 IST  host: eeiitj123.iitj.ac.in  load: 1.11 1.22 1.21
END:   2026-10-09 02:45:13 IST  exit code 0

---- results / notes (written after the run) ----
Settings: Reference 21-port PDN (1391 freqs), 3348 capacitor models, 50 particles x 40 iterations, MAX_CAPS 20, TARGET_PORT 0,
targets 0.05/0.045/0.04 ohm, 20 paired-seed runs per target, complex128, warm start fraction 1.0 / jitter 0.0 (= CPU run08; GPUBench2's
0.4/0.02 was NOT used). Pinned with taskset -c 24-31, BLAS 1 thread. Concurrent jobs: none on the GPU (nvidia-smi showed no compute
process at launch; an idle podman Ollama container exists on the machine); no CPU job on cores 24-31 during the run. Timings are valid.
Target 0.04 is never reached by the baseline with this warm start (0/20): those runs use the full budget (40000 evaluations).
Exp 5. Polish at the end of a stage whose best is within 1.3x of the target (416 events). Continuous relaxation: mean -11.3 % of the PSO
stage best (max 16 %); snapping to the nearest library model: +95 % worse (never used alone); exact local sweep over the 8 nearest
candidates: mean 0.14 % better, better than PSO in 17.8 % of events, never worse (0 events). Outcome: success at 0.04: 4/20 vs 0/20; caps
at 0.045: 6.4 vs 6.8; but time per run at 0.045 is 2.8 s vs 0.57 s (polish ~0.73 s per event). Re-scoring note: best_minz is the minimum over
stages, the stored placement is the last stage's; the difference rescored_minz - best_minz is 0 on every successful run and positive
(<= 2e-4 ohm) on failed runs, as expected.
