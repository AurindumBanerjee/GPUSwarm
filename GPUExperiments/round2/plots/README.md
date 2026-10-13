# Plots

`p1_impedance_before_after.py` draws |Z11| without decaps and with the placed decaps on the same axes, from any finished run
(round 1 or 2; every record stores `best_models` / `best_ports`):

    RESULTS=runs/gpu_experiments/run_baseline_<tag>,runs/gpu_experiments/run_polish_<tag> DATASET=ref21 ALL_RUNS=1 \
        bash GPUExperiments/round2/run_round2.sh --only p1

Outputs: `<method>_T<target>_best.png` (best run), `compare_T<target>.png` (best run of every method together),
`<method>_T<target>_allruns.png` (every run, green = met target) with `ALL_RUNS=1`. `DATASET`, `SP_FMAX` and `LIB_SOURCE`
must match how the runs loaded the data. The round-2 experiments already save their own before/after plots.
