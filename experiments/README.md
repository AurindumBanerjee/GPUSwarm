# GPUBench experiments

Six reformulations of the decap-placement benchmark, one folder each, in the
order proposed. All share `core/` (the GPUBench2 engine as a library):
`gbcore.py` (data, Woodbury fitness, evaluators, PSO, analysis) and
`macromodel.py` (vector fitting, R-L-C fit of the shunt decaps).

Decaps are modelled as SHUNT elements (pad to ground, admittance added to
the diagonal of Y). Use the shunt library: `decaps_sp.mat` with the SP
datasets, `decaps.mat` with the reference `y2.mat`.

| folder | idea | exact vs baseline? |
|---|---|---|
| 01_shared_threshold_runs | one run per (method, run) records every threshold crossing | yes, identical per-threshold results |
| 02_mixed_precision_refinement | complex64 LU + FP64 refinement; complex64 pruning screen | yes (refined to FP64; screen margin) |
| 03_incremental_woodbury_updates | rank-1 Sherman-Morrison updates of stored M^-1 when only models change | yes up to rounding (drift reported) |
| 04_macromodel_rational_peak | vector-fitted Z0 + RLC decaps, grid+zoom peak, optional extrapolation | no: different objective, re-scored on table |
| 05_continuous_relaxation_polish | gradient polish in (R,L,C) space, snap to real parts, exact re-score | no: changes the search, all scores exact |
| 06_gpu_resident_pso | whole PSO on GPU, vectorised decode/port resolution | no: different RNG stream |

Every script accepts the common options (`--y --d --freq --fmax --fmin
--methods --targets --runs --out --quick ...`) and writes
`results_long.csv`, `summary_table.csv`, `statistics.txt`, `timeouts.txt`,
`verification.json`, `figs/` under `--out`.

Quick smoke test of everything: `bash run_all.sh --quick`
