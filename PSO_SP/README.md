# PSO_SP

Functional PSO decap placement on the SP_Sep2026 boards, to 1 GHz.

```
pso_sp_core.py      shared engine
pso_mphy.py         MPHY package  - 5 rails + the 73-port full network
pso_vddq_ddr3.py    VDDQ DDR3     - 21-port and 79-port
```

## Where it comes from

The skeleton is `CPUTest/SB.py`, with its FIX 1-13 semantics kept verbatim:
per-`(run, n_caps)` reseeding (FIX 1), `floor(frac*K)` discretisation (FIX 2),
observation ports barred from placement in both the decode and the collision
resolver (FIX 3), warm start carrying the previous stage's best (FIX 4),
assembly only for methods that consume it (FIX 5), loud failure on unknown
methods (FIX 6), `perf_counter` (FIX 10), jitter plus a random remainder so the
warm block is not frozen (FIX 11), a resolver that tries every offset (FIX 12),
and `prev_best` reset when a stage yields nothing (FIX 13).

PSO constants are SB's unchanged: 50 particles, 15 iterations, w 0.9 to 0.4,
c1 = c2 = 1.5, 20 warm particles, jitter 0.02, base seed 12345.

Ported from `DecapStudy/`:

* **`Problem`** - ports come from the loaded matrix. The first `n_obs` indices
  are observation ports and never take a capacitor; pads are the rest, so
  `max_caps = N - n_obs`. Nothing is hardcoded. Several observation ports are
  supported, which is what `y_sp_mphy_full` (five) needs.
* **Multi-band objective** - B50, B200, BFULL. Every winning placement is
  re-scored on all three, so a result found on one band is comparable on the
  others. `RANK_BAND` picks the one PSO optimises.
* **Threaded evaluation** - `STUDY_THREADS`, with BLAS pinned to 1 before numpy
  imports. Results are consumed in particle order, so trajectories match serial.
* **Wall-clock limit** per run, flagged `timed_out` rather than silently cut.
* **One JSON record per run**, carrying the full placement (model, pad, part
  name and extrapolation slope for every capacitor).

## Data

`Data/` holds the inputs, all on one shared 2588-point grid
(110.89 Hz - 997.21 MHz). The board `freq_sp.mat` files are byte-identical to
`Decaps/sp_freq.mat`, so **nothing is resampled anywhere in this code**.

The library is 6695 shunt models (3348 legacy + 3347 AVX). There are no series
entries, so no fixture filtering is needed. Measured to ~99.8 MHz; above that
the inductive tail is extrapolated, which is **19.3% of the grid points**.

## Running

```
python pso_mphy.py --validate         # data, port rules, method agreement; no PSO
python pso_mphy.py                    # full run
python pso_vddq_ddr3.py

NUM_RUNS=3 STUDY_THREADS=16 python pso_mphy.py
METHODS=numpy,solve,sm RANK_BAND=BFULL python pso_vddq_ddr3.py
```

Environment: `NUM_RUNS` (10), `METHODS` (numpy), `RANK_BAND` (B50),
`STUDY_THREADS` (1), `TIME_LIMIT_S` (1800), `PDN_DATA` (`../Data`).

Results land in `results_<board>/results_long.jsonl`, with per-run folders
holding `run_N.log`, `run_plot.png`, `global_convergence.png` and
`impedance.png` (bare vs optimised, log-log, extrapolated region shaded).

By default `threshold=None`, so every run searches its full capacitor budget and
records the whole quality-versus-capacitor-count curve. Pass a threshold in ohm
to `run_pso` to stop on target instead - that is the mode for timing inversion
methods, where `time_to_target` is the metric.

## Bare peaks, for reference

| problem | N | pads | B50 | B200 | BFULL |
|---|---|---|---|---|---|
| mer1 | 16 | 15 | 1.627 | 6.529 | 30.69 |
| mer2 | 16 | 15 | 0.725 | 2.540 | 15.48 |
| pll1v0 | 15 | 14 | 1.663 | 16.58 | 39.95 |
| pll1v8 | 16 | 15 | 12.09 | 56.90 | 241.7 |
| mphyvdd | 10 | 9 | 1.453 | 33.70 | 33.70 |
| mphy_full | 73 | 68 | 12.09 | 56.90 | 241.7 |
| ddr21 | 21 | 20 | 0.2507 | 1.029 | 142.1 |
| ddr_full | 79 | 78 | 0.2507 | 1.029 | 142.1 |

`mphy_full` equals `pll1v8` because the objective is the worst of its five
observation ports and pll1v8 dominates. `ddr21` equals `ddr_full` because
dropping ports left OPEN cannot change the impedance at the observation port -
if those two ever differ, the sub-network reduction is wrong.

## Two things to watch

**mer1 and pll1v8 barely respond.** Their bare networks sit within a few percent
of the best achievable, because the observation port is at the die and package
inductance swamps the pads. Near-flat curves there are a property of the board.

**19.3% of the band rests on extrapolated capacitor models.** 682 of 6695 models
have a log-log slope outside 0.5-1.5 and 44 are negative, meaning their
impedance *falls* above 100 MHz, which is unphysical. PSO will favour exactly
those on BFULL. Every record carries `frac_caps_slope_outside_0p8_1p2`; if a
BFULL winner scores high there, re-run it with those models excluded before
believing the result.
