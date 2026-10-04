# Decap-placement study - Experiment 2 (CPU)

Own copy of the Exp 1 code (`../DecapStudy/`, not imported, not touched). One scientific change:
**targets are no longer derived from assumed voltages and currents.** They come from `targets.json`,
which `derive_targets.py` fills from Exp 1's results once Exp 1 has finished.

Everything this study writes stays inside `DecapStudy2/`: `results/` (`results_long.jsonl`, tables,
`figs/`, `report.md`) and `logs/`. Exp 1's results are only ever read, by `derive_targets.py`.

## Order of work

```bash
# 0. (any time) dry check: loads every dataset, checks shapes + port rules, prints the resolved
#    targets and the planned cells; no PSO, writes nothing. Exits 3 while targets.json has nulls.
./run_exp2.sh --validate

# 1. after Exp 1 has finished (its results/report.md exists)
python derive_targets.py                    # floors x [1.1, 1.25, 1.5] -> targets.json
python derive_targets.py --multipliers 1.05 1.2 1.4     # other levels
python derive_targets.py --floor-source any             # floors from every optimisation band

# 2. the study, detached under nohup (resumable; a reboot just needs the same command again)
./run_exp2.sh
```

`run_exp2.sh` exports `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` before python starts and runs
`run_all.py` under `nohup setsid`. `STUDY_THREADS` (default 16) is the particle-level parallelism.
The study **refuses to start** (exit code 2) while any target in `targets.json` is null.

## Targets

`targets.json` holds, per key, one value per threshold level: `targets_ohm[key][level] = floor * level`.
Keys: the five MPHY rails (`mer1 mer2 pll1v0 pll1v8 mphyvdd`), and the two DDR3 PDN sizes
(`ddr21`, `ddr_full` - their capacitor budgets differ, so their floors do too).

The **floor** is the lowest peak |Z11| on the <=50 MHz band (B2) that any Exp 1 stage-2 run reached while
using its full capacitor budget (`max_caps == n_pads` and the search went through the last capacitor count).
Rails come from the rail-wise A1 runs; `ddr21` from A4, `ddr_full` from A5. Each multiplier is a threshold
level. A run stops when the **tightest** level (smallest multiplier) is met; every looser level is
recorded the first time any evaluated placement meets it.

## Objectives

| scope | problem | objective | stop condition |
|---|---|---|---|
| A1 rail-wise | each rail file | raw peak \|Z11\| (ohm) | peak <= rail target (tightest level) |
| A2 whole package | `y_sp_mphy_full` (5 obs ports) | max_k \|Z_kk\| / Ztarget_k | every rail meets its target |
| A3 improvable only | mer2 + pll1v0 + mphyvdd sub-network | raw peak, max over its 3 obs ports (ohm) | every rail meets its target |
| A4 / A5 DDR3 | `ddr21` / `ddr_full` | raw peak \|Z11\| (ohm) | peak <= target |

A3 has three observation ports, so "raw peak" is read as the largest raw peak among them, with the
per-rail targets only deciding when to stop. Bands: B1 (<=20 MHz), B2 (<=50 MHz), B3 (full). Exp 1's B4
(full band, frequency-weighted) is dropped - it scored 1.04-1.11 against B2, which won - leaving 5 x 3 = 15
cells; the weighting code is dormant and `BANDS` in `study_config.py` brings it back. Every winning placement
is re-scored on all bands.

## Noise flags

A level whose target (floor x multiplier) lies **below the median** achieved peak of the Exp 1 runs is *inside
the floor spread*: a typical run could not reach it. `derive_targets.py` writes these flags to `targets.json`
(`flags[key][level]`); every run record carries them (`levels[lvl].inside_noise`, `inside_noise_rails`; a level of a
multi-rail problem is flagged if any of its rails is), and the report marks flagged success rates with a dagger.
A low success rate at a flagged level means "target inside measurement noise", not a difference between
scopes or bands. Grid 2 stops its PSO runs at the tightest level that is *not* flagged.

## Grid 2

* **Primary metric: wall-time for a fixed evaluation count** - `FIXED_EVALS` identical (model, pad)
  configurations (1..20 capacitors) per size, evaluated by every method in per-frequency loop mode; the cost
  agreement with numpy is recorded alongside. `pure_python` is measured at N=16 and 21 only and extrapolated
  by the measured n^3 scaling to N=73 and 79.
* **`iterative` is an approximation** (second-order Neumann series, full-inverse fallback where it diverges),
  not a method under test: it is timed per evaluation only, with its series fallback rate beside it, and is
  excluded from placement-match claims and every achieved-impedance comparison. It gets no PSO runs.
* **Secondary: PSO runs of the exact methods** (numpy, solve, sm, pure_python at N=16/21) for time-to-target
  (converged runs only) and placement match against numpy (runs not truncated by the unchanged 1800 s cap).
  If the first numpy run at a size hits the cap without reaching the stop level the rest of that size is skipped
  (recorded in a `grid2_skip` record).
* N=16 uses `mer1`: with the Exp 1 floors `mer2`'s bare network already meets every level, so its runs would be empty.

## Port rules

Observation ports come first in every dataset and never take a capacitor: one in every dataset except
`y_sp_mphy_full.mat`, which has five (indices 0-4). Decap pads are every remaining index, and a run's capacitor
limit is the number of pads `N - n_obs`, derived from the loaded matrix (`n_obs` is declared in
`study_config.N_OBS_BY_FILE`; nothing else is hardcoded). `--validate` checks this on every dataset.
Sub-networks (A3) are built through Z (`inv(y)[idx][:, idx]` then invert), never by slicing Y.

## Records

One JSON object per finished run is appended to `results/results_long.jsonl` (all factor levels as
fields). Besides Exp 1's fields each record carries:

* `placement`: `[{"cap", "model", "pad", "pad_global"}, ...]` - `model` is the 0-based row of
  `decaps_sp.mat`, `pad` the 0-based index in that problem's matrix, `pad_global` the index in the parent
  PDN (`y_sp_mphy_full` / the DDR matrix); `problem` names the matrix. Enough to re-score offline.
* `levels`: per threshold level `hit`, `caps_needed`, `iter`, `n_evals`, `time_s`, `target_ohm`, and the
  `placement` of the first placement that met it.
* `objective` (`raw` / `ratio`), `cost_unit`, `best_cost_own_band`, `curve_cost`, `histories`,
  `rescored[B1..B3] = {peaks_ohm, ratio, met_levels}`.

## Files

| file | role |
|---|---|
| `study_config.py` | paths, bands, port-rule declarations, PSO / staging / Grid 2 settings |
| `study_core.py` | data, targets loading (refuses nulls), fitness, PSO |
| `experiments.py` | stage runners, Grid 2, consistency check, JSONL logging |
| `analyze.py` | screening, matrices (per-level success), plots, report |
| `derive_targets.py` | Exp 1 results -> `targets.json` |
| `validate.py` | the `--validate` dry check |
| `run_all.py`, `run_exp2.sh` | sequencer and nohup launcher |
