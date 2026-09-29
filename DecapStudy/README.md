# Decap-placement study (CPU version)

Structured PSO study on the MPHY package (5 rails) and the DDR3 board. CPU first;
the GPU port reuses the same problem definitions and records.

## Run (on the server)

```bash
export PDN_DATA=/DATA/Aurindum/Swarming/Dataset   # MPHY/ VDDQ_DDR3/ Decaps/
export STUDY_THREADS=16                            # parallel particle evaluation, same results
nohup python run_all.py > run_all.out 2>&1 &
```

`run_all.py` runs everything sequentially and unattended, and is resumable
(finished runs are read back from `results/results_long.jsonl` and skipped):

1. rail responsiveness (bare vs every pad filled, per band)
2. Stage 1 - 20 cells x 3 runs, 20 particles x 6 iterations; screening table written to
   `results/stage1_table.md` and the log
3. Stage 2 - survivors x 10 runs, 50 particles x 15 iterations (starts by itself; `--pause-after-stage1` to stop)
4. Grid 2 - 5 methods x N = 16/21/73/79 at the winning band (pure_python at N=16,21 only, 2 runs; larger N extrapolated by n^3)
5. consistency check (A1 rails installed together in `y_sp_mphy_full`)
6. plots + `results/report.md`

`python run_all.py --smoke` checks the pipeline with tiny budgets.
`python analyze.py` rebuilds tables, plots and the report from the JSONL at any time.

## Design (what is decided where)

| topic | where |
|---|---|
| paths, Ztarget assumptions, bands, budgets, Grid 2 settings | `study_config.py` |
| data, fitness (5 inversion methods), PSO | `study_core.py` |
| Stage runners, Grid 2, consistency, JSONL logging | `experiments.py` |
| screening, matrices, plots, report | `analyze.py` |

* Cost = max over observation ports k of `peak_f w_k(f) |Z_kk| / Ztarget_k`; target is 1.0. `w = 1`
  on B1-B3; on **B4** it is the per-frequency "how much can the best single capacitor lower |Z_kk|"
  weight (exact rank-1 update over every model x pad, normalised to max 1).
* Every winning placement is re-scored on **all four** bands. Cells are ranked on `CARE_BAND`
  (default B2; env `CARE_BAND` or `--care-band`), never on their own objective.
* Observation ports (index 0; indices 0-4 on `y_sp_mphy_full`) are never capacitor sites; pads are
  the remaining indices, N comes from the loaded matrix. `MAX_CAPS` = number of pads (68 for A2, 38 for A3,
  78 for A5, 9-15 per MPHY rail), capped at 20 only for the 21-port DDR3 PDN.
* Sub-networks (A3) are built through Z (`inv(y)[idx][:, idx]` then invert), never by slicing Y.
* Paired seeds: `BASE_SEED + run_id*10000 + n_caps`, identical across cells and methods.
* PSO follows `CPUTest/SB.py` (FIX 1-13): 40 % hybrid warm start, duplicate-free port resolution,
  immediate stop at the target.

## Assumptions to check (all in `study_config.py`)

* Ztarget = V x 5 % / dI. V: MER1/MER2/PLL1V0/MPHY_VDD = 1.0 V (MPHY_VDD assumed), PLL1V8 = 1.8 V,
  DDR3 = 1.35 V. dI: 0.25 A (MER1, MER2), 0.10 A (PLL1V0, MPHY_VDD), 0.05 A (PLL1V8), 1 A (DDR3).
  The MPHY current steps are small on purpose: with tens-of-mA-per-rail targets of 0.2-1.8 ohm the
  library can actually reach them on the responsive rails; a milliohm target is unreachable on every
  MPHY rail (see the rail-responsiveness table).
* `CARE_BAND = B2`: the band on which cells are compared. Change it if you care about another band.
* A2 budget: shared across all 68 pads (MAX_CAPS = 68).
* A3 budget: the network contains only the 38 pads of mer2/pll1v0/mphyvdd, so mer1's and pll1v8's
  share is redistributed by construction (MAX_CAPS = 38).
* Cost warning: runs whose target is unreachable go through every capacitor count (up to 68-78 on
  A2/A5), which is very slow on CPU at N=73/79 (~1 s per evaluation). Use `STUDY_THREADS` and/or
  `--patience K` (stop after K capacitor counts without a >0.1 % gain).
