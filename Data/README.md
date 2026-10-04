# Data

Inputs for the PSO decap-placement runs. Everything here is on **one shared
frequency grid**: 2588 points, 110.89 Hz - 997.21 MHz (`Decaps/sp_freq.mat`).
`freq_sp.mat` in each board folder is byte-identical to it, so no resampling is
needed anywhere in the PSO code.

The `.mat` files are gitignored - they are ~650 MB and rebuildable from
`Dataset/Scripts/`. The READMEs are tracked.

## Decaps/
| file | contents |
|---|---|
| `combined_decaps.mat` | `decaps` (6695, 2588) complex admittance in siemens; plus `names`, `source`, `srf`, `loglog_slope`, `extrap_start_hz` |
| `sp_freq.mat` | `freq` (1, 2588), the shared grid |

6695 shunt-mounted models: 3348 legacy + 3347 AVX. No series entries, so no
filtering is needed - use all of them. Measured up to ~99.8 MHz; above that the
inductive tail is extrapolated, which is **19.3% of the grid points**.

## MPHY/  - five independent rails, 68 pads total
| file | rail | N | pads |
|---|---|---|---|
| `y_sp_mer1.mat` | VDDE1V0_MER1 | 16 | 15 |
| `y_sp_mer2.mat` | VDDE1V0_MER2 | 16 | 15 |
| `y_sp_pll1v0.mat` | VDD1V0_PLL | 15 | 14 |
| `y_sp_pll1v8.mat` | VDDA1V8_PLL | 16 | 15 |
| `y_sp_mphyvdd.mat` | MPHY_VDD | 10 | 9 |
| `y_sp_mphy_full.mat` | all five | 73 | 68 |

## VDDQ_DDR3/  - one rail
| file | N | pads |
|---|---|---|
| `y_sp_ddr21.mat` | 21 | 20 |
| `y_sp_ddr_full.mat` | 79 | 78 |

## Conventions

`y` has shape `(N, N, F)` complex128 - multiport **admittance**, siemens.
Observation ports come **first** and never take a capacitor: one in every file
except `y_sp_mphy_full.mat`, which has five (indices 0-4). Decap pads are every
remaining index, so the capacitor limit is `N - n_obs`, read from the matrix.

Install a capacitor with `y[:, pad, pad] += decaps[model]`. Never slice `y` to
build a smaller network - that shorts the ports you drop. Go through Z:
`Zs = inv(y)[:, idx, idx]; y_sub = inv(Zs)`.
