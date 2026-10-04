# VDDQ DDR3 board

A DDR3 memory board with a **single supply rail**, `VDD_DDR_1V35`. One PSO problem, not
five. This is the healthier of the two boards: capacitors visibly change the impedance,
so placement optimization has real leverage here.

Two extractions are present. **Use the SP_Data one.**

---

## Current extraction — `SP_Data_VDDQ_DDR3_78Caps.s79p`

739 MB Touchstone, created 4 Sep 2026, `# hz S ma R 50` (magnitude/angle, 50 ohm
reference). 79 ports, 2949 frequency points, 1 Hz to 5 GHz. No port-name comments — the
map below follows from the filename (78 caps + 1) and was confirmed by coupling
analysis: one connected group, with port 1 the odd port out.

### Port map

| ports | role |
|---|---|
| 1 | observation |
| 2-79 | 78 decap pads |

### Derived files

Variable `y`, shape `(N, N, F)` complex128, **admittance in siemens**, **port 1 =
observation**. Grid is `freq_sp.mat`: 2089 points, 100.9 Hz - 99.66 MHz.

| file | N | pads | size | min abs(Z11) | worst abs(Z11) | pads filled | gain |
|---|---|---|---|---|---|---|---|
| `y_sp_ddr_full.mat` | 79 | 78 | 209 MB | 0.567 mohm | 0.5017 ohm | - | - |
| `y_sp_ddr21.mat` | 21 | 20 | 14.7 MB | 0.567 mohm | 0.5017 ohm | 0.231 ohm | 54% |

Both report the same bare `Z11`, and that is the correctness check passing: dropping
ports that are left **open** cannot change the impedance at the observation port. If
those two numbers ever differ, the sub-network reduction is wrong.

`y_sp_ddr21.mat` keeps the observation port plus the first 20 pads. It exists for the
`pure_python` baseline in PythonBench: Gauss-Jordan is O(n^3), so one fitness evaluation
costs roughly **6.0 s at 79 ports versus 0.11 s at 21**. At paper scale (20 runs x
~15k evaluations) the 79-port version is months of compute for that method alone. Use
the full network for `numpy`/`solve`/`sm`/`iterative` (about 20 ms per evaluation) and
`y_sp_ddr21` for `pure_python`. It is also the same size as the original `y2.mat`
benchmark, so speedup figures stay comparable to results you already have.

Worst case is at **99.66 MHz**, the top edge of the band.

### Impedance profile, 20 pads filled with the best single model

| band | worst abs(Z11) |
|---|---|
| below 100 kHz | 0.82 mohm |
| 100 kHz - 1 MHz | 5.2 mohm |
| 1 - 10 MHz | 38.5 mohm |
| 10 - 50 MHz | 0.164 ohm |
| 50 - 100 MHz | 0.326 ohm |

Almost all the residual is in the top octave. Restricting the optimization band to
50 MHz and below would change what PSO converges on by roughly 2x.

---

## Legacy extraction — `PCB_010316_VDDQ_A5_DDR3_trial_flyby_B_pads9-5.S81P`

220 MB, Aug 2025, `# Hz S RI R 0.1`. Note the **0.1 ohm reference impedance** — not 50.
Any S-to-Y conversion that assumes 50 produces silent garbage for this file.
81 ports with port-name comments: 1-2 on an unrelated net (`N43826182`), 3-5 the die
ports (`U3003`, `U2000`, `U2001`), 6-81 the cap pads.

| file | contents | grid |
|---|---|---|
| `y81.mat` | all 81 ports | `freq81.mat` (925 pts, 0.1 Hz - 5 GHz) |
| `freq81.mat` | that grid | |
| `y_ddr.mat` | 79 ports: obs `U3003` + 78 pads | `Decaps/freq_band.mat` (99 pts) |
| `y_ddr21.mat` | 21 ports: obs + first 20 pads | same |

Bare worst case 2565 ohm at 4.2 kHz. This extraction is passive only — no VRM — so the
low-frequency end has nothing holding it down, unlike the SP_Data version which reaches
0.57 mohm. Best achievable with 20 capacitors on `y_ddr21` was measured at **0.345 ohm**
by a short PSO run; the 0.05 ohm threshold in PythonBench is not reachable on it.

---

## Loading

```python
import numpy as np, scipy.io as sio
y = np.transpose(sio.loadmat('y_sp_ddr21.mat')['y'], (2,0,1))  # -> (F, N, N)
f = sio.loadmat('freq_sp.mat')['freq'].ravel()
d = sio.loadmat('../Decaps/decaps_sp.mat')['decaps']           # (3347, 2089)
Z11 = np.linalg.inv(y)[:, 0, 0]
```

To add a capacitor: `y[:, port, port] += d[model]`, with `port` in `1..N-1`.
Never slice `y` directly to build a smaller network — that shorts every dropped port.
Go through Z: `Zs = inv(y)[:, idx, idx]; y_sub = inv(Zs)`.
