# MPHY package

A single chip package carrying **five electrically independent supply rails**. Each rail
is its own PSO problem: a capacitor on one rail cannot affect another. Measured
cross-rail transfer impedance is at least 25 dB below the self impedance.

Two extractions of the same package are here. **Use the SP_Data one.**

---

## Current extraction — `SP_Data_MPHY_68Caps.s73p`

647 MB Touchstone, created 4 Sep 2026, `# hz S ma R 50` (magnitude/angle, 50 ohm
reference). 73 ports, 2949 frequency points, 1 Hz to 5 GHz. No port-name comments in
the file — the port map below was recovered by clustering ports on normalized coupling
and confirmed by the pad counts matching the legacy extraction exactly.

### Port map

| ports | role |
|---|---|
| 1-5 | observation, one per rail (chip-side terminal) |
| 6-20 | 15 pads, VDDE1V0_MER1 |
| 21-35 | 15 pads, VDDE1V0_MER2 |
| 36-49 | 14 pads, VDD1V0_PLL |
| 50-64 | 15 pads, VDDA1V8_PLL |
| 65-73 | 9 pads, MPHY_VDD |

68 pads total, matching the filename. Observation port `k` owns pad block `k`; each
couples 30x to 2000x more strongly to its own block than to any other.

### Derived files

All carry variable `y`, shape `(N, N, F)` complex128, **admittance in siemens**, with
**port 1 = observation** and ports 2..N = that rail's pads. Grid is `freq_sp.mat`:
2089 points, 100.9 Hz - 99.66 MHz (clipped to where the decap library is valid).

| file | rail | N | pads | size | min abs(Z11) | worst abs(Z11) | all pads filled | gain |
|---|---|---|---|---|---|---|---|---|
| `y_sp_mer1.mat` | VDDE1V0_MER1 | 16 | 15 | 8.6 MB | 3.17 mohm | 3.241 ohm | 3.198 ohm | 1% |
| `y_sp_mer2.mat` | VDDE1V0_MER2 | 16 | 15 | 8.6 MB | 3.56 mohm | 1.160 ohm | 0.862 ohm | 26% |
| `y_sp_pll1v0.mat` | VDD1V0_PLL | 15 | 14 | 7.5 MB | 8.34 mohm | 3.516 ohm | 1.862 ohm | 47% |
| `y_sp_pll1v8.mat` | VDDA1V8_PLL | 16 | 15 | 8.6 MB | 14.6 mohm | 24.34 ohm | 23.18 ohm | 5% |
| `y_sp_mphyvdd.mat` | MPHY_VDD | 10 | 9 | 3.3 MB | 9.26 mohm | 3.152 ohm | 2.340 ohm | 26% |
| `y_sp_mphy_full.mat` | all five | 73 | 68 | 178 MB | - | - | - | - |

`y_sp_mphy_full.mat` keeps the native 73-port ordering, so its port 1 is the MER1
observation port and the other four rails are present but unused by a PSO that reads
only `Z[0,0]`. Use it to verify that rails optimized separately still hold when all
five sets of capacitors are installed together.

Worst-case impedance is at **99.66 MHz on every rail** — the top edge of the band.

"All pads filled" = the single best capacitor model from the AVX library placed on every
pad. It is an upper bound on what placement optimization can reach, not an optimized
result.

### Known limitation of this extraction

On MER1 and PLL1V8, filling every pad improves the worst case by 1-5%. The observation
ports appear to sit at the chip terminal, with package interconnect inductance between
the chip and every pad; above a few MHz that inductance dominates and the capacitors
cannot be seen from the observation port. PSO has very little leverage on those two
rails. MER2, PLL1V0 and MPHY_VDD respond normally (26-47%). Worth raising with
Prof. Tripathi before using MER1 or PLL1V8 as benchmark cases.

---

## Legacy extraction — `MPHY_C28.s78p`

198 MB, Aug 2025, `# Hz S RI R 50`. 78 ports with full port-name comments:
1-5 die ports, 6-10 observation (`ckt_*`), 11-78 the same 68 pads.
`MPHY_C28.s78p.txt` is a byte-identical copy.

| file | contents | grid |
|---|---|---|
| `y78.mat` | all 78 ports, S converted to Y | `freq78.mat` (925 pts, 0.1 Hz - 5 GHz) |
| `freq78.mat` | that grid | |
| `y_mer1/mer2/pll1v0/pll1v8/mphyvdd.mat` | rail slices, obs = the `ckt_*` port | `Decaps/freq_band.mat` (99 pts, 133 Hz - 98 MHz) |

Bare worst-case runs to 3.5e12 ohm (MER1, at 7.5 MHz) because this extraction is passive
only — no VRM or DC path — so the undamped rail is numerically an open circuit at
anti-resonance. MATLAB will warn `RCOND ~ 1e-16` on those frequencies. The values are
real but should not be quoted to four digits; call it "effectively an open circuit".

---

## Loading

```python
import numpy as np, scipy.io as sio
y = np.transpose(sio.loadmat('y_sp_mer1.mat')['y'], (2,0,1))   # -> (F, N, N)
f = sio.loadmat('freq_sp.mat')['freq'].ravel()
d = sio.loadmat('../Decaps/decaps_sp.mat')['decaps']           # (3347, 2089)
Z11 = np.linalg.inv(y)[:, 0, 0]
```

To add a capacitor: `y[:, port, port] += d[model]`, with `port` in `1..N-1`.
Never slice `y` to build a smaller network — slicing Y shorts the ports you drop.
Go through Z: `Zs = inv(y)[:, idx, idx]; y_sub = inv(Zs)`.
