# Capacitor libraries and frequency grids

Shared by both boards. A "decap model" is one part number's measured admittance vs
frequency; the PSO picks a model and a pad for each capacitor it places.

Admittance `Y` is the reciprocal of impedance: `Y = 1/Z`. The datasets are in
admittance because installing a capacitor is then a single addition on the diagonal
(`y[:, port, port] += d[model]`) instead of a network re-solve.

| file | variable | shape | what it is |
|---|---|---|---|
| `decaps_sp.mat` | `decaps` | (3347, 2089) complex | **use this.** AVX shunt models resampled onto the SP grid. Also carries `names`, the part numbers. |
| `freq_sp.mat` | `freq` | (1, 2089) | the SP grid, 100.9 Hz - 99.66 MHz. Identical for both boards. |
| `decaps_avx.mat` | `decaps` | (6695, 301) complex | source library, 100 Hz - 100 MHz. Also `names` and `fixture`. |
| `decaps.mat` | `decaps` | (3348, 1391) complex | legacy library, no part names |
| `decaps_band.mat` | `decaps` | (3348, 99) complex | legacy library clipped to the legacy 99-point band |
| `freq_band.mat` | `freq` | (1, 99) | legacy band, 133 Hz - 98 MHz |
| `Decaps_AVX/`, `Decaps_AVX.zip` | | | vendor source files the AVX library was built from |

## The series/shunt split

`decaps_avx.mat` holds each of 3348 part numbers twice, tagged by `fixture`:
**3348 `series`** and **3347 `shunt`** entries. A decoupling capacitor sits between the
power rail and ground, which is the **shunt** measurement — that is what
`decaps_sp.mat` contains. Using the series entries would model a capacitor inserted
into the rail itself, which is not what is being placed.

## Capacitance range

Estimated from `Y` at low frequency (`C = Im(Y) / omega`):

| | capacitance | abs(Z) at 1 kHz | abs(Z) at 1 MHz | abs(Z) at 99 MHz |
|---|---|---|---|---|
| small (2nd percentile) | 0.22 uF | 727 ohm | 3.72 ohm | 1.05 ohm |
| median | 22 uF | 7.33 ohm | 0.551 ohm | 0.773 ohm |
| large (97th percentile) | 678 uF | 0.239 ohm | 17.1 mohm | 1.22 ohm |

Every model slopes down with frequency, bottoms out at its self-resonance, then rises as
its own internal inductance takes over. Large capacitors win at low frequency and give
up earlier; small ones are the opposite. Mixing sizes is what the optimizer is for.

## The band-edge problem

The library stops at **100 MHz**, and on every dataset the worst-case impedance sits at
99.66 MHz — the last point in the band. The optimizer will therefore spend its whole
budget on the frequency where it has the least influence. Either restrict the
optimization band to roughly 50 MHz and below, or obtain capacitor data that extends
higher. This is the single most consequential choice in the setup.

## Resampling

`decaps_sp.mat` was built by interpolating real and imaginary parts separately in
log-frequency from the 301-point source onto the 2089-point PDN grid. That direction is
safe because capacitor admittance is smooth. Interpolating the PDN the other way would
fabricate or erase anti-resonance peaks.
