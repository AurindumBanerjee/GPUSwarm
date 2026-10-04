# 1 GHz datasets

Same port maps, same open-circuit reduction and the same filenames as the
`SP_Sep2026/` sets one level up. The only difference is the frequency clip:
**110.9 Hz - 997.2 MHz, 2588 points** instead of stopping at 99.66 MHz.

Nothing here is extrapolated. The SP_Data Touchstone files are characterised to
5 GHz; the 100 MHz ceiling on the other sets came from the AVX capacitor library,
not from the board data. These files simply keep measured points that were
previously discarded.

Pair them with a decap library covering the same span (`Decaps/Combined/*`),
whose values above ~98 MHz ARE extrapolated - see that folder's notes.

Built by `Scripts/build_sp_1ghz.py`.

## Bare peak |Z11|, 100 MHz set vs this one

| rail | peak to 99.66 MHz | peak to 997 MHz | where the 1 GHz peak sits |
|---|---|---|---|
| mer1 | 3.24 ohm | 30.7 ohm | 538 MHz |
| mer2 | 1.16 ohm | 15.5 ohm | 332 MHz |
| pll1v0 | 3.52 ohm | 39.9 ohm | 218 MHz |
| pll1v8 | 24.3 ohm | 241.7 ohm | 997 MHz (band edge) |
| mphyvdd | 3.15 ohm | 33.7 ohm | 197 MHz |
| ddr21 / ddr_full | 0.50 ohm | 142.1 ohm | 939 MHz |

The minima are unchanged, since they sit far below 100 MHz.

## Read this before optimising over the full span

The peaks above 100 MHz are an order of magnitude larger than anything below it,
and they are package and board cavity resonances. Discrete decoupling capacitors
cannot damp them: the capacitor is separated from the observation port by mount
and interconnect inductance, which dominates long before these frequencies. On
silicon, that region is handled by on-die capacitance instead.

So a PSO run scored on the full 1 GHz span will be dominated by a peak it cannot
move, exactly as the 100 MHz sets were dominated by the 99.66 MHz band edge, only
worse. Use these files to show where decap authority ends, or to score a band
that stops below the first cavity resonance. Do not simply widen the objective
band and expect the existing targets to mean anything.
