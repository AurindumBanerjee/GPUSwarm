# VectorFitting

Rational-function (pole-residue) modelling of PDN impedance and decap
impedance, as the basis for a frequency-matched decap selection strategy.

```
vectorfit.py        Gustavsen-Semlyen vector fitting, numpy only
decap_classify.py   characterise all 6695 parts, bucket them by SRF
pdn_poles.py        vector-fit a PDN's Z11, extract its resonances
```

## What this is, and what it is not

The strategy — bucket decaps by self-resonant frequency, bucket the PDN's
impedance peaks by frequency, and draw from the matching bucket — is **SRF
matching**, not vector fitting. It is a sound heuristic and it stands on its own.

Vector fitting is the tool that makes the "where are the peaks" half rigorous.
Reading peaks off a sampled `|Z11|` curve gives frequency and height. Fitting
`Z11(s) ≈ Σ rₖ/(s−pₖ) + d + s·e` gives, for each resonance:

| from the pole | what it tells you |
|---|---|
| `f₀ = Im(p)/2π` | the frequency, not quantised to the frequency grid |
| `Q = \|p\| / 2·Re(p)\|` | how sharp — whether one capacitor covers it or several must straddle it |
| `\|residue\| / σ` | how much of the peak height this pole owns, so overlapping resonances separate instead of reading as one hump |

Q and the residue are what decide how much capacitance a peak needs and over how
wide a band. You cannot get either by peak-picking.

## Direction matters: fit Z for the PDN, Y for a part

A PDN seen from one port is a lossy LC network, so `Z11(s)` has genuine complex
pole pairs at its **anti-resonances** — the peaks to damp.

A capacitor is the opposite. `Z(s) = sL + R + 1/(sC)` has one pole at the origin
and its self-resonance is a **zero**; the complex pole pair lives in
`Y = 1/Z`. Fitting a part's impedance returns only real poles and finds nothing.
This caught me out once and is now commented at both call sites.

## Status

**`vectorfit.py` — validated.** On a synthetic 3-resonance function with known
poles it recovers every `f₀` and `Q` exactly, plus `d` and `e`, at relative RMS
4e-15 in 3 iterations.

One thing that is not optional: the design matrix mixes columns differing by
~15 orders of magnitude (`1/(s−p)` against `s`), so the columns are normalised
before the least-squares solve. Without it the poles never move from their
starting positions and the fit silently returns the initial guess — which is
exactly what the first version did.

**`pdn_poles.py` — works well.** On `ddr21` over 110 Hz–100 MHz, 20 pole pairs
fit to 0.35% relative RMS and the poles land within **0.2%** of the sampled
local maxima. Seven resonances above 1% of the largest:

| f₀ | Q | bandwidth | decap bucket |
|---|---|---|---|
| 23.8 MHz | 647 | 37 kHz | 10–100 MHz |
| 47.1 MHz | 12.9 | 3.6 MHz | 10–100 MHz |
| 10.5 MHz | 24.5 | 429 kHz | 10–100 MHz |
| 1.12 MHz | 114 | 9.7 kHz | 1–10 MHz |
| 2.55 MHz | 8.6 | 296 kHz | 1–10 MHz |
| 1.70 MHz | 39 | 43 kHz | 1–10 MHz |
| 21.0 kHz | 0.57 | 37 kHz | 10–100 kHz |

Note the spread in Q: 647 at 23.8 MHz is a needle 37 kHz wide, while 12.9 at
47 MHz is a 3.6 MHz hump. Those need different parts, and nothing in the sampled
curve says so.

Fit the band you care about. Over the full span to 1 GHz the 142 Ω cavity
resonance at 939 MHz dwarfs everything and the fit spends its order there.

**`decap_classify.py` — works, but not via vector fitting.** Analytic extraction
from the asymptotes (C from the capacitive slope, ESL from the inductive slope,
ESR at the minimum, SRF by parabolic refinement) gives clean buckets:

| SRF bucket | parts | median C | median ESR | median ESL |
|---|---|---|---|---|
| 10–100 kHz | 3 | 240.6 µF | 131.6 mΩ | 16.7 nH |
| 100 kHz–1 MHz | 1431 | 100.2 µF | 40.4 mΩ | 2.37 nH |
| 1–10 MHz | 3602 | 6.40 µF | 118.8 mΩ | 2.16 nH |
| 10–100 MHz | 1615 | 0.43 µF | 535.4 mΩ | 1.65 nH |
| 100 MHz–1 GHz | 44 | 0.02 µF | 1393 mΩ | — |

C falls monotonically as SRF rises, which is the physics check passing.

**Vector fitting the parts does not work at low order, and that is a result.**
A real decap over three decades is not a 2-pole system — distributed dielectric
loss and secondary resonances mean a 1–2 pair fit lands at 2–25% relative error
and locates SRF to only ~40%. The asymptote extraction is both faster and more
accurate here. Use VF for the PDN, asymptotes for the parts.

The last bucket's 44 parts have their SRF at the band edge, so their ESL has no
inductive decade to fit — these are the same 44 whose extrapolated tails were
flagged as unphysical in `Data/Decaps/README_library.md`.

## Not built yet

The matcher. With poles on one side and buckets on the other, the next step is
to assign parts to peaks — nearest SRF, enough capacitance for the residue,
enough parts to cover the bandwidth implied by Q — and emit a candidate
placement.

The reason that is interesting here rather than just tidy: the warm-start study
showed PSO on this problem is effectively **greedy forward selection** that
plateaus around 8 capacitors and cannot revisit earlier choices
(`Comparisons/`). A placement seeded from the pole structure starts somewhere
physically sensible instead of at random, which is precisely the weakness greedy
has. Whether it actually beats a random start is an experiment, not a claim.

## Running

```
python decap_classify.py                       # buckets + decap_classes.json
python decap_classify.py --verify 25           # VF cross-check (expect it to disagree)
python pdn_poles.py --prob ddr21 --hi 1e8      # fit the decap-relevant band
python pdn_poles.py --board MPHY --prob mer1 --order 24 --plot
```

`PDN_DATA` overrides the data root (default `../Data`).
