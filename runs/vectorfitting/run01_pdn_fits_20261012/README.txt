RUN vectorfitting/run01_pdn_fits_20261012  [finished, retrofitted]
Vector fits of the PDN Z11 (ddr21, ddr_full, mer1, mer2, pll1v0, pll1v8, mphyvdd, mphy_full) over 110 Hz-100 MHz at order 24, reruns at order 20 (ddr21 reference) and 32 (ddr21, mer1, mer2, pll1v0, mphyvdd), and the SRF bucketing of all 6695 decap models.

Start 2026-10-12 19:25:15 IST  End 2026-10-12 19:27 IST  Host eeiitj123.iitj.ac.in
Environment: conda env work, numpy 2.2.5, scipy 1.15.3, matplotlib 3.10.7, BLAS threads 1; no GPU
Code: the committed version f1495e3 (VectorFitting/), copied unchanged into code/; the two shell scripts are the exact commands that were run.
Layout: code/ (copies), output/ (json fits per order + decap_classes.json; git-ignored), logs/ (console logs; git-ignored),
        summary/ (fit_quality.csv, resonances.csv, sampled_maxima.csv, decap_buckets.txt; tracked), plots/ (poles_*.png per order; tracked).
Replay: bash run.sh  (writes to output/rerun_<timestamp>/, never overwrites).
output/laptop/: the two fits first made on the laptop before the server run (ddr21 order 20: reproduces the server's order-20 fit to 3e-9 relative in f0 and 6e-14 in RMS; mer1 order 20: 3.23 % RMS, not run on the server). The server replay (run.sh) reproduces all 14 fits and the bucket table.

KEY RESULTS
* Fit RMS at order 24: ddr21/ddr_full 0.35 %, pll1v8 0.58 %, pll1v0 2.96 %, mer1/mphy_full 3.19 %, mphyvdd 3.25 %, mer2 5.47 % (warning). Order 32 does not
  improve any of them (mer2 5.45 %, mer1 3.05 %). ddr_full duplicates ddr21 and mphy_full duplicates mer1 (same observation port).
* ddr21 at order 20 reproduces the reference poles exactly (23.8 MHz Q=647, 47.1 MHz Q=12.9, 10.5 MHz Q=24.5, 1.12 MHz Q=114, 2.55 MHz Q=8.6; 0.347 %).
  At orders 24 and 32 the RMS is the same but the pole list changes materially: the individual poles/Q of ddr21 are not determined by the data.
* In 110 Hz-100 MHz the sampled |Z11| of ddr21, ddr_full and pll1v8 has no resonance peak (smooth inductive ramp); their poles are sub-grid needles.
* All MPHY rails share two real bumps at 27.2 and 31.07 MHz (about 1 Ohm); the 5 poles in 26-31 MHz of mer1 are near-duplicate pairs describing these two bumps.
* The dominant bare peaks lie outside the fitted band: mer1 539 MHz 30.7 Ohm, mer2 332 MHz 15.5 Ohm, pll1v0 218 MHz 40 Ohm, mphyvdd 197 MHz 33.7 Ohm, ddr 939 MHz 142 Ohm.
* Decap SRF buckets (6695 parts): 10-100 kHz 3, 100 kHz-1 MHz 1431, 1-10 MHz 3602, 10-100 MHz 1615 (median ESR 535 mOhm), 100 MHz-1 GHz 44 (SRF at the band edge).

---- timeline ----
