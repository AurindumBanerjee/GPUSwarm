# 04 Rational macromodel + grid-free peak

Z0(s) is fitted once with common-pole vector fitting (`--vf-pairs`,
`--vf-iter`); every shunt decap is fitted as series R-L-C. The fitness
searches a coarse grid (`--ngrid`) and zooms on the top `--kpeaks` local
maxima, so peaks between samples are found. `--fmax-ext` searches above the
data band (extrapolation -- flagged in the log).

Needs `--freq`. Fit quality goes to `fit_info.json`; every run's final
placement is re-scored on the table (`best_minz_table`). Check the
normalised VF error before trusting results; raise `--vf-pairs` if needed.

Methods: numpy, fast, macro.
