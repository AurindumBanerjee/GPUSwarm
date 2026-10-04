VDDQ_DDR3: To1GHz understanding plots
grid 2588 pts 110.9-9.972e+08 Hz (the 100 MHz sets stop at 9.966e+07 Hz)
library decaps_combined_Y_loginterp_loglogextrap.mat, 6695 models, MEASURED below 97.9 MHz and EXTRAPOLATED above it
best-fill scan uses every 40th model (168 of 6695)

VDD_DDR_1V35   N= 21  bare peak     142.1 ohm @    939.2 MHz | filled     3.406 ohm ( 97.6% gain) | 100MHz-set peak 0.5017 ohm | coupling 0.028 @10MHz -> 0.191 @1GHz

Plots:
  z11_<rail>_1ghz.png     bare vs every pad filled; dashed = the 100 MHz dataset;
                          shaded = extrapolated decap data; red dotted = old 99.66 MHz ceiling
  coupling_<rail>_1ghz.png  median normalised obs<->pad transfer impedance vs frequency.
                          This is decap authority: where it collapses, capacitors cannot help.
  library_measured_vs_extrapolated.png  six capacitors across the range, with the extrapolated part shaded

Caution: every number above 97.9 MHz rests on extrapolated capacitor models
(constant-ESL inductive tail), not vendor measurements. The PDN data itself is measured to 5 GHz.
