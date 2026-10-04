MPHY: To1GHz understanding plots
grid 2588 pts 110.9-9.972e+08 Hz (the 100 MHz sets stop at 9.966e+07 Hz)
library decaps_combined_Y_loginterp_loglogextrap.mat, 6695 models, MEASURED below 97.9 MHz and EXTRAPOLATED above it
best-fill scan uses every 40th model (168 of 6695)

VDDE1V0_MER1   N= 16  bare peak     30.69 ohm @    537.9 MHz | filled     29.57 ohm (  3.6% gain) | 100MHz-set peak 3.241 ohm | coupling 0.015 @10MHz -> 0.008 @1GHz
VDDE1V0_MER2   N= 16  bare peak     15.48 ohm @    331.6 MHz | filled      8.12 ohm ( 47.6% gain) | 100MHz-set peak 1.16 ohm | coupling 0.082 @10MHz -> 0.147 @1GHz
VDD1V0_PLL     N= 15  bare peak     39.95 ohm @    218.1 MHz | filled     18.29 ohm ( 54.2% gain) | 100MHz-set peak 3.516 ohm | coupling 0.048 @10MHz -> 0.017 @1GHz
VDDA1V8_PLL    N= 16  bare peak     241.7 ohm @    997.2 MHz | filled     241.3 ohm (  0.2% gain) | 100MHz-set peak 24.34 ohm | coupling 0.024 @10MHz -> 0.025 @1GHz
MPHY_VDD       N= 10  bare peak      33.7 ohm @    197.1 MHz | filled     16.55 ohm ( 50.9% gain) | 100MHz-set peak 3.152 ohm | coupling 0.174 @10MHz -> 0.013 @1GHz

Plots:
  z11_<rail>_1ghz.png     bare vs every pad filled; dashed = the 100 MHz dataset;
                          shaded = extrapolated decap data; red dotted = old 99.66 MHz ceiling
  coupling_<rail>_1ghz.png  median normalised obs<->pad transfer impedance vs frequency.
                          This is decap authority: where it collapses, capacitors cannot help.
  library_measured_vs_extrapolated.png  six capacitors across the range, with the extrapolated part shaded

Caution: every number above 97.9 MHz rests on extrapolated capacitor models
  all_rails_1ghz.png      all five rails bare on one axis
(constant-ESL inductive tail), not vendor measurements. The PDN data itself is measured to 5 GHz.
