# Decap placement study -- report

Ranking band (CARE_BAND): B2. Targets: Ztarget = V*5%/dI (assumptions in study_config.py).

# Stage 1 screening

Ranking band: **B2** (median over runs of the re-scored peak/Ztarget; <=1 meets target). Cells with x_vs_best > 2.0 are dropped.

| scope | label | band | n_runs | med_R_B1 | med_R_B2 | med_R_B3 | med_R_B4 | x_vs_best | keep | success | caps_used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | MPHY rail-wise | B1 | 3 | 3.212 | 8.063 | 16.06 | 3.136 | 1.011 | True | 0 | 27 |
| A1 | MPHY rail-wise | B2 | 3 | 3.268 | 7.977 | 16.06 | 2.59 | 1 | True | 0 | 56 |
| A1 | MPHY rail-wise | B3 | 3 | 3.257 | 8.09 | 15.99 | 2.294 | 1.014 | True | 0 | 66 |
| A1 | MPHY rail-wise | B4 | 3 | 3.257 | 8.09 | 15.99 | 2.492 | 1.014 | True | 0 | 66 |
| A2 | MPHY whole package | B1 | 3 | 3.211 | 8.063 | 16.06 | 2.776 | 1.015 | True | 0 | 68 |
| A2 | MPHY whole package | B2 | 3 | 3.28 | 7.941 | 16.05 | 3.11 | 1 | True | 0 | 50 |
| A2 | MPHY whole package | B3 | 3 | 3.241 | 8.048 | 16.02 | 2.832 | 1.014 | True | 0 | 68 |
| A2 | MPHY whole package | B4 | 3 | 3.254 | 8.077 | 16.08 | 2.311 | 1.017 | True | 0 | 68 |
| A3 | MPHY improvable only | B1 | 3 | 0.9921 | 3.59 | 5.191 | 2.973 | 1.008 | True | 1 | 9 |
| A3 | MPHY improvable only | B2 | 3 | 0.9925 | 3.561 | 5.066 | 2.837 | 1 | True | 0 | 38 |
| A3 | MPHY improvable only | B3 | 3 | 1.034 | 3.598 | 4.797 | 2.815 | 1.011 | True | 0 | 38 |
| A3 | MPHY improvable only | B4 | 3 | 1.062 | 3.609 | 5.07 | 2.306 | 1.014 | True | 0 | 38 |
| A4 | DDR3 21-port | B1 | 3 | 0.9834 | 2.555 | 5.102 | 2.62 | 1.153 | True | 1 | 9 |
| A4 | DDR3 21-port | B2 | 3 | 1.315 | 2.216 | 4.864 | 2.498 | 1 | True | 0 | 20 |
| A4 | DDR3 21-port | B3 | 3 | 1.066 | 2.335 | 4.42 | 2.303 | 1.054 | True | 0 | 20 |
| A4 | DDR3 21-port | B4 | 3 | 1.066 | 2.335 | 4.432 | 2.295 | 1.054 | True | 0 | 20 |
| A5 | DDR3 79-port | B1 | 3 | 0.975 | 2.401 | 4.778 | 2.716 | 2.1 | False | 1 | 3 |
| A5 | DDR3 79-port | B2 | 3 | 0.6231 | 1.144 | 2.43 | 1.381 | 1 | True | 0 | 78 |
| A5 | DDR3 79-port | B3 | 3 | 0.6439 | 1.338 | 2.247 | 1.325 | 1.17 | True | 0 | 78 |
| A5 | DDR3 79-port | B4 | 3 | 0.5464 | 1.194 | 2.264 | 1.287 | 1.044 | True | 0 | 78 |


## Grid 1 matrix (stage 2; R_x = median peak/Ztarget re-scored on band x)

| scope | band | n_runs | caps_used | R_B1 | R_B2 | R_B3 | R_B4 | IMP_B1 | IMP_B2 | IMP_B3 | IMP_B4 | caps_needed | conv_iter | success_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | B1 | 10 | 18.5 | 3.211 | 8.063 | 16.06 | 3.135 | 0.9981 | 3.641 | 5.207 | 3.135 |  | 172 | 0 |
| A1 | B2 | 10 | 49.5 | 3.28 | 7.942 | 16.05 | 2.588 | 1.041 | 3.559 | 4.944 | 2.588 |  | 211 | 0 |
| A1 | B3 | 10 | 48.5 | 3.268 | 8.097 | 15.98 | 2.27 | 1.23 | 3.729 | 4.636 | 2.27 |  | 211 | 0 |
| A1 | B4 | 10 | 50.5 | 3.268 | 8.097 | 15.98 | 2.268 | 1.242 | 3.674 | 4.636 | 2.268 |  | 211 | 0 |
| A2 | B1 | 10 | 65.5 | 3.211 | 8.063 | 16.06 | 2.825 | 1.015 | 3.6 | 5.074 | 2.825 |  | 968.5 | 0 |
| A2 | B2 | 10 | 55.5 | 3.277 | 7.941 | 16.05 | 2.856 | 1.039 | 3.585 | 5.407 | 2.856 |  | 823.5 | 0 |
| A2 | B3 | 10 | 53 | 3.27 | 8.099 | 15.98 | 3.068 | 1.068 | 3.587 | 5.393 | 3.068 |  | 782 | 0 |
| A2 | B4 | 10 | 57.5 | 3.248 | 8.07 | 16.07 | 2.271 | 1.228 | 3.693 | 5.066 | 2.271 |  | 848.5 | 0 |
| A3 | B1 | 10 | 4.5 | 0.9978 | 3.62 | 5.189 | 3.119 | 0.9978 | 3.62 | 5.189 | 3.119 | 4.5 | 60 | 1 |
| A3 | B2 | 10 | 32.5 | 1.062 | 3.559 | 5.467 | 2.87 | 1.062 | 3.559 | 5.467 | 2.87 |  | 480.5 | 0 |
| A3 | B3 | 10 | 38 | 1.074 | 3.658 | 4.668 | 2.791 | 1.074 | 3.658 | 4.668 | 2.791 |  | 556 | 0 |
| A3 | B4 | 10 | 30.5 | 1.184 | 3.663 | 5.061 | 2.277 | 1.184 | 3.663 | 5.061 | 2.277 |  | 450 | 0 |
| A4 | B1 | 10 | 4.5 | 0.9861 | 2.589 | 5.196 | 2.668 |  |  |  |  | 4.5 | 54.5 | 1 |
| A4 | B2 | 10 | 20 | 1.342 | 2.092 | 4.788 | 2.458 |  |  |  |  |  | 286 | 0 |
| A4 | B3 | 10 | 20 | 1.304 | 2.711 | 4.142 | 2.56 |  |  |  |  |  | 286 | 0 |
| A4 | B4 | 10 | 20 | 1.093 | 2.261 | 4.319 | 2.221 |  |  |  |  |  | 286 | 0 |
| A5 | B2 | 10 | 78 | 0.6653 | 1.079 | 2.425 | 1.379 |  |  |  |  | 59 | 1156 | 0.1 |
| A5 | B3 | 10 | 78 | 0.6213 | 1.318 | 2.109 | 1.301 |  |  |  |  |  | 1156 | 0 |
| A5 | B4 | 10 | 78 | 0.6254 | 1.166 | 2.134 | 1.213 |  |  |  |  |  | 1156 | 0 |


### Per-rail median peaks (ohm)
| scope | band | rail | target_ohm | peak_B1_ohm | peak_B2_ohm | peak_B3_ohm | peak_B4_ohm |
|---|---|---|---|---|---|---|---|
| A1 | B1 | mer1 | 0.2 | 0.6422 | 1.613 | 3.212 | 0.3277 |
| A1 | B1 | mer2 | 0.2 | 0.1996 | 0.7283 | 0.9931 | 0.4794 |
| A1 | B1 | mphyvdd | 0.5 | 0.4954 | 1.243 | 2.603 | 1.151 |
| A1 | B1 | pll1v0 | 0.5 | 0.4976 | 1.244 | 2.534 | 1.568 |
| A1 | B1 | pll1v8 | 1.8 | 4.541 | 11.66 | 23.4 | 2.834 |
| A1 | B2 | mer1 | 0.2 | 0.656 | 1.588 | 3.211 | 0.3275 |
| A1 | B2 | mer2 | 0.2 | 0.195 | 0.7118 | 0.9711 | 0.4688 |
| A1 | B2 | mphyvdd | 0.5 | 0.5198 | 1.107 | 2.472 | 1.093 |
| A1 | B2 | pll1v0 | 0.5 | 0.4266 | 1.035 | 2.092 | 1.294 |
| A1 | B2 | pll1v8 | 1.8 | 4.715 | 11.47 | 23.31 | 2.823 |
| A1 | B3 | mer1 | 0.2 | 0.6536 | 1.619 | 3.196 | 0.326 |
| A1 | B3 | mer2 | 0.2 | 0.232 | 0.7458 | 0.8368 | 0.4321 |
| A1 | B3 | mphyvdd | 0.5 | 0.5513 | 1.235 | 2.318 | 1.025 |
| A1 | B3 | pll1v0 | 0.5 | 0.6149 | 1.236 | 1.835 | 1.135 |
| A1 | B3 | pll1v8 | 1.8 | 4.741 | 11.62 | 23.15 | 2.804 |
| A1 | B4 | mer1 | 0.2 | 0.6536 | 1.619 | 3.196 | 0.326 |
| A1 | B4 | mer2 | 0.2 | 0.2196 | 0.7348 | 0.8498 | 0.4102 |
| A1 | B4 | mphyvdd | 0.5 | 0.5513 | 1.235 | 2.318 | 1.025 |
| A1 | B4 | pll1v0 | 0.5 | 0.6212 | 1.242 | 1.833 | 1.134 |
| A1 | B4 | pll1v8 | 1.8 | 4.741 | 11.62 | 23.15 | 2.804 |
| A2 | B1 | mer1 | 0.2 | 0.6421 | 1.613 | 3.212 | 0.3277 |
| A2 | B1 | mer2 | 0.2 | 0.1966 | 0.7201 | 0.9651 | 0.4659 |
| A2 | B1 | mphyvdd | 0.5 | 0.493 | 1.227 | 2.537 | 1.122 |
| A2 | B1 | pll1v0 | 0.5 | 0.4723 | 1.148 | 2.283 | 1.412 |
| A2 | B1 | pll1v8 | 1.8 | 4.671 | 11.67 | 23.37 | 2.83 |
| A2 | B2 | mer1 | 0.2 | 0.6554 | 1.588 | 3.21 | 0.3275 |
| A2 | B2 | mer2 | 0.2 | 0.1971 | 0.717 | 0.9696 | 0.468 |
| A2 | B2 | mphyvdd | 0.5 | 0.5025 | 1.23 | 2.543 | 1.125 |
| A2 | B2 | pll1v0 | 0.5 | 0.4795 | 1.158 | 2.308 | 1.428 |
| A2 | B2 | pll1v8 | 1.8 | 4.671 | 11.67 | 23.36 | 2.83 |
| A2 | B3 | mer1 | 0.2 | 0.6539 | 1.62 | 3.196 | 0.326 |
| A2 | B3 | mer2 | 0.2 | 0.1954 | 0.7175 | 0.968 | 0.4673 |
| A2 | B3 | mphyvdd | 0.5 | 0.4933 | 1.223 | 2.525 | 1.117 |
| A2 | B3 | pll1v0 | 0.5 | 0.5102 | 1.229 | 2.48 | 1.534 |
| A2 | B3 | pll1v8 | 1.8 | 4.688 | 11.69 | 23.38 | 2.832 |
| A2 | B4 | mer1 | 0.2 | 0.6495 | 1.614 | 3.214 | 0.3278 |
| A2 | B4 | mer2 | 0.2 | 0.2164 | 0.7387 | 0.9377 | 0.4526 |
| A2 | B4 | mphyvdd | 0.5 | 0.4919 | 1.224 | 2.533 | 1.12 |
| A2 | B4 | pll1v0 | 0.5 | 0.6141 | 1.239 | 1.835 | 1.135 |
| A2 | B4 | pll1v8 | 1.8 | 4.674 | 11.67 | 23.37 | 2.831 |
| A3 | B1 | mer2 | 0.2 | 0.1967 | 0.7241 | 0.9882 | 0.477 |
| A3 | B1 | mphyvdd | 0.5 | 0.4942 | 1.237 | 2.595 | 1.147 |
| A3 | B1 | pll1v0 | 0.5 | 0.4945 | 1.237 | 2.521 | 1.559 |
| A3 | B2 | mer2 | 0.2 | 0.195 | 0.7117 | 0.9711 | 0.4688 |
| A3 | B2 | mphyvdd | 0.5 | 0.5106 | 1.267 | 2.611 | 1.155 |
| A3 | B2 | pll1v0 | 0.5 | 0.469 | 1.154 | 2.32 | 1.435 |
| A3 | B3 | mer2 | 0.2 | 0.2112 | 0.7315 | 0.9332 | 0.4505 |
| A3 | B3 | mphyvdd | 0.5 | 0.5186 | 1.252 | 2.331 | 1.031 |
| A3 | B3 | pll1v0 | 0.5 | 0.4708 | 1.135 | 2.256 | 1.396 |
| A3 | B4 | mer2 | 0.2 | 0.214 | 0.7327 | 0.9416 | 0.4548 |
| A3 | B4 | mphyvdd | 0.5 | 0.4914 | 1.225 | 2.53 | 1.119 |
| A3 | B4 | pll1v0 | 0.5 | 0.592 | 1.234 | 1.839 | 1.137 |
| A4 | B1 | ddr | 0.0675 | 0.06656 | 0.1748 | 0.3507 | 0.1801 |
| A4 | B2 | ddr | 0.0675 | 0.09061 | 0.1412 | 0.3232 | 0.1659 |
| A4 | B3 | ddr | 0.0675 | 0.08803 | 0.183 | 0.2796 | 0.1728 |
| A4 | B4 | ddr | 0.0675 | 0.07377 | 0.1526 | 0.2916 | 0.1499 |
| A5 | B2 | ddr | 0.0675 | 0.04491 | 0.07281 | 0.1637 | 0.09307 |
| A5 | B3 | ddr | 0.0675 | 0.04194 | 0.08898 | 0.1423 | 0.08783 |
| A5 | B4 | ddr | 0.0675 | 0.04221 | 0.07871 | 0.144 | 0.08188 |


## Rail responsiveness (best single model on every pad vs bare)
mer1 and pll1v8 barely respond: the observation port sits at the die and package inductance hides the pads. They stay in A1/A2; A3 exists because of them.
| rail | B1_bare_ohm | B2_bare_ohm | B3_bare_ohm | B4_bare_ohm | B1_gain_% | B2_gain_% | B3_gain_% | B4_gain_% |
|---|---|---|---|---|---|---|---|---|
| mer1 | 0.6549 | 1.627 | 3.241 | 0.3306 | 1.911 | 1.701 | 1.345 | 1.345 |
| mer2 | 0.2329 | 0.725 | 1.16 | 0.56 | 26.36 | 1.787 | 25.69 | 25.69 |
| pll1v0 | 0.6643 | 1.663 | 3.516 | 2.175 | 43.4 | 37.41 | 47.03 | 47.03 |
| pll1v8 | 4.835 | 12.09 | 24.34 | 2.947 | 4.888 | 4.394 | 4.756 | 4.756 |
| mphyvdd | 0.5785 | 1.453 | 3.152 | 1.394 | 24.32 | 24.18 | 25.76 | 25.76 |
| ddr21 | 0.1008 | 0.2507 | 0.5017 | 0.2576 | 40.68 | 42.38 | 54.03 | 44.2 |


## Grid 2 timing
| N | method | sec_per_eval | source | speedup_eval_vs_pure | max_rel_dev_vs_numpy | runs | success_rate | timeouts | n_evals_med | time_to_target_s_med | placement_match_vs_numpy | max_ratio_dev_vs_numpy | speedup_total_vs_pure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | pure_python | 1.555 | measured | 1 | 9.952e-16 | 2 | 0 | 2 | 1201 | 1855 | 0 | 0.02355 | 1 |
| 16 | numpy | 0.04232 | measured | 36.73 | 0 | 5 | 0 | 0 | 1.125e+04 | 443.4 | 1 | 0 | 4.183 |
| 16 | solve | 0.03774 | measured | 41.19 | 9.952e-16 | 5 | 0 | 0 | 1.125e+04 | 320.4 | 1 | 2.22e-15 | 5.789 |
| 16 | sm | 0.1572 | measured | 9.889 | 7.383e-16 | 5 | 0 | 0 | 1.125e+04 | 1142 | 1 | 1.776e-15 | 1.625 |
| 16 | iterative | 0.0752 | measured | 20.67 | 0 | 5 | 0 | 0 | 1.125e+04 | 632 | 0 | 0.01988 | 2.935 |
| 21 | pure_python | 3.401 | measured | 1 | 3.363e-16 | 2 | 0 | 2 | 551 | 1868 | 0 | 0.7424 | 1 |
| 21 | numpy | 0.05375 | measured | 63.27 | 0 | 5 | 0 | 0 | 1.5e+04 | 728.5 | 1 | 0 | 2.564 |
| 21 | solve | 0.04872 | measured | 69.8 | 5.044e-16 | 5 | 0 | 0 | 1.5e+04 | 586.6 | 1 | 8.882e-16 | 3.184 |
| 21 | sm | 0.1772 | measured | 19.19 | 4.73e-16 | 5 | 0 | 5 | 1.325e+04 | 1803 | 0.2 | 6.418e-05 | 1.036 |
| 21 | iterative | 0.07775 | measured | 43.74 | 0.07901 | 5 | 0 | 0 | 1.5e+04 | 911.6 | 0 | 0.191 | 2.048 |
| 73 | pure_python | 142.9 | extrapolated n^3 | 1 |  |  |  |  |  |  |  |  |  |
| 73 | numpy | 0.2599 | measured | 549.6 | 0 | 5 | 0 | 5 | 7551 | 1808 | 1 | 0 | 596.5 |
| 73 | solve | 0.1566 | measured | 912.1 | 0 | 5 | 0 | 5 | 1.255e+04 | 1802 | 0 | 0.03352 | 598.7 |
| 73 | sm | 0.724 | measured | 197.3 | 8.755e-16 | 5 | 0 | 5 | 5701 | 1804 | 0 | 0.0006435 | 598.1 |
| 73 | iterative | 0.4248 | measured | 336.3 | 0 | 5 | 0 | 5 | 4851 | 1806 | 0 | 0.03546 | 597.2 |
| 79 | pure_python | 181.1 | extrapolated n^3 | 1 |  |  |  |  |  |  |  |  |  |
| 79 | numpy | 0.3085 | measured | 587 | 0 | 5 | 0 | 5 | 6201 | 1809 | 1 | 0 | 620.5 |
| 79 | solve | 0.1639 | measured | 1105 | 6.472e-16 | 5 | 0 | 5 | 1.2e+04 | 1803 | 0 | 0.3255 | 622.8 |
| 79 | sm | 0.8382 | measured | 216 | 7.868e-16 | 5 | 0 | 5 | 5251 | 1823 | 0 | 0.123 | 616 |
| 79 | iterative | 0.5141 | measured | 352.2 | 0 | 5 | 0 | 5 | 3951 | 1806 | 0 | 0.4658 | 621.6 |


Measured pure_python scaling exponent (N=16 -> 21): 2.88 (n^3 assumed for N=73/79)


## Consistency check (B1)
A1 decomposition validated: combined-network rail impedances match the isolated results.  reduction check 1.52e-13, worst combined peak deviation 1.93e-05
| rail | caps | peak_isolated_ohm | peak_alone_in_full_ohm | peak_combined_ohm | max_rel_dev_alone | max_rel_dev_combined | peak_rel_dev_combined |
|---|---|---|---|---|---|---|---|
| mer1 | 5 | 3.21 | 3.21 | 3.21 | 9.987e-15 | 3.552e-07 | 5.346e-08 |
| mer2 | 1 | 0.9919 | 0.9919 | 0.9919 | 1.104e-14 | 6.992e-06 | 5.298e-06 |
| pll1v0 | 1 | 2.447 | 2.447 | 2.447 | 3.644e-14 | 3.045e-05 | 1.933e-05 |
| pll1v8 | 2 | 23.4 | 23.4 | 23.4 | 1.94e-14 | 4.992e-06 | 5.384e-08 |
| mphyvdd | 1 | 2.598 | 2.598 | 2.598 | 1.519e-13 | 1.662e-07 | 8.98e-08 |


## Consistency check (B2)
A1 decomposition validated: combined-network rail impedances match the isolated results.  reduction check 1.58e-13, worst combined peak deviation 2.71e-05
| rail | caps | peak_isolated_ohm | peak_alone_in_full_ohm | peak_combined_ohm | max_rel_dev_alone | max_rel_dev_combined | peak_rel_dev_combined |
|---|---|---|---|---|---|---|---|
| mer1 | 6 | 3.211 | 3.211 | 3.211 | 1.166e-14 | 8.647e-06 | 6.885e-06 |
| mer2 | 10 | 0.9682 | 0.9682 | 0.9682 | 1.379e-14 | 7.5e-06 | 6.328e-06 |
| pll1v0 | 14 | 2.144 | 2.144 | 2.144 | 3.333e-14 | 0.0001651 | 2.711e-05 |
| pll1v8 | 9 | 23.29 | 23.29 | 23.29 | 2.106e-14 | 2.317e-05 | 1.606e-05 |
| mphyvdd | 6 | 2.474 | 2.474 | 2.474 | 1.578e-13 | 2.639e-06 | 1.661e-06 |


## Consistency check (B3)
A1 decomposition validated: combined-network rail impedances match the isolated results.  reduction check 1.52e-13, worst combined peak deviation 6.98e-06
| rail | caps | peak_isolated_ohm | peak_alone_in_full_ohm | peak_combined_ohm | max_rel_dev_alone | max_rel_dev_combined | peak_rel_dev_combined |
|---|---|---|---|---|---|---|---|
| mer1 | 11 | 3.196 | 3.196 | 3.196 | 9.973e-15 | 5.962e-06 | 7.701e-07 |
| mer2 | 8 | 0.8353 | 0.8353 | 0.8353 | 1.308e-14 | 1.342e-05 | 4.825e-06 |
| pll1v0 | 5 | 1.827 | 1.827 | 1.827 | 3.719e-14 | 0.0002708 | 6.981e-06 |
| pll1v8 | 13 | 23.15 | 23.15 | 23.15 | 2.427e-14 | 6.453e-07 | 6.884e-08 |
| mphyvdd | 4 | 2.314 | 2.314 | 2.314 | 1.516e-13 | 2.571e-07 | 9.94e-08 |


## Consistency check (B4)
A1 decomposition validated: combined-network rail impedances match the isolated results.  reduction check 1.52e-13, worst combined peak deviation 6.98e-06
| rail | caps | peak_isolated_ohm | peak_alone_in_full_ohm | peak_combined_ohm | max_rel_dev_alone | max_rel_dev_combined | peak_rel_dev_combined |
|---|---|---|---|---|---|---|---|
| mer1 | 11 | 3.196 | 3.196 | 3.196 | 9.973e-15 | 5.961e-06 | 7.699e-07 |
| mer2 | 8 | 0.8433 | 0.8433 | 0.8433 | 1.4e-14 | 1.177e-05 | 5.577e-06 |
| pll1v0 | 5 | 1.827 | 1.827 | 1.827 | 3.719e-14 | 0.0002707 | 6.981e-06 |
| pll1v8 | 13 | 23.15 | 23.15 | 23.15 | 2.427e-14 | 6.453e-07 | 6.885e-08 |
| mphyvdd | 4 | 2.314 | 2.314 | 2.314 | 1.516e-13 | 2.67e-07 | 1.08e-07 |


## Recommendation
Ranking band: **B2**. Winning band by geometric-mean relative score: **B2** (<=50 MHz); scores {"B1": 1.067, "B2": 1.0, "B3": 1.111, "B4": 1.045}.

- MPHY (compared on the improvable rails mer2/pll1v0/mphyvdd, the only set common to A1-A3): best scope **A3 MPHY improvable only** at B2, median IMP_B2=3.559, success rate 0.00, median caps used 32.

- DDR3: best scope **A5 DDR3 79-port** at B2, median R_B2=1.079, success rate 0.10, median caps used 78.
