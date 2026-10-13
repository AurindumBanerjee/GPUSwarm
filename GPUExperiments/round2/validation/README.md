# Validation experiments

| id | script | what we try | what to look at |
|---|---|---|---|
| v1 | `v1_hard_target_stats.py` | 100 runs (`NUM_RUNS_HARD`) of fast and polish at the hard target(s) (`HARD_TARGETS`, default 0.04) + 20 numpy runs for timing. Wilson 95 % intervals, Fisher exact test, time per success. | `summary/hard_target_summary.txt`, `plots/success_ci.png` |
| v2 | `v2_more_pdns.py` | numpy vs fast on MPHY rails and DDR3 (`DATASETS`) at relative targets below 50 MHz; then every MPHY rail's best placement installed together in the full 73-port package to check rails optimised separately still meet their targets. | `summary/pdn_summary.txt`, `plots/before_after_*`, `plots/mphy_all_rails_combined.png` |
