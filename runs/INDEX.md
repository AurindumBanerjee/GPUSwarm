# GPUSwarm runs

Every run has its own folder: code/ (exact copy), output/ and logs/ (raw, git-ignored), summary/ and plots/ (tracked), README.txt, manifest.json. New runs: reference/new_run.py.

| run | status | start | what | code |
|---|---|---|---|---|
| decapstudy/run01_exp1_20260930 | finished | 2026-09-30 | Exp 1 decap study: 5 scopes x 4 bands PSO grid (staged), Grid 2 inversion-method timing, consistency checks; assumed-voltage targets. | commit e403d48 |
| decapstudy2/run01_exp2_stopped_20261004 | stopped (killed at Stage 1 run 15/45; targets redesigned) | 2026-10-04 | Exp 2 first launch: targets from Exp 1 B2 floors x [1.02..1.5], one B2-derived target set applied to all bands, Grid 1 stopped on the tighte | commit 655bc52 |
| decapstudy2/run02_floor_pass_20261004 | finished | 2026-10-04 | Floor pass: full-budget numpy PSO runs for the (key, band) floors Exp 1 cannot supply (B1: mer2, pll1v0, mphyvdd, ddr21 x10 runs; ddr_full x | commit ed9b2a1 |
| pso_sp/run01_full_budget_20261005 | finished | 2026-10-05 | PSO_SP full-budget runs (no early stop) on MPHY (6 problems) and VDDQ DDR3 (2 problems), numpy only, 3 runs each; lowest impedance each netw | commit a043279 |
| comparisons/run01_warmstart_20261006 | finished | 2026-10-06 | Warm-start comparison on the 21-port benchmark PDN: naive (40% verbatim) vs current (FIX 11 elite + jitter 0.02) at SB.py thresholds (10 run | commit 130fd29 |
| reference/run01_template_sample | created | 2026-10-08 | Toy PSO produced by reference/template_experiment.py: sample of every output file type (jsonl, log, txt, csv, png, manifest). | 6b369c7 template_experiment.py |
