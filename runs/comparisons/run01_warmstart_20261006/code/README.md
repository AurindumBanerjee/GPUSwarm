# Comparisons

Warm-start studies on the **original 21-port benchmark PDN**
(`Data/Reference/{y2,decaps,freq2}.mat`) — the dataset `ScratchBench` and
`CPUTest/SB.py` were written against. Not MPHY, not DDR3.

`SB.py` added a warm start (FIX 4) and then, in FIX 11, refined it with an exact
elite particle plus Gaussian jitter on the carried genes. The argument there is
that a naive warm start freezes the carried dimensions, because with every warm
particle identical in those genes both PSO velocity terms vanish. These two
scripts test that claim rather than assuming it.

| script | question | thresholds | cost |
|---|---|---|---|
| `warmstart_thresholds.py` | how fast does each warm start reach a target? | SB.py's 0.05 / 0.045 / 0.04 / 0.03 | cheap — runs stop on target |
| `warmstart_arms.py` | does the warm start change the answer at all? | none, full capacitor budget | expensive — nothing stops early |

`warmstart_thresholds.py` compares two arms (naive vs current).
`warmstart_arms.py` adds a third, no warm start at all, which is the original
ScratchBench behaviour — that is the larger design question, FIX 4 before FIX 11.

Both hold everything else constant: numpy as the only inversion method, SB.py's
PSO constants, the same discretisation and collision resolver, and **paired
seeds** — every arm at run `r` and capacitor count `n` starts from the same RNG
state, so any difference is the seeding rule and nothing else.

Both also report the mechanism directly: `frozen_stages` counts stages whose
warm block began with exactly zero spread across the carried genes, and
`spread_start` / `spread_end` say whether those genes ever moved.

## Running

```
python warmstart_thresholds.py --estimate     # measured cost on this machine, no PSO
python warmstart_arms.py --estimate

THREADS=16 NUM_RUNS=10 nohup python -u warmstart_thresholds.py > thr.out 2>&1 &
THREADS=16 NUM_RUNS=5  nohup python -u warmstart_arms.py       > arms.out 2>&1 &
```

Env: `THREADS` (1), `NUM_RUNS`, `PDN_DATA` (`../Data/Reference`).

Run `--estimate` first. Measured on a 2-core machine at 19–25 ms per evaluation,
the threshold sweep projects to roughly 2 h and the full-budget sweep to 1.5 h;
on a 32-core server both are far quicker. Results append to
`*_results.jsonl` — delete between sweeps or the summary mixes old rows in.

Outputs are gitignored; the scripts are not.
