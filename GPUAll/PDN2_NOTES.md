# PDN 2 (28-port PCB): conditioning and the GPU verification limit

## Accuracy limit
`run_gpuall.py` checks every GPU arm against AllCPU's CPU numpy before a PDN's study (max relative deviation over random placements, all bands).
The limit is **1e-8 for PDN 1, 3 and 4** and **1e-6 for PDN 2** (`VERIFY_TOL` in `run_gpuall.py`).

Measured (5 GHz grid, Murata library): PDN 1/3/4 deviate by 1e-14 to 1e-15. PDN 2 deviates by 2.2e-7 (`gpu_numpy`) and 1.6e-7 (`woodbury_batch`, `fast`).
The limit of 1e-6 is ~5x above that, and far below the ~3e-5 that cond x machine epsilon allows (1.3e11 x 2.2e-16).
Consequence: on PDN 2 the exact arms are exact only to ~1e-7, so a placement that is a near-tie can rank differently from the CPU run and the trajectories may diverge.
Trajectory identity versus CPU numpy is therefore reported for PDN 2 as measured, not assumed.

## Why PDN 2 is ill-conditioned (measured with CPU numpy on `PSO_PCB28/Data/y_pcb28.mat`, 28 ports, 2828 frequencies)
| | PDN 1 | PDN 2 | PDN 3 |
|---|---|---|---|
| median cond(Y) | 3.8e2 | 2.6e6 | 1.1e3 |
| worst cond(Y) | 1.0e3 | 1.3e11 | 2.8e4 |
| entry range of Y (S) | 6e-8 .. 1.7e3 | 7e-9 .. 7.5e8 | 4e-12 .. 1.2e4 |

Two separate causes:
1. **Two ports are electrically the same node.** Ports 17 and 21 (0-based; 18 and 22 in 1-based numbering) have |Y17,17| = |Y21,21| = |Y17,21| = 2.4e6 S (others: 20 to 170 S), and the effective impedance between them is 4.2e-7 ohm
   (typical port pair: 0.24 ohm). Their rows/columns are nearly equal and opposite, so Y has one singular value of 4.8e6 against 2e2 .. 0.5 for the rest: median cond ~1e6 to 1e7.
   Diagonal scaling removes about 15x of this (median 1.7e5) but not the near-linear dependence of the two ports.
2. **Resonances.** The worst conditioning (1e10 to 1e11, 18 % of placement/frequency points above 1e8) is at frequency indices ~2350 to 2710 of the 3 GHz grid, where Y has a near-null direction spread over about a dozen ports (parallel resonances).

Not established: why the board model has two ports 0.4 micro-ohm apart (e.g. two ports on one via/pad or on one plane location, or an artefact of the extraction). That needs the board's port map.
Possible remedy, not applied (it changes the problem): merge the two ports into one before optimisation.

## Related
- The cost model's out-of-memory failure was unrelated: it uses synthetic matrices (F=3000, B=200, k=40 corner of its grid), fixed in commit "Skip cost-model cells that do not fit in GPU memory".
- The earlier pure-Python vs numpy difference of 1.2e-5 on this PDN has the same cause.
