# AllCPU - answered, assumed and still open

## Answered by you (implemented)
1. Four PDNs = 21-port, 28-port, MPHY full (5 obs), DDR3 full (1 obs).  2. MPHY/DDR optimised on several bands up to 5 GHz (B20, B50, B200, B1G, B5G), 5 GHz band used for optimisation too.
3. PDN 2 stays on its 3 GHz grid.  4. New (Murata) vs old (combined) library pilot before committing.  5. Patience stop accepted.  6. Allowance = ports - observation ports for all PDNs.
7. Parallel first, serial for theory.  8. Imbalance of job lengths accepted.  9. Server paths /DATA/Aurindum/..., cores free except your two training runs.  10. Swarms 50x15, 100x15, 50x40, 100x40, 200x40 (warm 20 %).

## Assumptions I made (change if wrong)
* **The pilot compares floors** (lowest peak reached, numpy, 50x15, 3 runs, patience 6) of `1g_combined` vs `1g_murata` on B20/B50/B200/B1G. "Improves" is read as a lower floor; the table prints % improvement per rail / per DDR band.
  The 5 GHz configuration can only use the Murata library (the combined one stops at 1 GHz), so the pilot decides nothing for 5g beyond showing what the library change does where both exist.
* **Library range:** most Murata parts are measured to 3-6 GHz, some only to 3 GHz; above a part's last measured point the log-log extrapolation is used (as for PDN 2). Treat B5G results above ~3 GHz with that in mind.
* **Patience:** 6 capacitor counts, relative gain threshold 0.1 % (`PATIENCE_TOL`). "Major improvement" could mean 1 % - the tolerance is one env variable.
* **Levels for PDN 3/4** are 50 / 25 / 10 % of the way from floor to bare impedance (floors from 50x15 floor runs). Larger swarms may beat the 50x15 floor, then T1 is easier than intended, never unreachable.
* **MPHY objective:** ratio max_k peak_k / floor_k with rail-wise floors; the five single-rail files are used only for the floors.
* **Warm start** 20 % with jitter 0 for all PDNs (DecapStudy used 40 % with jitter 0.02).
* **Runs per level:** 4 (as in ParallelStudy). 4 methods; iterative is an approximation and untested at N = 73 / 79.
* CPUParallel uses the first 8 physical cores of each socket (minus core 0 and `--exclude`); the load regime changes late in the study when only PDN 3/4 jobs remain.

## Still open - please answer
A. **Cost.** With 5 swarms, up to 5 bands x 3 levels x 4 runs x 4 methods per swarm, PDN 3/4 at 100x40 and 200x40 are weeks to months of CPU time per job. Which subset do you want for the large swarms
   (e.g. only B1G and B5G? 2 runs? only numpy and solve?) - or should I order the queue so the cheap swarms finish first and the big ones run as far as time allows (that is the current order)?
B. **Does 200x40 apply to PDN 1 / PDN 2 as well?** Currently yes (all five swarms on all four PDNs). PDN 1/2 are cheap, so this is fine unless you want the paper's two PDNs limited to 50x15 and 100x40.
C. **What counts as an improvement in the library pilot?** Proposal: new library lowers the floor by >= 5 % on a majority of the (rail, band) cells (MPHY) and DDR bands. If it does not, should PDN 3/4 keep the combined library on the 1 GHz grid (bands up to B1G only) and use Murata only for B5G?
D. **PDN 1 library** stays the legacy `decaps.mat` (as in the earlier studies); `PDN1_LIB=murata` exists but its whole-number targets would need re-deriving. Leave as is?
E. **Patience tolerance** 0.1 % or 1 %?
F. **Which CPUs do your training runs sit on?** Give them as `EXCLUDE` (the prep and the parallel runner then avoid them); otherwise the first physical cores are taken.
