# AllCPU - all four PDNs, CPU only

Own copy of the code: nothing in `PSO_REF21/`, `PSO_PCB28/`, `PSO_SP/`, `DecapStudy*/`, `ParallelStudy*/`, `Data/` or `Dataset/` is imported-and-modified or written to.
Core = `PSO_PCB28/pso_sp_core.py` (SB.py FIX 1-13, warm-start jitter 0, per-(run, n_caps) reseeding) with four marked ALLCPU changes: band ranges, a ratio objective, a patience stop, extra record fields.
Priority: **CPUParallel first**; CPUSerial is the idle-machine reference for theory, run later.

## The four PDNs (`common/allcpu_pdns.py`)

| key | board | N | obs ports | decap ports (allowance = N - n_obs) | grid | library | objective |
|---|---|---|---|---|---|---|---|
| pdn1 | 21-port reference | 21 | 1 (idx 0) | 20 | 1391 pts, 101 Hz - 99.8 MHz | legacy `decaps.mat` (3348) | peak abs(Z11), ohm; targets 60/50/45 mOhm |
| pdn2 | 28-port full PCB | 28 | 1 (idx 0) | 27 | 2828 pts, 100.9 Hz - 3 GHz (optimised on the whole grid) | Murata RLC (16171) | peak abs(Z11), ohm; targets 140/130/125/120 |
| pdn3 | MPHY package, all 5 rails | 73 | 5 (idx 0-4) | 68 | 5g: 110.9 Hz - 5 GHz (default) or 1g: - 997 MHz | Murata RLC on that grid (default) or `combined` (1g only) | max_k peak_k / floor_k |
| pdn4 | DDR3 board | 79 | 1 (idx 0) | 78 | same as pdn3 | same as pdn3 | peak abs(Z11), ohm |

Only observation ports are excluded from the allowance, for every PDN (read from the matrix). PDN 2 therefore allows 27 caps (ParallelStudy2 setting), not 14.

### Data configuration of PDN 3 / PDN 4 (`CFG = <grid>_<library>`, env `ALLCPU_GRID`, `ALLCPU_LIB`)

* `5g_murata` (default): boards extracted to 5 GHz from the Touchstone files, Murata RLC library resampled on that grid (`common/build_data.py`).
* `1g_murata`, `1g_combined`: the existing 1 GHz sets (GPUSwarm/Data, 2588 pts) with the Murata library resampled on that grid, or the old combined library (legacy 3348 + AVX 3347 = 6695 models, extrapolated above 99.83 MHz).
  These two exist for the library pilot: does the newer, wider-range Murata library lower the floors compared with the combined one? (Murata parts are measured to 3-6 GHz; the combined library to ~99.8 MHz.)

### Bands (PDN 3 / PDN 4), optimised one at a time, every record re-scored on all of them

`B20` (<= 20 MHz), `B50` (<= 50 MHz), `B200` (<= 200 MHz), `B1G` (<= 1 GHz), and on the 5g grid `B5G` (the whole grid, the canonical 100 Hz - 5 GHz range, used for optimisation too).
PDN 1 has its single native band; PDN 2 is optimised on its whole 3 GHz grid and re-scored on <= 1 GHz. Above ~200 MHz the board peaks are cavity resonances that capacitors cannot move
(`Data/MPHY/README.md`); the 5 GHz band is there to expose the impedance profile, expect its floor to be dominated by those peaks.

## Settings

Swarms (particles x iterations): **50x15, 100x15, 50x40, 100x40, 200x40**; warm start **20 % of the particles** for all (10/20/40), jitter 0, paired seeds `12345 + run*10000 + n_caps` for every method.
Methods numpy / solve (= "Proposed") / sm / iterative (pure_python is a separate baseline, not part of AllCPU). 4 runs per level (`--runs`). Per-run time limit 28800 s (`timed_out`).
PDN 3/4 patience: stop after 6 capacitor counts without a > 0.1 % gain (`stagnated`; PDN 1/2 off). A run stops the instant the running best meets its level.
PDN 3/4 levels per band: `floor + g x (bare - floor)`, g = 0.5 / 0.25 / 0.10 (T3 easiest ... T1 hardest), floors from `floor_pass.py` (rail-wise single-rail runs for MPHY, ratio objective, floor = 1).

## Layout

```
AllCPU/
  run_prep.sh                  server preparation + tests (nohup)         -> prep/prep_report.txt
  common/   allcpu_core.py  allcpu_pdns.py  job_worker.py  build_data.py  floor_pass.py  validate.py  bench_cost.py  prep_all.py  targets_allcpu.json
  CPUParallel/run_parallel.py  16 pinned single-thread processes (4 PDNs x 4 methods)
  CPUSerial/run_serial.py      one process, one thread, one PDN at a time
  DOUBTS.md                    open questions
  (created on use)  data_built/  floor_pass/<cfg>/  prep/  CPUParallel/out_<cfg>/  CPUSerial/out_<cfg>/
```
Output of both runners: `out_<cfg>/<pdn>/<swarm>/<method>/results_long.jsonl` (one record per run: placement, curve, time to target, all-band peaks, warm count ...) and
`.../<band>/<level>/run_k/` (log + plots). Resumable: the same command again skips finished runs.

## Server order of work (server paths: GPUSWARM_ROOT=/DATA/Aurindum/GPUSwarm, Dataset=/DATA/Aurindum/Swarming/Dataset)

```bash
cd /DATA/Aurindum/GPUSwarm/AllCPU
ps -eo pid,psr,pcpu,cmd --sort=-pcpu | head -5            # which logical CPUs do your two training runs sit on?
EXCLUDE=<those cpus, comma separated> nohup ./run_prep.sh > prep.out 2>&1 &
```
`run_prep.sh` (prep_all.py) does, resumably: build the 5 GHz boards and the Murata libraries (5g and 1g) -> validate the three configurations (selftests, bands, smoke PSO) ->
bench seconds-per-evaluation and hours-per-capacitor-count for every swarm -> a 16-job parallel smoke test on PDN 1 + PDN 2 (checks pinning/guard) -> the **library pilot**
(floors with 1g_combined vs 1g_murata, B20..B1G, 3 runs, 50x15) with the comparison table -> the 5g_murata floors (all five bands) -> final validate.
`--stop-after-pilot` ends after the comparison. Send back `prep/prep_report.txt`. Then:

```bash
cd CPUParallel && python run_parallel.py --dry                         # plan (rolling queue: all 50x15 jobs first, then 100x15, 50x40, 100x40, 200x40)
nohup python run_parallel.py --exclude <cpus> > logs/run_parallel.out 2>&1 &
# narrower: --swarms 50x15,100x40 --bands B50,B1G --levels T1,T2 --runs 2 --pdns pdn3,pdn4 --methods numpy,solve
# serial reference, later: cd ../CPUSerial && python run_serial.py --dry
```
The runners refuse to start (exit 2) while a needed floor is null. `ALLCPU_GRID` / `ALLCPU_LIB` select the configuration (default 5g_murata; outputs go to `out_<cfg>`).

## CPUParallel

16 jobs at a time = 4 methods x 4 PDNs, each one process pinned (taskset) to its own physical core (one logical CPU per physical core, core 0 of socket 0 and its sibling left free,
`--exclude` skips CPUs you name), BLAS = 1 thread. A job = (PDN, method, swarm): it works through its bands x levels x runs in sequence. Swarms run in the order given; a core freed by a
short job (PDN 1 finishes in hours, PDN 3/4 in days) immediately takes the next job (`--strict-waves` waits for the whole swarm instead). A guard checks every job's first `RUN_START`
line (particles, iterations, warm count) and stops everything on a mismatch.

## CPUSerial

One process at a time, single thread, PDN -> swarm -> method; nothing else running, so these are the idle-machine timings (for theory, run after the parallel study). `--core N` pins it.

## Cost (see `prep/prep_report.txt` for the measured table)

One capacitor count = particles x iterations evaluations: 750 (50x15), 1500 (100x15), 2000 (50x40), 4000 (100x40), 8000 (200x40). At N = 73 / 79 one evaluation is ~1 s single thread (laptop),
so a run needing ~10 capacitors costs ~2 h at 50x15 and ~22 h at 200x40; 4 bands (5 on 5g) x 3 levels x 4 runs per (method, swarm) makes PDN 3/4 weeks at the large swarms.
Use `--swarms/--bands/--levels/--runs` to cut. Raw data (`data_built/`, ~1.8 GB) is not for git.
