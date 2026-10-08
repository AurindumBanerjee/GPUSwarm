# GPUSwarm

PSO implementation speed-up using GPU-based methodologies, and the decoupling-capacitor placement studies built on it.
Runs execute on the EEDept server (`/DATA/Aurindum/GPUSwarm`); this repository is the record of the code, run manifests, summaries
and plots. Raw outputs and data (`*.mat`) are not tracked.

## Layout

| path | what |
|---|---|
| `reference/` | **start here for new work**: base template script, run-folder creator `new_run.py`, and `output_style/` samples of every output file type (txt, log, csv, jsonl, png, json) |
| `runs/` | **one folder per run**: `runs/<experiment>/<run_id>/{code,output,logs,summary,plots}` + `run.sh`, `manifest.json`, `README.txt`; `runs/INDEX.md` lists them |
| `CPUTest/` | `SB.py`, `SB2.py`: CPU reference PSO (the algorithm the studies are checked against) |
| `DecapStudy/` | Exp 1: PSO grid over 5 scopes x bands, inversion-method timing, consistency checks (finished; results in `DecapStudy/results`) |
| `DecapStudy2/` | Exp 2: targets from Exp 1 floors (`derive_targets.py`, `targets.json`, `floor_pass.py`), per-band targets, noise flags |
| `PSO_SP/` | single-pass PSO on the MPHY package (5 rails + whole package) and the VDDQ DDR3 board |
| `Comparisons/` | warm-start comparison on the 21-port benchmark PDN (`warmstart_thresholds.py`, `warmstart_arms.py`) |
| `GPUBench.py`, `GPUBench2.py` | GPU (PyTorch) ports of ScratchBench; `GPUBench2.py` adds Woodbury / batched / pruned evaluation |
| `Data/` | `Reference/` (21-port benchmark PDN: `y2.mat`, `decaps.mat`, `freq2.mat`), `Decaps/`, `MPHY/`, `VDDQ_DDR3/`; READMEs tracked, `*.mat` ignored |
| `MinTime/` | older GPU timing result folders (legacy, tracked as they were) |

The code folders (`DecapStudy*`, `PSO_SP`, `Comparisons`, `CPUTest`) hold the **current** source of each study. A **run** is a frozen
copy of the code together with everything it produced: results are always traced to `runs/<experiment>/<run_id>/code/`, never to
the current source.

## Runs convention

```
runs/<experiment>/<run_id>/
    code/        exact copy of the code that ran (tracked)
    output/      raw results            (git-ignored)
    logs/        console / study logs   (git-ignored)
    summary/     tables, reports        (tracked)
    plots/       figures                (tracked)
    run.sh  manifest.json  README.txt   launcher, what/when/where (md5, git commit, env, times), question and result
```

* New run: `python reference/new_run.py --experiment E --id runNN_name --desc "..." --source path/to/script.py [--set 'KEY = v'] [--env K=V]`, then `bash runs/E/runNN_name/run.sh`. Details in `reference/README.md`.
* Never edit a copy that has produced results; a change is a new run with a new id.
* Runs before this convention (the five in `runs/INDEX.md` dated 2026-09-30 .. 2026-10-06) were **retrofitted**: code copied from the git commit that ran, outputs copied from their original folders (the originals were not moved).
* Code that predates the convention writes next to itself; for isolated runs make it read `RUN_DIR` (see `reference/template_experiment.py`).

## Working on the server

* Threads: BLAS/OMP pinned to 1 thread (set before numpy is imported); parallelism is particle threads (`STUDY_THREADS` / `THREADS`), at most one per physical core (32 on the server).
* Long jobs: `nohup setsid bash run.sh > logs/driver.log 2>&1 < /dev/null &`; use `taskset -c` to keep a job off the cores a timing run uses.
* The server cannot reach GitHub: code moves with `git bundle` (create on one side, `git fetch bundle`, `git merge --ff-only`).
* Data: `Data/**/*.mat` are not in git; copy them separately and check md5s.
