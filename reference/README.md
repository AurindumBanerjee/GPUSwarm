# reference/

Starting point for every new GPUSwarm experiment.

| file | use |
|---|---|
| `template_experiment.py` | base template: settings block, `RUN_DIR` handling, code copy, `run.log` / `results_long.jsonl` / summary / plots / manifest. Copy it, rename it, replace `run_one()` |
| `new_run.py` | creates `runs/<experiment>/<run_id>/` with an exact code copy, launcher, manifest and README, and adds a row to `runs/INDEX.md` |
| `output_style/` | real samples of every output file type and `STYLE.md` describing their formats |

## New experiment in four steps

1. `cp reference/template_experiment.py MyExperiment/my_experiment.py`; set `EXPERIMENT`, replace `run_one()`, keep the settings as plain top-level assignments.
2. `python reference/new_run.py --experiment myexp --id run01_first_try --desc "what and why" --source MyExperiment/my_experiment.py --env NUM_RUNS=20 --env THREADS=16`
3. `nohup setsid bash runs/myexp/run01_first_try/run.sh > runs/myexp/run01_first_try/logs/driver.log 2>&1 < /dev/null &`
4. When it ends: look at `summary/` and `plots/`, write the result into `README.txt`, commit (`output/` and `logs/` are ignored).

## Rules that keep versions traceable

* One run = one folder = one code copy. Never edit a copy that produced results; to change anything, run `new_run.py` again with a new id (`--set 'KEY = value'` changes top-level assignments and the README records the exact diff).
* The code must write only under `RUN_DIR` (the template does). A script that writes next to itself cannot be isolated; make it read `RUN_DIR` first.
* Pin BLAS threads to 1 (the template and `run.sh` do); use `--taskset` to keep a job off the cores another timing run uses.
* Timer is `time.perf_counter()`; seeds are explicit and paired across arms; failed runs stay in the tables.
* Record environment/hardware in the manifest (the template does) and any concurrent job in the README of the run it could have disturbed.
