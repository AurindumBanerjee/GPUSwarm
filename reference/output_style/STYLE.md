# Output style reference

Every file here is a real output of `reference/template_experiment.py` (a toy PSO), produced through `reference/new_run.py`.
A new experiment should produce the same set, with its own columns/fields. Parsers in this repo (and in the Swarming repo's
`summarize_baseline.py`) rely on the line formats below.

| file | format | rules |
|---|---|---|
| `results_long.jsonl` | one JSON object per run, all factor levels as fields | append-only; one run = one line; include `seed`, `success`, the time (`-1` on failure), the best value, and whatever curve/placement lets the result be re-scored offline |
| `run.log` | `YYYY-mm-dd HH:MM:SS,mmm LEVEL: TAG | key=value | ...` | tags: `RUN_START` (all settings of the run), `STAGE` (progress, optional), `RESULT` (final numbers, `time_s=-1 ... success=False` on failure), `DONE`. A log can hold several `RUN_START` blocks if relaunched: parsers use the LAST block only |
| `summary.txt` | one line per group, `key=value` separated by ` \| ` | human-readable; same numbers as `summary.csv` |
| `summary.csv` | header + one row per group | what goes into tables/papers; failed runs counted in `runs`, not in the time mean |
| `statistics.txt` | `key = value` lines | totals over all runs: records, successes, failures/timeouts, best overall |
| `plots/*.png` | matplotlib, `Agg` backend, dpi 150 | axes labelled with units, log scale where values span decades, median curve with min-max band for repeated runs |
| `manifest.json` | JSON | what/when/where: experiment, run id, status (`running` until the script finishes -- a run that died stays `running`), start/end, host, python/numpy, code md5 and git commit, settings, BLAS threads, timer |
| `README.txt` | text | question the run answers, result in 2-3 lines, launch command, timeline (`START` / `END ... exit code`) |
| `run.sh` | bash | exports `RUN_DIR`, `RUN_ID`, BLAS pinned to 1 thread, optional `taskset`/conda env, runs the code copy, appends the timeline to the README |

Where files live in a run folder (`runs/<experiment>/<run_id>/`): `code/` (exact copy of the code), `output/` (jsonl, per-run folders),
`logs/` (run.log, console.log), `summary/`, `plots/`, plus `manifest.json`, `README.txt`, `run.sh`. Tracked in git: code, summary,
plots, manifest, README, run.sh. Ignored: `output/` and `logs/`.

Other styles already used in this repo (keep when extending those experiments): `report.md` + `stage1_table.{md,csv}` +
`grid1_matrix.csv` (DecapStudy*), `statistics.txt` / `timeouts.txt` / `top10_fastest.txt` (ScratchBench), per-run
`run_N/run_N.log` + `run_plot.png` + `global_convergence.png` (ScratchBench).
