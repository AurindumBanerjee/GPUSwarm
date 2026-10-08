#!/usr/bin/env python
"""
new_run.py -- create an isolated, documented run folder for a GPUSwarm experiment.

Every run gets ITS OWN copy of the code, its own output tree, a launcher, a manifest and a README; runs/INDEX.md lists them.
Existing runs are never touched or overwritten, and a copy of a script is never edited after it has produced results:
change something -> make a new run (new id, new code copy).

    runs/<experiment>/<run_id>/
        code/<stem>__<run_id>.py     the script as run = source copy + ONLY the --set lines
        output/  logs/               raw results and logs   (git-ignored)
        summary/ plots/              tables and figures      (tracked)
        run.sh  manifest.json  README.txt

The script finds its run folder through the environment (see reference/template_experiment.py): RUN_DIR, RUN_ID, plus any --env.

Example:
    python reference/new_run.py --experiment warmstart --id run02_jitter0 --desc "naive warm start, 10 runs" \\
        --source Comparisons/warmstart_thresholds.py --set 'WARM_JITTER = 0.0' --env NUM_RUNS=10 --env THREADS=16
    bash runs/warmstart/run02_jitter0/run.sh            # or: nohup setsid bash .../run.sh > .../logs/driver.log 2>&1 &

Options: --args 'CLI ARGS'  --taskset 16-23  --conda-env work  --python python  --dry-run
Legacy scripts that write to a fixed folder next to themselves cannot be isolated this way: make them read RUN_DIR first.
"""
import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def md5_text(s):
    return hashlib.md5(s.encode()).hexdigest()


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *a], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        return None


def set_line(text, assignment, where):
    key = assignment.split("=", 1)[0].strip()
    pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    n = len(pat.findall(text))
    if n != 1:
        sys.exit(f"{where}: expected exactly one top-level '{key} = ...' line, found {n}")
    return pat.sub(lambda m: assignment.strip(), text, count=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="folder under runs/, e.g. pso_sp, comparisons")
    ap.add_argument("--id", required=True, help="run id, e.g. run03_threshold_levels (never reused)")
    ap.add_argument("--desc", required=True)
    ap.add_argument("--source", required=True, help="script to copy (repo-relative or absolute)")
    ap.add_argument("--set", action="append", default=[], help="top-level assignment to replace, e.g. 'NUM_RUNS = 20' (repeatable)")
    ap.add_argument("--args", default="", help="command-line arguments for the script")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE exported by run.sh (repeatable)")
    ap.add_argument("--taskset", default=None, help="CPU list for taskset -c")
    ap.add_argument("--conda-env", default=None, help="conda env to activate in run.sh")
    ap.add_argument("--python", default="python")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.source if os.path.isabs(a.source) else os.path.join(ROOT, a.source)
    if not os.path.isfile(src):
        sys.exit(f"source not found: {src}")
    run = os.path.join(RUNS, a.experiment, a.id)
    if os.path.exists(run):
        sys.exit(f"{run} already exists -- runs are never overwritten; choose a new --id")

    text = open(src, newline="").read().replace("\r\n", "\n")
    new = text
    for s in a.set:
        new = set_line(new, s, a.source)
    stem = os.path.splitext(os.path.basename(src))[0]
    code_name = f"{stem}__{a.id}.py"
    diff = "".join(difflib.unified_diff(text.splitlines(1), new.splitlines(1), os.path.basename(src), code_name, n=0)) or "(identical to source)"
    if a.dry_run:
        print(f"would create {run}/code/{code_name}\n{diff}")
        return

    for d in ("code", "output", "logs", "summary", "plots"):
        os.makedirs(os.path.join(run, d))
    code_path = os.path.join(run, "code", code_name)
    open(code_path, "w", newline="").write(new)

    envs = " ".join(a.env)
    sh = ["#!/bin/bash",
          f"# Run {a.experiment}/{a.id}. Launch from anywhere; the script runs from the repository root.",
          f'RUN="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"', f'cd "$RUN/../../.." || exit 1',
          *( ["source $(conda info --base 2>/dev/null || echo /DATA/Aurindum/conda)/etc/profile.d/conda.sh", f"conda activate {a.conda_env}"]
             if a.conda_env else []),
          "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
          f'export RUN_DIR="$RUN" RUN_ID="{a.id}"' + (f" {envs}" if envs else ""),
          "stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }",
          'echo "START: $(stamp)  host: $(hostname)  load: $(cut -d" " -f1-3 /proc/loadavg 2>/dev/null)" >> "$RUN/README.txt"',
          (f"taskset -c {a.taskset} " if a.taskset else "") + f'{a.python} -u "$RUN/code/{code_name}" {a.args} > "$RUN/logs/console.log" 2>&1',
          'echo "END:   $(stamp)  exit code $?" >> "$RUN/README.txt"', ""]
    open(os.path.join(run, "run.sh"), "w", newline="").write("\n".join(sh))
    os.chmod(os.path.join(run, "run.sh"), 0o755)

    commit, dirty = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", a.source))
    manifest = {"experiment": a.experiment, "run_id": a.id, "description": a.desc, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source": {"path": a.source, "md5": md5_text(text), "git": commit, "modified_in_working_tree": dirty},
                "code": {"file": f"code/{code_name}", "md5": md5_text(new), "set_lines": a.set},
                "execution": {"args": a.args, "env": a.env, "taskset": a.taskset, "conda_env": a.conda_env, "python": a.python},
                "host_at_creation": platform.node()}
    json.dump(manifest, open(os.path.join(run, "manifest.json"), "w"), indent=2)
    with open(os.path.join(run, "README.txt"), "w") as fh:
        fh.write(f"RUN {a.experiment}/{a.id}\n{a.desc}\n\nCreated: {manifest['created']}  on {platform.node()}\n"
                 f"Source: {a.source}  (git {commit}{', MODIFIED in working tree' if dirty else ''}, md5 {md5_text(text)})\n"
                 f"As run:  code/{code_name}  (md5 {md5_text(new)})\n\nChanges vs source (only these lines differ):\n{diff}\n"
                 f"Execution: args '{a.args}'; env {a.env or 'none'}; taskset {a.taskset or 'none'}; conda env {a.conda_env or 'none'}\n"
                 f"\nLaunch: nohup setsid bash {os.path.relpath(os.path.join(run, 'run.sh'), ROOT)} > {os.path.relpath(os.path.join(run, 'logs', 'driver.log'), ROOT)} 2>&1 < /dev/null &\n"
                 "\n---- timeline ----\n")
    idx = os.path.join(RUNS, "INDEX.md")
    if not os.path.exists(idx):
        open(idx, "w").write("# GPUSwarm runs\n\n| run | status | start | what | code |\n|---|---|---|---|---|\n")
    with open(idx, "a") as fh:
        fh.write(f"| {a.experiment}/{a.id} | created | {time.strftime('%Y-%m-%d')} | {a.desc[:140]} | {commit} {os.path.basename(a.source)} |\n")
    print(f"created {run}\n  code/{code_name}  md5 {md5_text(new)}\n{diff}\nlaunch: bash {os.path.relpath(os.path.join(run, 'run.sh'), ROOT)}")


if __name__ == "__main__":
    main()
