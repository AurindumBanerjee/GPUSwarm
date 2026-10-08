"""
BASE TEMPLATE for a GPUSwarm experiment.  Copy it, rename it, replace run_one().  Do not edit a copy that has already produced results:
make a new run instead (reference/new_run.py), so every output folder keeps the exact code that made it.

What it gives you (see reference/output_style/ for real samples of every file):

    <RUN_DIR>/
        code/<this script>              copy of the code that ran (md5 in manifest.json)
        output/results_long.jsonl       one JSON record per run (all factor levels as fields)   [machine-readable, append-only]
        logs/run.log                    RUN_START / STAGE / RESULT lines                          [human + parser friendly]
        summary/summary.txt             one line per group, same fields as the log RESULT line    [human]
        summary/summary.csv             the same table                                            [spreadsheets / pandas]
        summary/statistics.txt          key = value statistics over all runs                      [human]
        plots/convergence.png           figure 1 (median curve + min/max band)
        plots/runs.png                  figure 2 (per-run result)
        manifest.json                   what/when/where: code md5, git commit, env, settings, start/end, status
        README.txt                      created if absent

Settings come from the environment (so a launcher can set them and the manifest records them):
    RUN_DIR      run folder (default: ./run_local next to this script); new_run.py sets it to runs/<experiment>/<run_id>
    RUN_ID       label stored in the manifest
    NUM_RUNS     repetitions (default 5)        BASE_SEED  (default 12345; run r uses BASE_SEED + r, so runs are paired)
    THREADS      particle-level worker threads (default 1; BLAS stays pinned to 1 thread -- see below)

Conventions worth keeping:
  * BLAS/OMP threads are pinned to 1 BEFORE numpy is imported (oversubscription made one study 30x slower).
  * time.perf_counter() for timing; record it in the manifest.
  * Seeds are explicit and paired across arms, so differences between arms are the thing under test.
  * Failed runs are logged with time = -1 and success = False and stay in the table (never silently dropped).
  * Exit status is recorded: a run that dies leaves manifest.json with status "running" -- that is the tell.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import hashlib
import json
import logging
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "template_toy"                       # <- name of the experiment (matches runs/<experiment>/)
RUN_DIR = os.path.abspath(os.environ.get("RUN_DIR", os.path.join(HERE, "run_local")))
RUN_ID = os.environ.get("RUN_ID", "run_local")
NUM_RUNS = int(os.environ.get("NUM_RUNS", "5"))
BASE_SEED = int(os.environ.get("BASE_SEED", "12345"))
THREADS = int(os.environ.get("THREADS", "1"))

# ---- experiment settings (a real experiment keeps its knobs here, as plain top-level assignments, so new_run.py can set them)
DIM = 6                       # toy problem: minimise sum((x - 0.3)^2) over [0, 1]^DIM
N_PARTICLES = 20
N_ITERATIONS = 30
THRESHOLDS = [0.05, 0.02, 0.01]      # stop at the first iteration whose best value is below a threshold


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def md5_file(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def setup_run_dir():
    for d in ("code", "output", "logs", "summary", "plots"):
        os.makedirs(os.path.join(RUN_DIR, d), exist_ok=True)
    me = os.path.abspath(__file__)
    dst = os.path.join(RUN_DIR, "code", os.path.basename(me))
    if not os.path.exists(dst):                       # new_run.py has already put the exact copy there; ad-hoc runs get it here
        shutil.copy2(me, dst)
    log = logging.getLogger(EXPERIMENT)
    log.setLevel(logging.INFO)
    h = logging.FileHandler(os.path.join(RUN_DIR, "logs", "run.log"))
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(h)
    log.addHandler(logging.StreamHandler(sys.stdout))
    return log, dst


def git_commit():
    try:
        out = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", HERE, "status", "--porcelain", "--", HERE], stderr=subprocess.DEVNULL).strip())
        return out + ("+dirty" if dirty else "")
    except Exception:                                  # noqa: BLE001
        return None


def write_manifest(code_path, status, start, end=None):
    m = {"experiment": EXPERIMENT, "run_id": RUN_ID, "status": status, "start": iso(start), "end": iso(end) if end else None,
         "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
         "numpy": np.__version__, "cpu_count": os.cpu_count(), "timer": "time.perf_counter",
         "code": {"file": os.path.relpath(code_path, RUN_DIR).replace(os.sep, "/"), "md5": md5_file(code_path), "git": git_commit()},
         "settings": {"NUM_RUNS": NUM_RUNS, "BASE_SEED": BASE_SEED, "THREADS": THREADS, "DIM": DIM, "N_PARTICLES": N_PARTICLES,
                      "N_ITERATIONS": N_ITERATIONS, "THRESHOLDS": THRESHOLDS},
         "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    json.dump(m, open(os.path.join(RUN_DIR, "manifest.json"), "w"), indent=2)


# ------------------------------------------------------------------------------------------------ the experiment
def run_one(run_id, threshold, log):
    """REPLACE ME. Do one seeded run and return a JSON-serialisable record. Toy PSO on a quadratic bowl."""
    rng = np.random.default_rng(BASE_SEED + run_id)              # paired seeds across thresholds / arms
    x = rng.random((N_PARTICLES, DIM))
    v = np.zeros_like(x)
    pbest, pval = x.copy(), ((x - 0.3) ** 2).sum(1)
    gbest, gval = pbest[pval.argmin()].copy(), pval.min()
    curve, t0 = [], time.perf_counter()
    for it in range(N_ITERATIONS):
        w = 0.9 - 0.5 * it / N_ITERATIONS
        v = w * v + 1.5 * rng.random(x.shape) * (pbest - x) + 1.5 * rng.random(x.shape) * (gbest - x)
        x = np.clip(x + v, 0, 1)
        val = ((x - 0.3) ** 2).sum(1)
        better = val < pval
        pbest[better], pval[better] = x[better], val[better]
        if pval.min() < gval:
            gval, gbest = pval.min(), pbest[pval.argmin()].copy()
        curve.append(float(gval))
        if it % 10 == 9:
            log.info("STAGE | threshold=%g | run=%d | iter=%d | best=%.6f", threshold, run_id, it + 1, gval)
        if gval < threshold:
            break
    t = time.perf_counter() - t0
    ok = bool(gval < threshold)
    return {"threshold": threshold, "run": run_id, "seed": BASE_SEED + run_id, "success": ok,
            "time_s": t if ok else -1.0, "iterations": len(curve), "best": float(gval), "curve": curve}


def summarise(records):
    groups = {}
    for r in records:
        groups.setdefault(r["threshold"], []).append(r)
    rows = []
    for thr, rs in sorted(groups.items(), reverse=True):
        ok = [r for r in rs if r["success"]]
        rows.append({"threshold": thr, "runs": len(rs), "successes": len(ok), "success_rate": len(ok) / len(rs),
                     "time_mean_s": statistics.mean(r["time_s"] for r in ok) if ok else None,
                     "iterations_mean": statistics.mean(r["iterations"] for r in ok) if ok else None,
                     "best_min": min(r["best"] for r in rs), "best_mean": statistics.mean(r["best"] for r in rs)})
    with open(os.path.join(RUN_DIR, "summary", "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RUN_DIR, "summary", "summary.txt"), "w") as fh:
        for r in rows:
            fh.write(" | ".join(f"{k}={('-' if v is None else format(v, '.6g'))}" for k, v in r.items()) + "\n")
    with open(os.path.join(RUN_DIR, "summary", "statistics.txt"), "w") as fh:
        fh.write(f"records = {len(records)}\nsuccesses = {sum(r['success'] for r in records)}\n")
        fh.write(f"failures_or_timeouts = {sum(not r['success'] for r in records)}\n")
        fh.write(f"best_overall = {min(r['best'] for r in records):.6g}\n")
    return rows


def plots(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for thr in sorted({r["threshold"] for r in records}, reverse=True)[-1:]:
        cs = [r["curve"] for r in records if r["threshold"] == thr]
        n = max(map(len, cs))
        arr = np.array([c + [c[-1]] * (n - len(c)) for c in cs])
        x = np.arange(1, n + 1)
        ax.plot(x, np.median(arr, 0), label=f"median, threshold {thr:g}")
        ax.fill_between(x, arr.min(0), arr.max(0), alpha=0.25)
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("best value"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RUN_DIR, "plots", "convergence.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for thr in sorted({r["threshold"] for r in records}, reverse=True):
        rs = [r for r in records if r["threshold"] == thr]
        ax.plot([r["run"] for r in rs], [r["best"] for r in rs], "o-", label=f"threshold {thr:g}")
    ax.set_xlabel("run"); ax.set_ylabel("best value reached"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RUN_DIR, "plots", "runs.png"), dpi=150); plt.close(fig)


def main():
    start = time.time()
    log, code_path = setup_run_dir()
    write_manifest(code_path, "running", start)
    jsonl = os.path.join(RUN_DIR, "output", "results_long.jsonl")
    open(jsonl, "w").close()                                   # a run folder holds ONE run: start the file empty
    records = []
    for thr in THRESHOLDS:
        for run_id in range(1, NUM_RUNS + 1):
            log.info("RUN_START | experiment=%s | threshold=%.6f | run=%d | seed=%d | particles=%d | iterations=%d",
                     EXPERIMENT, thr, run_id, BASE_SEED + run_id, N_PARTICLES, N_ITERATIONS)
            try:
                rec = run_one(run_id, thr, log)
            except Exception as e:                              # noqa: BLE001  (a failed run is data, not a crash)
                log.error("RESULT | time_s=-1 | best=nan | success=False | error=%r", e)
                continue
            log.info("RESULT | time_s=%.4f | iterations=%d | best=%.6f | success=%s", rec["time_s"], rec["iterations"], rec["best"], rec["success"])
            with open(jsonl, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            records.append(rec)
    summarise(records)
    plots(records)
    readme = os.path.join(RUN_DIR, "README.txt")
    if not os.path.exists(readme):
        open(readme, "w").write(f"RUN {EXPERIMENT}/{RUN_ID}\nDescribe what this run is for, the question it answers and the result in 2-3 lines.\n")
    write_manifest(code_path, "finished", start, time.time())
    log.info("DONE | outputs under %s", RUN_DIR)


if __name__ == "__main__":
    main()
