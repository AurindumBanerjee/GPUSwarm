#!/usr/bin/env python3
"""AllCPU / CPUParallel -- 4 PDNs x 4 methods = 16 single-thread processes at a time, each pinned (taskset) to its OWN physical core, BLAS = 1 thread.
A job = one (PDN, method, swarm): it works through its bands x levels x runs sequentially (job_worker.py, resumable). Swarms are run in the order given
(default 50x15, 100x15, 50x40, 100x40, 200x40), warm start 20 % of the swarm, paired seeds BASE_SEED + run*10000 + n_caps for every method.
   python run_parallel.py --dry                 plan only (jobs, cores, nothing started)
   python run_parallel.py                       run everything (re-running resumes: finished runs are skipped)
   python run_parallel.py --swarms 50x15 --pdns pdn1,pdn2 --methods numpy,solve --bands B50 --runs 2 --cores 1,2,3,4
   --strict-waves    start the next swarm only when EVERY job of the previous one has finished (default: rolling queue -- a core freed by a short job
                     (PDN 1 finishes in hours, PDN 3/4 in days) immediately takes the next job, so the machine stays at 16 busy cores)
Cores: by default the first 8 physical cores of each socket after dropping core 0 of socket 0 (and its hyper-thread sibling); one logical CPU per physical core.
Guard: the first RUN_START line of every job must show the requested particles/iters and warm count; otherwise everything is stopped.
Writes: out/<pdn>/<swarm>/<method>/{results_long.jsonl,<band>/<level>/run_k/}, logs/<pdn>_<swarm>_<method>.log, manifest.jsonl, logs/PARALLEL_DONE."""
import os, sys, json, time, glob, signal, subprocess, datetime, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(H), "common"))
import allcpu_pdns as R
ap = argparse.ArgumentParser()
ap.add_argument("--swarms", default=",".join(R.SWARMS)); ap.add_argument("--pdns", default=",".join(R.PDN_KEYS)); ap.add_argument("--methods", default=",".join(R.METHODS))
ap.add_argument("--bands"); ap.add_argument("--levels"); ap.add_argument("--runs", type=int, default=int(os.environ.get("NUM_RUNS", 4))); ap.add_argument("--cores"); ap.add_argument("--out", default=os.path.join(H, "out_" + R.CFG))
ap.add_argument("--exclude", default="", help="logical CPUs to leave alone (e.g. the ones your training runs sit on)"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--strict-waves", action="store_true"); ap.add_argument("--max-parallel", type=int, default=16)
a = ap.parse_args()
LOGS = os.environ.get("ALLCPU_LOGDIR") or os.path.join(H, "logs"); MAN = os.path.join(LOGS if os.environ.get("ALLCPU_LOGDIR") else H, "manifest.jsonl"); os.makedirs(LOGS, exist_ok=True)
def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(): return [round(x, 2) for x in os.getloadavg()]
def put(o):
    with open(MAN, "a") as fh: fh.write(json.dumps(o) + "\n")


def topology():
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); socks = sorted({k[0] for k in keys}); s = [[phys[k][0] for k in keys if k[0] == sk] for sk in socks]
    assert len(socks) == 2 and all(len(x) >= 8 for x in s), f"expected 2 sockets with >= 8 usable physical cores each, found {[len(x) for x in s]}; pass --cores explicitly"
    return s[0][:8] + s[1][:8]


pdns, methods, swarms = a.pdns.split(","), a.methods.split(","), a.swarms.split(",")
bands = a.bands.split(",") if a.bands else None
miss = R.targets_status(pdns, bands)
if miss: print("targets not ready (run common/floor_pass.py first):", ", ".join(miss)); sys.exit(2)
try: cores = [int(c) for c in a.cores.split(",")] if a.cores else topology()
except Exception as e:
    if not a.dry: raise
    cores = []; print("[dry] core map unavailable here:", str(e)[:120])
excl = {int(c) for c in a.exclude.split(',') if c.strip()}; cores = [c for c in cores if c not in excl][:a.max_parallel]
jobs = []
for wi, sw in enumerate(swarms):
    for p in pdns:
        for m in methods: jobs.append(dict(wave=wi, swarm=sw, pdn=p, method=m, est=R.SPEC[p]["est_cost"] * R.METHOD_COST[m] * R.SWARM_COST[sw] * len(R.SPEC[p]["opt_bands"] if not bands else bands)))
jobs.sort(key=lambda j: (j["wave"], -j["est"]))
print(f"[{utc()}] AllCPU parallel: {len(jobs)} jobs ({len(pdns)} PDNs x {len(methods)} methods x {len(swarms)} swarms), {len(cores)} cores {cores}, runs/level {a.runs}, strict waves {a.strict_waves}")
for j in jobs[:40]: print(f"   wave {j['wave']} {j['swarm']:7} {j['pdn']} {j['method']:9} rel.cost {j['est']:8.1f}")
if a.dry: sys.exit(0)
if not cores: sys.exit("no cores")
children = []
def stop_all(*_):
    for p in children:
        if p.poll() is None: p.terminate()
    sys.exit(1)
signal.signal(signal.SIGTERM, stop_all); signal.signal(signal.SIGINT, stop_all)


def first_start(j):
    P, I = j["swarm"].split("x"); warm = int(round(0.2 * int(P)))
    for f in sorted(glob.glob(os.path.join(j["dir"], "**", "run_*.log"), recursive=True)):
        try:
            for l in open(f, errors="replace"):
                if "RUN_START" in l: return l.strip(), (f"particles={P} iters={I}" in l and f"warm={warm}/{P}" in l)
        except OSError: pass
    return None, None


free, running, queue = list(cores), [], list(jobs)
while queue or running:
    while queue and free:
        if a.strict_waves and any(r["wave"] < queue[0]["wave"] for r in running): break      # next swarm waits for every job of the previous one
        j = queue.pop(0); cpu = free.pop(0); j["dir"] = os.path.join(a.out, j["pdn"], j["swarm"], j["method"]); os.makedirs(j["dir"], exist_ok=True)
        cmd = ["taskset", "-c", str(cpu)] + R.job_cmd(j["pdn"], j["method"], j["swarm"], a.out, bands, a.runs, a.levels.split(",") if a.levels else None)
        log = open(os.path.join(LOGS, f"{j['pdn']}_{j['swarm']}_{j['method']}.log"), "a")
        j["proc"] = subprocess.Popen(cmd, env=R.thread_env(), stdout=log, stderr=subprocess.STDOUT); children.append(j["proc"])
        j.update(cpu=cpu, t0=time.time(), start=utc(), load0=load(), verified=False); running.append(j)
        put(dict(event="start", pdn=j["pdn"], swarm=j["swarm"], method=j["method"], cpu=cpu, pid=j["proc"].pid, start_utc=j["start"], loadavg_start=j["load0"]))
        print(f"[{j['start']}] start {j['pdn']} {j['swarm']} {j['method']} on cpu {cpu} (load {j['load0']})", flush=True)
    time.sleep(10)
    for j in list(running):
        if not j["verified"]:
            line, ok = first_start(j)
            if ok: j["verified"] = True; print(f"[{utc()}] {j['pdn']} {j['swarm']} {j['method']}: {line}", flush=True)
            elif ok is False or (line is None and j["proc"].poll() is not None) or time.time() - j["t0"] > 1800:
                print(f"[{utc()}] WRONG/MISSING RUN_START ({line!r}) -- stopping everything", flush=True)
                put(dict(event="abort", pdn=j["pdn"], swarm=j["swarm"], method=j["method"], run_start=line, utc=utc())); stop_all()
        rc = j["proc"].poll()
        if rc is not None:
            put(dict(event="end", pdn=j["pdn"], swarm=j["swarm"], method=j["method"], cpu=j["cpu"], start_utc=j["start"], end_utc=utc(), exit_code=rc, loadavg_start=j["load0"], loadavg_end=load(), wall_s=round(time.time() - j["t0"])))
            print(f"[{utc()}] done {j['pdn']} {j['swarm']} {j['method']} exit {rc} after {(time.time() - j['t0']) / 3600:.2f} h", flush=True)
            free.append(j["cpu"]); running.remove(j)
open(os.path.join(LOGS, "PARALLEL_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] all jobs finished", flush=True)
