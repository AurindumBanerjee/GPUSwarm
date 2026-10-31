#!/usr/bin/env python3
"""AllCPU / CPUSerial -- ONE process at a time, ONE thread: PDN by PDN, and inside a PDN swarm by swarm, method by method (numpy, solve, sm, iterative), each running all
its bands x levels x runs before the next starts.  Nothing else runs concurrently, so these timings are the clean (idle-machine) reference for the CPUParallel ones.
   python run_serial.py --dry
   python run_serial.py                             all four PDNs, both swarms (resumable)
   python run_serial.py --pdns pdn1 --swarms 50x15 --methods numpy,solve --core 3          a single piece, pinned to logical CPU 3
Order: PDN -> swarm -> method.  Same worker, seeds, warm start (20 %), library and targets as CPUParallel.
Writes: out/<pdn>/<swarm>/<method>/..., logs/<pdn>_<swarm>_<method>.log, manifest.jsonl, logs/SERIAL_DONE."""
import os, sys, json, time, subprocess, datetime, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(H), "common"))
import allcpu_pdns as R
ap = argparse.ArgumentParser()
ap.add_argument("--swarms", default=",".join(R.SWARMS)); ap.add_argument("--pdns", default=",".join(R.PDN_KEYS)); ap.add_argument("--methods", default=",".join(R.METHODS))
ap.add_argument("--bands"); ap.add_argument("--levels"); ap.add_argument("--runs", type=int, default=int(os.environ.get("NUM_RUNS", 4))); ap.add_argument("--core", type=int, help="pin to this logical CPU (taskset)")
ap.add_argument("--out", default=os.path.join(H, "out_" + R.CFG)); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
LOGS = os.environ.get("ALLCPU_LOGDIR") or os.path.join(H, "logs"); MAN = os.path.join(LOGS if os.environ.get("ALLCPU_LOGDIR") else H, "manifest.jsonl"); os.makedirs(LOGS, exist_ok=True)
def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def put(o):
    with open(MAN, "a") as fh: fh.write(json.dumps(o) + "\n")
pdns, methods, swarms = a.pdns.split(","), a.methods.split(","), a.swarms.split(","); bands = a.bands.split(",") if a.bands else None
miss = R.targets_status(pdns, bands)
if miss: print("targets not ready (run common/floor_pass.py first):", ", ".join(miss)); sys.exit(2)
jobs = [(p, sw, m) for p in pdns for sw in swarms for m in methods]
print(f"[{utc()}] AllCPU serial: {len(jobs)} jobs, one at a time, single thread" + (f", cpu {a.core}" if a.core is not None else ""))
for p, sw, m in jobs: print(f"   {p} {sw:7} {m}")
if a.dry: sys.exit(0)
for p, sw, m in jobs:
    cmd = R.job_cmd(p, m, sw, a.out, bands, a.runs, a.levels.split(",") if a.levels else None)
    if a.core is not None: cmd = ["taskset", "-c", str(a.core)] + cmd
    t0 = time.time(); put(dict(event="start", pdn=p, swarm=sw, method=m, start_utc=utc(), loadavg_start=[round(x, 2) for x in os.getloadavg()]))
    print(f"[{utc()}] start {p} {sw} {m}", flush=True)
    rc = subprocess.call(cmd, env=R.thread_env(), stdout=open(os.path.join(LOGS, f"{p}_{sw}_{m}.log"), "a"), stderr=subprocess.STDOUT)
    put(dict(event="end", pdn=p, swarm=sw, method=m, end_utc=utc(), exit_code=rc, wall_s=round(time.time() - t0)))
    print(f"[{utc()}] done {p} {sw} {m} exit {rc} after {(time.time() - t0) / 3600:.2f} h", flush=True)
open(os.path.join(LOGS, "SERIAL_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] serial run finished")
