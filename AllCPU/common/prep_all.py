#!/usr/bin/env python3
"""AllCPU preparation + tests, resumable, for the server (run with nohup, see run_prep.sh).  Everything it writes goes to AllCPU/data_built, AllCPU/floor_pass and AllCPU/prep.
Stages (each leaves a marker in prep/markers; re-running skips finished stages):
  boards_5g          build_data.py --grid 5g --part boards       MPHY + DDR3 boards to 5 GHz from the Touchstone files
  lib_5g, lib_1g     build_data.py --part library                Murata RLC library on the 5 GHz grid and on the existing 1 GHz grid
  validate_<cfg>     validate.py (selftests, bands, smoke PSO)   for 1g_combined, 1g_murata, 5g_murata  (rc 3 = floors not derived yet = fine here)
  bench_<cfg>        bench_cost.py                               seconds per evaluation and hours per capacitor count for every swarm
  smoke_parallel     run_parallel.py on PDN 1 + PDN 2, swarm 20x2, 1 run, 8 pinned cores   -> checks pinning, guard, resume
  pilot_<cfg>        floor_pass.py on 1g_combined and 1g_murata (B20,B50,B200,B1G)       -> does the new library lower the floors?
  pilot_compare      floor_pass.py --compare 1g_combined,1g_murata
  floors_5g_murata   floor_pass.py for the 5g_murata configuration, all five bands (skipped with --stop-after-pilot)
  validate_final     validate.py for 5g_murata, expected rc 0 (all targets ready)
After every stage prep/prep_report.txt is rewritten -- that file is what to send back.
   nohup python prep_all.py > prep.out 2>&1 &        (or ./run_prep.sh)"""
import os, sys, subprocess, time, argparse, datetime, re
H = os.path.dirname(os.path.abspath(__file__)); ALLCPU = os.path.dirname(H); sys.path.insert(0, H)
ap = argparse.ArgumentParser()
ap.add_argument("--cores", default=os.environ.get("CORES", "")); ap.add_argument("--exclude", default=os.environ.get("EXCLUDE", ""), help="logical CPUs to avoid (e.g. your training runs)")
ap.add_argument("--procs", type=int, default=8); ap.add_argument("--pilot-runs", type=int, default=3); ap.add_argument("--pilot-swarm", default="50x15"); ap.add_argument("--floor-runs", type=int, default=3); ap.add_argument("--floor-swarm", default="50x15")
ap.add_argument("--dataset", default=os.environ.get("ALLCPU_DATASET", "/DATA/Aurindum/Swarming/Dataset")); ap.add_argument("--stop-after-pilot", action="store_true")
ap.add_argument("--only", help="comma list of stages to run"); ap.add_argument("--force", action="store_true", help="ignore markers")
a = ap.parse_args()
PREP = os.path.join(ALLCPU, "prep"); MK = os.path.join(PREP, "markers"); LG = os.path.join(PREP, "logs"); os.makedirs(MK, exist_ok=True); os.makedirs(LG, exist_ok=True)
os.environ.setdefault("GPUSWARM_ROOT", os.path.dirname(ALLCPU))
def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
CFGS = {"1g_combined": ("1g", "combined"), "1g_murata": ("1g", "murata"), "5g_murata": ("5g", "murata")}


def physical_cores():
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); socks = sorted({k[0] for k in keys}); s = [[phys[k][0] for k in keys if k[0] == sk] for sk in socks]
    return [c for sk in s for c in sk]
excl = {int(c) for c in a.exclude.split(",") if c.strip()}
if a.cores: CORES = [int(c) for c in a.cores.split(",") if c.strip()]
else:
    pc = [c for c in physical_cores() if c not in excl]; CORES = pc[:a.procs]
CORES = [c for c in CORES if c not in excl][:max(a.procs, 1)]
CORESTR = ",".join(map(str, CORES))


def env_for(cfg=None, name=None):
    e = dict(os.environ, ALLCPU_DATASET=a.dataset, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", STUDY_THREADS="1")
    if cfg: e["ALLCPU_GRID"], e["ALLCPU_LIB"] = CFGS[cfg]
    if name == "smoke_parallel": e["ALLCPU_LOGDIR"] = os.path.join(PREP, "smoke_parallel_logs")
    return e


STAGES = []
def stage(name, cmd, cfg=None, ok=(0,), pin=False):
    STAGES.append(dict(name=name, cmd=cmd, cfg=cfg, ok=ok, pin=pin))
P = lambda s: os.path.join(H, s)
PY = sys.executable
stage("boards_5g", [PY, P("build_data.py"), "--grid", "5g", "--part", "boards", "--dataset", a.dataset])
stage("lib_5g", [PY, P("build_data.py"), "--grid", "5g", "--part", "library", "--dataset", a.dataset])
stage("lib_1g", [PY, P("build_data.py"), "--grid", "1g", "--part", "library", "--dataset", a.dataset])
for c in CFGS:
    stage(f"validate_{c}", [PY, P("validate.py")], cfg=c, ok=(0, 3), pin=True)
for c in CFGS:
    stage(f"bench_{c}", [PY, P("bench_cost.py")], cfg=c, pin=True)
stage("smoke_parallel", [PY, os.path.join(ALLCPU, "CPUParallel", "run_parallel.py"), "--pdns", "pdn1,pdn2", "--swarms", "20x2", "--methods", "numpy,solve,sm,iterative", "--runs", "1", "--levels", "T3",
                         "--out", os.path.join(PREP, "smoke_parallel_out"), "--cores", CORESTR])
for c in ("1g_combined", "1g_murata"):
    stage(f"pilot_{c}", [PY, P("floor_pass.py"), "--bands", "B20,B50,B200,B1G", "--runs", str(a.pilot_runs), "--swarm", a.pilot_swarm, "--procs", str(a.procs), "--cores", CORESTR], cfg=c)
stage("pilot_compare", [PY, P("floor_pass.py"), "--compare", "1g_combined,1g_murata"], cfg="1g_murata")
if not a.stop_after_pilot:
    stage("floors_5g_murata", [PY, P("floor_pass.py"), "--runs", str(a.floor_runs), "--swarm", a.floor_swarm, "--procs", str(a.procs), "--cores", CORESTR], cfg="5g_murata")
    stage("validate_final", [PY, P("validate.py")], cfg="5g_murata", ok=(0,), pin=True)


def report(status):
    L = [f"AllCPU preparation report   {utc()}   cores {CORESTR}   GPUSWARM_ROOT {os.environ['GPUSWARM_ROOT']}", ""]
    L.append("stage status: " + "  ".join(f"{n}={s}" for n, s in status.items())); L.append("")
    def tail(name, pat=None, n=40):
        f = os.path.join(LG, name + ".log")
        if not os.path.exists(f): return
        lines = open(f, errors="replace").read().splitlines()
        if pat: lines = [l for l in lines if re.search(pat, l)]
        L.append(f"--- {name} ---"); L.extend(lines[-n:]); L.append("")
    for n in ("boards_5g", "lib_5g", "lib_1g"): tail(n, n=12)
    for c in CFGS: tail(f"validate_{c}", r"===|N=|band|targets|s/eval|smoke|validate:|Error|Traceback", 60)
    for c in CFGS: tail(f"bench_{c}", n=30)
    tail("smoke_parallel", r"start|RUN_START|done|WRONG|finished|Error", 30)
    tail("pilot_compare", n=60)
    tail("floors_5g_murata", r"floor|updated|Error", 80); tail("validate_final", r"===|targets|validate:|Error", 40)
    open(os.path.join(PREP, "prep_report.txt"), "w").write("\n".join(L) + "\n")


status = {s["name"]: ("done" if os.path.exists(os.path.join(MK, s["name"])) and not a.force else "pending") for s in STAGES}
only = set(a.only.split(",")) if a.only else None
print(f"[{utc()}] AllCPU prep: {len(STAGES)} stages, cores {CORESTR}", flush=True)
for s in STAGES:
    n = s["name"]
    if (only and n not in only) or status[n] == "done": print(f"[{utc()}] skip {n} ({status[n]})", flush=True); continue
    cmd = (["taskset", "-c", CORES[0] and str(CORES[0]) or "0"] + s["cmd"]) if (s["pin"] and CORES) else s["cmd"]
    print(f"[{utc()}] start {n}", flush=True); status[n] = "running"; report(status); t0 = time.time()
    rc = subprocess.call(cmd, env=env_for(s["cfg"], n), stdout=open(os.path.join(LG, n + ".log"), "w"), stderr=subprocess.STDOUT)
    if rc in s["ok"]:
        status[n] = "done" if rc == 0 else f"done(rc={rc})"; open(os.path.join(MK, n), "w").write(utc() + "\n")
    else: status[n] = f"FAILED(rc={rc})"
    print(f"[{utc()}] {n}: {status[n]} after {(time.time() - t0) / 60:.1f} min", flush=True); report(status)
    if status[n].startswith("FAILED"): print("stopping: see", os.path.join(LG, n + ".log")); sys.exit(1)
report(status); print(f"[{utc()}] prep finished; report: {os.path.join(PREP, 'prep_report.txt')}", flush=True)
