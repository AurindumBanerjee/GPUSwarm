#!/usr/bin/env python3
"""Derives the floors of PDN 3 (MPHY, rail-wise: the five single-rail files) and PDN 4 (DDR3) for every band of the ACTIVE configuration
(ALLCPU_GRID / ALLCPU_LIB, default 5g_murata) from numpy full-budget runs (no target, patience stop) -- the DecapStudy2 rule:
floor = lowest peak reached (min over runs); if the spread (max-min)/min exceeds 10 % the floor is the MEDIAN.  Writes the floors into targets_allcpu.json.
   python floor_pass.py --plan                    list the jobs
   python floor_pass.py [--swarm 50x15] [--runs 3] [--procs 8] [--cores 1,2,...]      run them (resumable), then aggregate
   python floor_pass.py --aggregate [--dry-run]   only (re)aggregate
   python floor_pass.py --compare 1g_combined,1g_murata     table of the floors of two configurations already in the targets file (library pilot)
Library pilot (does the new Murata library improve on the old combined one?):
   ALLCPU_GRID=1g ALLCPU_LIB=combined python floor_pass.py ...   then   ALLCPU_GRID=1g ALLCPU_LIB=murata python floor_pass.py ...   then   --compare 1g_combined,1g_murata
Default swarm 50x15 (a lower swarm can only RAISE a floor, which makes the targets easier, never unreachable)."""
import os, sys, json, subprocess, time, statistics, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import allcpu_pdns as R
ap = argparse.ArgumentParser(); ap.add_argument("--plan", action="store_true"); ap.add_argument("--aggregate", action="store_true"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--compare"); ap.add_argument("--swarm", default="50x15"); ap.add_argument("--runs", type=int, default=3); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--cores")
ap.add_argument("--bands", default=",".join(R.BANDS_34)); ap.add_argument("--out", default=os.path.join(R.ALLCPU, "floor_pass", R.CFG))
a = ap.parse_args(); bands = a.bands.split(",")
if a.compare:
    T = R.targets(); c1, c2 = a.compare.split(","); print(f"floor comparison  {c1} (old)  vs  {c2} (new);  improvement = 1 - new/old  (positive = new library reaches lower impedance)")
    for b in R.BANDS_1G:
        try:
            for r in R.MPHY_RAILS:
                o, n = T["pdn3"][c1][b]["floors_ohm"][r], T["pdn3"][c2][b]["floors_ohm"][r]
                print(f"  pdn3 {b:5} {r:8} old {o:10.5g}  new {n:10.5g}  improvement {100 * (1 - n / o):6.1f} %")
            o, n = T["pdn4"][c1][b]["floor_ohm"], T["pdn4"][c2][b]["floor_ohm"]; print(f"  pdn4 {b:5} ddr      old {o:10.5g}  new {n:10.5g}  improvement {100 * (1 - n / o):6.1f} %")
        except (KeyError, TypeError): print(f"  {b}: floors missing in one of the configurations")
    sys.exit(0)
jobs = []
for b in bands:
    for rail in R.MPHY_RAILS + [None]:
        for k in range(1, a.runs + 1):
            pdn = "pdn3" if rail else "pdn4"; d = os.path.join(a.out, pdn, rail or "full", b, f"run{k}"); jobs.append(dict(pdn=pdn, rail=rail, band=b, run=k, out=d))
def finished(j): return os.path.exists(os.path.join(j["out"], "results_long.jsonl")) and os.path.getsize(os.path.join(j["out"], "results_long.jsonl")) > 0
if a.plan or not a.aggregate:
    print(f"configuration {R.CFG}: {len(jobs)} floor jobs ({len([j for j in jobs if finished(j)])} finished): bands {bands}, swarm {a.swarm}, runs {a.runs}")
    if a.plan: [print("  ", j["pdn"], j["rail"] or "ddr_full", j["band"], "run", j["run"]) for j in jobs]; sys.exit(0)
if not a.aggregate:
    cores = [int(c) for c in a.cores.split(",")] if a.cores else None; todo = [j for j in jobs if not finished(j)]; running = []; free = list(cores) if cores else [None] * a.procs
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    while todo or running:
        while todo and free and len(running) < a.procs:
            j = todo.pop(0); cpu = free.pop(0)
            cmd = [sys.executable, "-u", os.path.join(H, "job_worker.py"), "--floor", "--pdn", j["pdn"], "--band", j["band"], "--run", str(j["run"]), "--swarm", a.swarm, "--out", j["out"]]
            if j["rail"]: cmd += ["--rail", j["rail"]]
            if cpu is not None: cmd = ["taskset", "-c", str(cpu)] + cmd
            j["cpu"] = cpu; env = R.thread_env(); env["PATIENCE"] = os.environ.get("PATIENCE", "6")
            j["proc"] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(a.out, "logs", f"{j['pdn']}_{j['rail'] or 'full'}_{j['band']}_run{j['run']}.log"), "w"), stderr=subprocess.STDOUT); running.append(j)
            print("started", j["pdn"], j["rail"] or "ddr_full", j["band"], "run", j["run"], flush=True)
        time.sleep(5)
        for j in list(running):
            if j["proc"].poll() is not None: running.remove(j); free.append(j["cpu"]); print("exit", j["proc"].returncode, j["pdn"], j["rail"] or "ddr_full", j["band"], "run", j["run"], flush=True)
def best_costs(pdn, rail, b):
    v = []
    for k in range(1, a.runs + 1):
        f = os.path.join(a.out, pdn, rail or "full", b, f"run{k}", "results_long.jsonl")
        if os.path.exists(f):
            for l in open(f):
                if l.strip(): v.append(json.loads(l)["best_cost"])
    return v
def floor_of(v):
    if not v: return None, "no runs"
    spread = (max(v) - min(v)) / min(v)
    return (statistics.median(v), f"median of {len(v)} (spread {spread*100:.1f} % > 10 %)") if spread > 0.10 else (min(v), f"min of {len(v)} (spread {spread*100:.1f} %)")
T = R.targets(); changed = False
for b in bands:
    fl = {}
    for rail in R.MPHY_RAILS:
        f, why = floor_of(best_costs("pdn3", rail, b)); fl[rail] = f; print(f"{R.CFG} pdn3 {b:5} {rail:8} floor {f if f is None else format(f, '.6g'):>10}  {why}")
    f4, why = floor_of(best_costs("pdn4", None, b)); print(f"{R.CFG} pdn4 {b:5} ddr_full floor {f4 if f4 is None else format(f4, '.6g'):>10}  {why}")
    src = f"floor_pass.py swarm {a.swarm} runs {a.runs}"
    if all(v is not None for v in fl.values()): T["pdn3"][R.CFG][b]["floors_ohm"] = fl; T["pdn3"][R.CFG][b]["floor_source"] = src; changed = True
    if f4 is not None: T["pdn4"][R.CFG][b]["floor_ohm"] = f4; T["pdn4"][R.CFG][b]["floor_source"] = src; changed = True
if changed and not a.dry_run:
    json.dump(T, open(R.TARGETS_FILE, "w"), indent=1); print("targets_allcpu.json updated")
