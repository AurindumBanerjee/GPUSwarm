#!/usr/bin/env python3
"""Step 3B. pure_python baseline, independent of run_methods.py (never waits for it, has its own scheduler and manifest).
28 jobs, one job = (pdn, swarm, target, run):  28-port 4 targets x 2 swarms x 2 runs = 16,  21-port 3 targets x 2 swarms x 2 runs = 12.
16 slots, each pinned (taskset) to its own PHYSICAL core of pool P (disjoint from pool M of run_methods.py), longest job first, a slot is refilled as soon as
its job exits. One OUT dir and one results_long.jsonl per job. A timed-out run is recorded (TIME_LIMIT_S), never retried.
   python run_pure.py                         # scheduler
   python run_pure.py --smoke                 # 20 s dry run with 16 dummy jobs
   python run_pure.py --job <pdn> <PxI> <target> <run>      # worker: ONE job in this process (needs OUT in the environment; used by the scheduler and short_baseline.py)
Pausing: while logs/PAUSE_PURE exists no new job is launched (idle_reference.py sends SIGSTOP/SIGCONT to the running ones; pids in logs/pure_pids.json)."""
import os, sys, json, time, glob, signal, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out"); LOGS = os.path.join(H, "logs"); MAN = os.path.join(H, "manifest_pure.jsonl")
TIME_LIMIT_S = os.environ.get("TIME_LIMIT_S", "604800"); SWARMS = [(100, 40), (50, 15)]
PDNS = {"pcb28full": dict(targets=[120.0, 125.0, 130.0, 140.0], mod="pso_pcb28_pure", dir=os.path.join(ROOT, "PSO_PCB28", "baseline_pure_python"), s_eval=11.9,
                          caps={120.0: 6, 125.0: 5, 130.0: 4.5, 140.0: 2.5}),
        "ref21": dict(targets=[0.045, 0.05, 0.06], mod="pso_ref21_pure", dir=os.path.join(ROOT, "PSO_REF21", "baseline_pure_python"), s_eval=2.7,
                      caps={0.045: 7.4, 0.05: 5.4, 0.06: 3.4})}
RUNS = [1, 2]


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(): return [round(x, 2) for x in os.getloadavg()]
def put(obj):
    with open(MAN, "a") as fh: fh.write(json.dumps(obj) + "\n")
def tag(p, i): return f"{p}x{i}"


def pools():
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0])
    socks = sorted({k[0] for k in keys}); assert len(socks) == 2, "expected 2 sockets"
    s = [[phys[k][0] for k in keys if k[0] == sk] for sk in socks]
    M = s[0][:4] + s[1][:4]; P = s[0][4:12] + s[1][4:12]
    cmap = {cpu: {"socket": k[0], "core": k[1], "siblings": phys[k]} for k in phys for cpu in phys[k][:1]}
    assert not set(M) & set(P) and len(set(M + P)) == 24
    return M, P, cmap


def threads_env():
    e = dict(os.environ)
    for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): e[k + "_NUM_THREADS"] = "1"
    e["STUDY_THREADS"] = "1"; return e


# ----------------------------------------------------------------------------------------------- worker: one (pdn, swarm, target, run)
def worker(pdn, sw, target, run):
    P, I = (int(x) for x in sw.split("x")); out = os.environ["OUT"]; target = float(target); run = int(run)
    os.environ.update(N_PARTICLES=str(P), N_ITERATIONS=str(I), TIME_LIMIT_S=os.environ.get("TIME_LIMIT_S", TIME_LIMIT_S))
    os.environ.update(dict(NET="full", MAX_CAPS="14") if pdn == "pcb28full" else dict(LIB="ref"))       # LIB must NOT be set for the 28-port drivers (default murata)
    cfg = PDNS[pdn]; sys.path.insert(0, cfg["dir"]); d = __import__(cfg["mod"]); core = d.core
    os.makedirs(out, exist_ok=True)
    if pdn == "pcb28full":
        core.library(); prob = core.Problem(**d.PROB)
        lvl = next(k for k, v in d.T.items() if abs(v - target) < 1e-9); thr = d.T[lvl]; res = os.path.join(out, "results_long.jsonl")
        r = core.run_pso(prob, "pure_python", run, os.path.join(out, prob.name, "pure_python", lvl, f"run_{run}"), res, out_root=out, threshold=thr, level=lvl, max_caps=d.MAX_CAPS)
        band = "B3G"
    else:
        d.base.load("ref"); prob = core.Problem("ref21", os.path.join(d.base.DATA, "y2.mat"), n_obs=1, obs_labels=["port0"])
        lvl = next(k for k, v in d.LEVELS.items() if abs(v - target) < 1e-12); thr = d.LEVELS[lvl]; o2 = os.path.join(out, "ref"); os.makedirs(o2, exist_ok=True)
        r = core.run_pso(prob, "pure_python", run, os.path.join(o2, prob.name, "pure_python", lvl, f"run_{run}"), os.path.join(o2, "results_long.jsonl"),
                         out_root=o2, threshold=thr, level=lvl)
        band = "BFULL"
    print(f"RESULT {pdn} {sw} {lvl}={thr:g} run {run}: best {r['best_peak_ohm'][band]:.6g} @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s, "
          f"success={r['success']} timed_out={r['timed_out']} evals={r['n_fitness_evals']}", flush=True)


# ----------------------------------------------------------------------------------------------- scheduler
def job_list():
    jobs = []
    for pdn, c in PDNS.items():
        for (P, I) in SWARMS:
            for t in c["targets"]:
                for run in RUNS:
                    cost = c["caps"][t] * P * I * c["s_eval"]
                    jobs.append(dict(pdn=pdn, swarm=tag(P, I), P=P, I=I, target=t, run=run, est_h=round(cost / 3600, 1),
                                     out=os.path.join(OUT, f"{pdn}_{tag(P, I)}", "pure_python", f"{t:g}", f"run_{run}")))
    jobs.sort(key=lambda j: -j["est_h"]); return jobs


def first_start(out, P, I):
    for f in sorted(glob.glob(os.path.join(out, "**", "run_*.log"), recursive=True)):
        try:
            for l in open(f, errors="replace"):
                if "RUN_START" in l: return l.strip(), (f"particles={P} iters={I}" in l)
        except OSError: pass
    return None, None


running = {}
def write_pids():
    json.dump([dict(pid=j["proc"].pid, pdn=j["pdn"], swarm=j["swarm"], target=j["target"], run=j["run"], cpu=j["cpu"]) for j in running.values() if j["proc"].poll() is None],
              open(os.path.join(LOGS, "pure_pids.json"), "w"))
def stop_all(*_):
    for j in running.values():
        if j["proc"].poll() is None: j["proc"].terminate()
    sys.exit(1)
signal.signal(signal.SIGTERM, stop_all)


def schedule():
    M, Pp, cmap = pools(); os.makedirs(LOGS, exist_ok=True); free = list(Pp); queue = [j for j in job_list() if not os.path.exists(os.path.join(j["out"], "results_long.jsonl"))
                                                                                   and not os.path.exists(os.path.join(j["out"], "ref", "results_long.jsonl"))]
    print(f"[{utc()}] {len(queue)} pure_python jobs, {len(free)} slots on cpus {free}", flush=True)
    for j in queue: print(f"   est {j['est_h']:7.1f} h  {j['pdn']} {j['swarm']} target {j['target']:g} run {j['run']}", flush=True)
    while queue or running:
        paused = os.path.exists(os.path.join(LOGS, "PAUSE_PURE"))
        while queue and free and not paused:
            j = queue.pop(0); cpu = free.pop(0); os.makedirs(j["out"], exist_ok=True)
            env = threads_env(); env.update(OUT=j["out"], TIME_LIMIT_S=TIME_LIMIT_S)
            log = open(os.path.join(LOGS, f"pure_{j['pdn']}_{j['swarm']}_{j['target']:g}_run{j['run']}.log"), "w")
            j["proc"] = subprocess.Popen(["taskset", "-c", str(cpu), PY, "-u", os.path.abspath(__file__), "--job", j["pdn"], j["swarm"], str(j["target"]), str(j["run"])],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
            j.update(cpu=cpu, t0=time.time(), start=utc(), load0=load(), verified=False); running[j["proc"].pid] = j
            put(dict(event="start", pdn=j["pdn"], swarm=j["swarm"], method="pure_python", target=j["target"], run=j["run"], cpu=cpu, pid=j["proc"].pid,
                     start_utc=j["start"], loadavg_start=j["load0"])); write_pids()
        time.sleep(5)
        for pid, j in list(running.items()):
            if not j["verified"]:
                line, ok = first_start(j["out"], j["P"], j["I"])
                if ok: j["verified"] = True; print(f"[{utc()}] {j['pdn']} {j['swarm']} {j['target']:g} run {j['run']}: {line}", flush=True)
                elif ok is False or (line is None and j["proc"].poll() is not None) or time.time() - j["t0"] > 1800:
                    print(f"[{utc()}] WRONG/MISSING RUN_START ({line!r}) -- killing all pure_python jobs and stopping", flush=True)
                    put(dict(event="abort", pdn=j["pdn"], swarm=j["swarm"], target=j["target"], run=j["run"], run_start=line, utc=utc())); stop_all()
            rc = j["proc"].poll()
            if rc is not None:
                put(dict(event="end", pdn=j["pdn"], swarm=j["swarm"], method="pure_python", target=j["target"], run=j["run"], cpu=j["cpu"], start_utc=j["start"], end_utc=utc(),
                         exit_code=rc, loadavg_start=j["load0"], loadavg_end=load(), wall_s=round(time.time() - j["t0"])))
                print(f"[{utc()}] done {j['pdn']} {j['swarm']} {j['target']:g} run {j['run']} exit {rc} after {(time.time() - j['t0']) / 3600:.2f} h", flush=True)
                free.append(j["cpu"]); del running[pid]; write_pids()
    open(os.path.join(LOGS, "PURE_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] all pure_python jobs finished", flush=True)


if __name__ == "__main__":
    if "--job" in sys.argv:
        a = sys.argv[sys.argv.index("--job") + 1:]; worker(*a[:4]); sys.exit(0)
    M, Pp, cmap = pools(); os.makedirs(LOGS, exist_ok=True)
    json.dump(dict(pool_M=M, pool_P=Pp, cores={str(c): cmap[c] for c in M + Pp}), open(os.path.join(LOGS, "pools_pure.json"), "w"), indent=1)
    print("pool P (pure_python):", [(c, cmap[c]["socket"], cmap[c]["core"]) for c in Pp], flush=True)
    if "--smoke" in sys.argv:
        ps = [subprocess.Popen(["taskset", "-c", str(c), PY, "-c", "import time; time.sleep(20)"]) for c in Pp]
        time.sleep(3); got = {}
        for c, p in zip(Pp, ps):
            for l in open(f"/proc/{p.pid}/status"):
                if l.startswith("Cpus_allowed_list"): got[c] = l.split()[1]
        json.dump(dict(pool=Pp, allowed=got, cores={c: [cmap[c]["socket"], cmap[c]["core"]] for c in Pp}), open(os.path.join(LOGS, "smoke_pure.json"), "w"), indent=1)
        print("smoke: 16 dummy jobs, affinity:", got, flush=True)
        for p in ps: p.wait()
        sys.exit(0)
    schedule()
