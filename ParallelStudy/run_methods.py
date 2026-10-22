#!/usr/bin/env python3
"""Step 3A. Methods study, two phases (50x15 then 100x40); phase 2 starts only when every phase-1 job has exited.
Each phase = 8 jobs at once = 4 methods (numpy, solve, sm, iterative) x 2 PDNs, each job pinned (taskset) to its own PHYSICAL core of pool M.
A job is one existing driver process (METHODS=<one method>, NUM_RUNS=4, easiest target first), nothing in the drivers is changed.
   python run_methods.py            # run both phases
   python run_methods.py --smoke    # 20 s dry run with dummy jobs: prints / saves the core map and the affinity of every dummy job
Env: PY (python), NUM_RUNS (4), TIME_LIMIT_S (14400)."""
import os, sys, json, time, glob, signal, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out"); LOGS = os.path.join(H, "logs"); MAN = os.path.join(H, "manifest_methods.jsonl")
NUM_RUNS = int(os.environ.get("NUM_RUNS", 4)); TIME_LIMIT_S = os.environ.get("TIME_LIMIT_S", "14400")
SWARMS = [(50, 15), (100, 40)]; METHODS = ["numpy", "solve", "sm", "iterative"]
PDNS = {"pcb28full": dict(driver=os.path.join(ROOT, "PSO_PCB28", "full28_study", "pso_full28.py"), env=dict(NET="full", MAX_CAPS="14"), what="T4(140)..T1(120)"),
        "ref21": dict(driver=os.path.join(ROOT, "PSO_REF21", "pso_ref21.py"), env=dict(LIB="ref", TARGETS="0.06,0.05,0.045", FLOOR_RUNS="0"), what="T1(0.06)..T3(0.045)")}
os.makedirs(LOGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(): return [round(x, 2) for x in os.getloadavg()]
def put(obj):
    with open(MAN, "a") as fh: fh.write(json.dumps(obj) + "\n")


def pools():
    """One logical CPU per physical core, never core 0 of socket 0 or its sibling. M = 8 cores, P = 16 cores, disjoint, split evenly over both sockets."""
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0])                                   # drops socket-0 core 0 together with its sibling thread
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


def start_line(out, P, I):
    for f in sorted(glob.glob(os.path.join(out, "**", "run_*.log"), recursive=True)):
        try:
            for l in open(f, errors="replace"):
                if "RUN_START" in l: return l.strip(), (f"particles={P} iters={I}" in l)
        except OSError: pass
    return None, None


children = []
def stop_all(*_):
    for p in children:
        if p.poll() is None: p.terminate()
    sys.exit(1)
signal.signal(signal.SIGTERM, stop_all)


def run_phase(swarm, M, phase):
    P, I = swarm; tag = f"{P}x{I}"; jobs = []
    for pdn in PDNS:
        for meth in METHODS:
            jobs.append(dict(pdn=pdn, method=meth, swarm=tag, out=os.path.join(OUT, f"{pdn}_{tag}", meth)))
    for j, cpu in zip(jobs, M):
        os.makedirs(j["out"], exist_ok=True)
        env = threads_env(); env.update(PDNS[j["pdn"]]["env"]); env.update(N_PARTICLES=str(P), N_ITERATIONS=str(I), METHODS=j["method"], NUM_RUNS=str(NUM_RUNS),
                                                                     TIME_LIMIT_S=TIME_LIMIT_S, OUT=j["out"])
        drv = PDNS[j["pdn"]]["driver"]
        log = open(os.path.join(LOGS, f"methods_{j['pdn']}_{tag}_{j['method']}.log"), "w")
        j["proc"] = subprocess.Popen(["taskset", "-c", str(cpu), PY, "-u", drv], cwd=os.path.dirname(drv), env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(j["proc"]); j.update(cpu=cpu, t0=time.time(), start=utc(), load0=load(), verified=False)
        put(dict(event="start", phase=phase, pdn=j["pdn"], swarm=tag, method=j["method"], target=PDNS[j["pdn"]]["what"], run=f"1-{NUM_RUNS}", cpu=cpu, pid=j["proc"].pid,
                 start_utc=j["start"], loadavg_start=j["load0"]))
    pending = set(range(len(jobs)))
    while pending:
        time.sleep(5)
        for i in sorted(pending):
            j = jobs[i]
            if not j["verified"]:
                line, ok = start_line(j["out"], P, I)
                if ok:
                    j["verified"] = True; print(f"[{utc()}] {j['pdn']} {tag} {j['method']}: {line}", flush=True)
                elif ok is False or (line is None and j["proc"].poll() is not None) or time.time() - j["t0"] > 1200:
                    print(f"[{utc()}] WRONG/MISSING RUN_START for {j['pdn']} {tag} {j['method']}: {line!r} -- killing all method jobs and stopping", flush=True)
                    put(dict(event="abort", pdn=j["pdn"], swarm=tag, method=j["method"], run_start=line, utc=utc())); stop_all()
            rc = j["proc"].poll()
            if rc is not None:
                put(dict(event="end", phase=phase, pdn=j["pdn"], swarm=tag, method=j["method"], target=PDNS[j["pdn"]]["what"], run=f"1-{NUM_RUNS}", cpu=j["cpu"],
                         start_utc=j["start"], end_utc=utc(), exit_code=rc, loadavg_start=j["load0"], loadavg_end=load(), wall_s=round(time.time() - j["t0"])))
                print(f"[{utc()}] {j['pdn']} {tag} {j['method']} exit {rc}", flush=True); pending.discard(i)


if __name__ == "__main__":
    M, P_, cmap = pools()
    json.dump(dict(pool_M=M, pool_P=P_, cores={str(c): cmap[c] for c in M + P_}), open(os.path.join(LOGS, "pools.json"), "w"), indent=1)
    print("pool M (methods):", [(c, cmap[c]["socket"], cmap[c]["core"]) for c in M], flush=True)
    if "--smoke" in sys.argv:
        ps = [subprocess.Popen(["taskset", "-c", str(c), PY, "-c", "import time; time.sleep(20)"]) for c in M]
        time.sleep(3); got = {}
        for c, p in zip(M, ps):
            for l in open(f"/proc/{p.pid}/status"):
                if l.startswith("Cpus_allowed_list"): got[c] = l.split()[1]
        json.dump(dict(pool=M, allowed=got, cores={c: [cmap[c]["socket"], cmap[c]["core"]] for c in M}), open(os.path.join(LOGS, "smoke_methods.json"), "w"), indent=1)
        print("smoke: 8 dummy jobs, affinity:", got, flush=True)
        for p in ps: p.wait()
        sys.exit(0)
    for ph, sw in enumerate(SWARMS, 1):
        print(f"[{utc()}] phase {ph}: {sw[0]}x{sw[1]}", flush=True); run_phase(sw, M, ph)
    open(os.path.join(LOGS, "METHODS_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] all method jobs finished", flush=True)
