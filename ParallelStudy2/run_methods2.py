#!/usr/bin/env python3
"""ParallelStudy 2: same as ParallelStudy (28x28 full-board PDN, numpy / solve / sm / iterative, 50x15 and 100x40, targets 140/130/125/120 ohm, 4 runs, same seeds and warm start)
but the decap allowance is NOT limited to 14: it is N-1 = 27 (every pad except the observation port). The 21-port PDN already allows N-1 = 20, so it is not rerun.
One process per run, pinned to its own physical core of pool M (8 cores; pool P keeps serving the study-1 pure_python jobs). Longest job first (estimates from study 1).
A run stops at the first stage that reaches the target (as in study 1), so every run that hit within 14 caps in study 1 must REPRODUCE exactly (control); only previously missed
runs (and runs that hit at exactly 14) can go further.
   python run_methods2.py --dry         plan only (job list, estimated hours)
   python run_methods2.py --validate    checks the setup in-process (allowance, seeds, 1-stage smoke of the worker with a tiny swarm); no study output written
   python run_methods2.py               waits for the idle reference to finish (idle_reference.json has 16 rows and PAUSE_PURE is gone), then runs the queue
   python run_methods2.py --now         skip the wait
   python run_methods2.py --worker <swarm> <method> <T level> <run>
Env: SUBSET (all | missed; default all) -- missed = study-1 misses + study-1 hits at exactly 14 caps + run 1 of T4 per cell as extra controls
     SWARMS (50x15,100x40), NRUN (4), TIME_LIMIT_S (28800), PY."""
import os, sys, json, time, glob, signal, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H); S1 = os.path.join(ROOT, "ParallelStudy")
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out"); LOGS = os.path.join(H, "logs"); MAN = os.path.join(H, "manifest_methods2.jsonl")
TIME_LIMIT_S = os.environ.get("TIME_LIMIT_S", "28800"); NRUN = int(os.environ.get("NRUN", 4)); SUBSET = os.environ.get("SUBSET", "all")
SWARMS = os.environ.get("SWARMS", "50x15,100x40").split(","); METHODS = ["numpy", "solve", "sm", "iterative"]
PDN = "pcb28full"; LEVELS = ["T4", "T3", "T2", "T1"]; THR = {"T4": 140.0, "T3": 130.0, "T2": 125.0, "T1": 120.0}; ALLOW = 27     # N - 1 = 28 - 1 (observation port 1 takes no decap)
os.makedirs(LOGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(): return [round(x, 2) for x in os.getloadavg()]
def put(obj):
    with open(MAN, "a") as fh: fh.write(json.dumps(obj) + "\n")


def topology():
    """Identical to run_methods.py: one logical CPU per physical core, never core 0 / its sibling; M = 8 cores (4 per socket)."""
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); socks = sorted({k[0] for k in keys}); s = [[phys[k][0] for k in keys if k[0] == sk] for sk in socks]
    if len(s) != 2 and os.environ.get('PS2_ALLOW_NO_TOPO'): return []        # only for dry checks on a machine that is not the server
    assert len(s) == 2, 'expected 2 sockets'
    return s[0][:4] + s[1][:4]


def study1():
    """{(swarm, method, level, run): record} from the study-1 results (all result files of the 28-port PDN, pure_python excluded)."""
    rec = {}
    for sw in SWARMS:
        for f in glob.glob(os.path.join(S1, "out", f"{PDN}_{sw}", "**", "results_long.jsonl"), recursive=True):
            if f"{os.sep}pure_python{os.sep}" in f: continue
            for l in open(f):
                if l.strip():
                    r = json.loads(l); rec[(sw, r["method"], r["threshold_level"], r["run_id"])] = r
    return rec


def plan():
    s1 = study1(); jobs = []
    for sw in SWARMS:
        for me in METHODS:
            for lv in LEVELS:
                for run in range(1, NRUN + 1):
                    r = s1.get((sw, me, lv, run)); keep = SUBSET == "all"
                    if SUBSET == "missed": keep = r is None or (not r["success"]) or r["best_n_caps"] >= 14 or (lv == "T4" and run == 1)
                    if not keep: continue
                    w = r["total_wall_time_s"] if r else 3600.0
                    est = w * (ALLOW / 14.0) if (r is None or not r["success"]) else w         # a miss may now use up to 27/14 times as many stages
                    jobs.append(dict(swarm=sw, method=me, lvl=lv, run=run, est=est, s1=None if r is None else dict(success=r["success"], caps=r["best_n_caps"], peak=r["best_peak_ohm"]["B3G"], evals=r["n_fitness_evals"])))
    jobs.sort(key=lambda j: -j["est"]); return jobs


# ----------------------------------------------------------------------------------------------------------------- worker
def worker(swarm, method, lvl, run, smoke=False):
    out = os.environ["OUT"]; run = int(run); P, I = (int(x) for x in swarm.split("x"))
    os.environ.update(N_PARTICLES=str(P), N_ITERATIONS=str(I), METHODS=method, TIME_LIMIT_S=TIME_LIMIT_S, STUDY_THREADS="1", NET="full", MAX_CAPS=str(ALLOW))
    sys.path.insert(0, os.path.join(ROOT, "PSO_PCB28", "full28_study")); d = __import__("pso_full28"); core = d.core; core.library(); prob = core.Problem(**d.PROB)
    assert prob.N - prob.n_obs == len(prob.pads) == prob.max_caps == ALLOW == prob.N - 1, (prob.N, prob.n_obs, len(prob.pads), prob.max_caps)
    assert core.N_PARTICLES == P and core.N_ITERATIONS == I, (core.N_PARTICLES, core.N_ITERATIONS)
    os.makedirs(out, exist_ok=True)
    r = core.run_pso(prob, method, run, os.path.join(out, prob.name, method, lvl, f"run_{run}"), os.path.join(out, "results_long.jsonl"), pool=None, out_root=out,
                     threshold=d.T[lvl], level=lvl, max_caps=None)                                  # None -> prob.max_caps = 27
    assert r["max_caps"] == ALLOW, r["max_caps"]
    print(f"RESULT {method} {lvl} run {run}: best {r['best_peak_ohm']['B3G']:.6g} @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s, success={r['success']} timed_out={r['timed_out']} evals={r['n_fitness_evals']}", flush=True)


def first_start(out, swarm):
    P, I = swarm.split("x")
    for f in sorted(glob.glob(os.path.join(out, "**", "run_*.log"), recursive=True)):
        try:
            for l in open(f, errors="replace"):
                if "RUN_START" in l: return l.strip(), (f"particles={P} iters={I}" in l and f"max_caps={ALLOW}" in l)
        except OSError: pass
    return None, None


def validate():
    import tempfile
    M = topology(); jobs = plan(); print(f"pool M (cores): {M}"); assert (len(M) == 8 and len(set(M)) == 8) or os.environ.get('PS2_ALLOW_NO_TOPO')
    cells = {(j['swarm'], j['method'], j['lvl']) for j in jobs}; print(f"jobs {len(jobs)} ({SUBSET}); cells {len(cells)}; missing study-1 records: {sum(j['s1'] is None for j in jobs)}")
    sys.path.insert(0, os.path.join(ROOT, "PSO_PCB28", "full28_study")); os.environ.update(NET="full", MAX_CAPS=str(ALLOW)); d = __import__("pso_full28"); core = d.core; core.library()
    prob = core.Problem(**d.PROB); print(f"N={prob.N} n_obs={prob.n_obs} pads={len(prob.pads)} max_caps={prob.max_caps} (study 1 ceiling was 14)"); assert prob.max_caps == ALLOW
    print("targets:", d.T, "floor", d.FLOOR_OHM); assert all(d.T[k] == THR[k] for k in THR)
    # 3-stage smoke test with a tiny swarm in a temp dir, ceiling forced to 3 for speed only (does not touch the study output)
    tmp = tempfile.mkdtemp(prefix="ps2_smoke_"); core.N_PARTICLES, core.N_ITERATIONS = 24, 2
    r = core.run_pso(prob, "solve", 1, os.path.join(tmp, "r"), os.path.join(tmp, "res.jsonl"), pool=None, out_root=tmp, threshold=1.0, level="T1", max_caps=3)
    print(f"smoke ok: {r['n_fitness_evals']} evals, {r['best_n_caps']} caps, ceiling recorded {r['max_caps']} (tmp {tmp})")
    tot = sum(j["est"] for j in jobs); print(f"estimated serial work {tot / 3600:.1f} h; on 8 slots ~ {tot / 3600 / 8:.1f} h (+ tail: longest job {jobs[0]['est'] / 3600:.1f} h)")


children = []
def stop_all(*_):
    for p in children:
        if p.poll() is None: p.terminate()
    sys.exit(1)
signal.signal(signal.SIGTERM, stop_all)


def idle_done():
    f = os.path.join(S1, "out", "idle_reference", "idle_reference.json")
    try: n = len(json.load(open(f)))
    except Exception: n = 0
    return n >= 16 and not os.path.exists(os.path.join(S1, "logs", "PAUSE_PURE"))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        a = sys.argv[sys.argv.index("--worker") + 1:]; worker(*a[:4]); sys.exit(0)
    if "--validate" in sys.argv: validate(); sys.exit(0)
    jobs = plan(); M = topology()
    n_miss = sum(1 for j in jobs if j["s1"] and not j["s1"]["success"])
    print(f"[{utc()}] ParallelStudy2: {len(jobs)} runs ({SUBSET}), swarms {SWARMS}, allowance {ALLOW}, pool M {M}; study-1 misses among them: {n_miss}", flush=True)
    if "--dry" in sys.argv:
        for j in jobs[:12]: print(f"   {j['swarm']:7} {j['method']:9} {j['lvl']} run {j['run']}  est {j['est'] / 60:6.0f} min  study1 {j['s1']}")
        print(f"   ... estimated {sum(j['est'] for j in jobs) / 3600 / 8:.1f} h on 8 slots"); sys.exit(0)
    if "--now" not in sys.argv:
        print(f"[{utc()}] waiting for the study-1 idle reference to finish", flush=True)
        while not idle_done(): time.sleep(60)
    put(dict(event="begin", utc=utc(), runs=len(jobs), subset=SUBSET, allowance=ALLOW, pool_M=M, loadavg=load()))
    free = list(M); running = []
    while jobs or running:
        while jobs and free:
            j = jobs.pop(0); cpu = free.pop(0); out = os.path.join(OUT, f"{PDN}_{j['swarm']}", j["method"], j["lvl"], f"run_{j['run']}"); os.makedirs(out, exist_ok=True)
            env = dict(os.environ, OUT=out)
            for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): env[k + "_NUM_THREADS"] = "1"
            log = open(os.path.join(LOGS, f"s2_{j['swarm']}_{j['method']}_{j['lvl']}_run{j['run']}.log"), "w")
            j["proc"] = subprocess.Popen(["taskset", "-c", str(cpu), PY, "-u", os.path.abspath(__file__), "--worker", j["swarm"], j["method"], j["lvl"], str(j["run"])], env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(j["proc"]); j.update(cpu=cpu, out=out, t0=time.time(), start=utc(), load0=load(), verified=False); running.append(j)
            put(dict(event="start", swarm=j["swarm"], method=j["method"], target=THR[j["lvl"]], run=j["run"], cpu=cpu, pid=j["proc"].pid, start_utc=j["start"], loadavg_start=j["load0"]))
        time.sleep(5)
        for j in list(running):
            if not j["verified"]:
                line, ok = first_start(j["out"], j["swarm"])
                if ok: j["verified"] = True; print(f"[{utc()}] {j['swarm']} {j['method']} {j['lvl']} run {j['run']}: {line}", flush=True)
                elif ok is False or (line is None and j["proc"].poll() is not None) or time.time() - j["t0"] > 1200:
                    print(f"[{utc()}] WRONG/MISSING RUN_START ({line!r}) -- stopping all jobs", flush=True)
                    put(dict(event="abort", swarm=j["swarm"], method=j["method"], run=j["run"], run_start=line, utc=utc())); stop_all()
            rc = j["proc"].poll()
            if rc is not None:
                put(dict(event="end", swarm=j["swarm"], method=j["method"], target=THR[j["lvl"]], run=j["run"], cpu=j["cpu"], start_utc=j["start"], end_utc=utc(), exit_code=rc,
                         loadavg_start=j["load0"], loadavg_end=load(), wall_s=round(time.time() - j["t0"])))
                print(f"[{utc()}] done {j['swarm']} {j['method']} {j['lvl']} run {j['run']} exit {rc} after {(time.time() - j['t0']) / 60:.0f} min", flush=True)
                free.append(j["cpu"]); running.remove(j)
    open(os.path.join(LOGS, "STUDY2_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] ParallelStudy2 finished", flush=True)
