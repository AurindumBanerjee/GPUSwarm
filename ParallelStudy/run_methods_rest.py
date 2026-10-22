#!/usr/bin/env python3
"""Replaces the sequential tail of run_methods.py phase 2 (28-port full network, 100x40) by one process per run on idle physical cores.
Started AFTER the old scheduler (run_methods.py) was SIGKILLed; its drivers (numpy / solve / sm / iterative, one process each) keep running meanwhile and are ADOPTED:
  1. wait until the solve driver (its final run) has exited; nothing new is launched before that;
  2. numpy / sm / iterative drivers: the run each one is in the middle of is allowed to complete (otherwise hours of work are thrown away); the moment it is
     written to results_long.jsonl the driver is SIGKILLed, before it can make progress on its next run. Its core is then free for the queue;
  3. every other remaining (method, target, run) of numpy / sm / iterative runs as its OWN process pinned to its own free PHYSICAL core, longest first
     (spare cores = physical cores that belong to neither pool M nor pool P and are not core 0; plus pool-M cores as they free up). One OUT dir and one results_long.jsonl per run:
     out/pcb28full_100x40/<method>_par/<T level>/run_<k>/ ;
  4. logs/METHODS_DONE is written when everything has finished (that triggers idle_reference.py).
   python run_methods_rest.py [--dry]        --dry: print the plan only (no kill, no launch)
   python run_methods_rest.py --worker <method> <T level> <run>      one run in this process (OUT in the environment)
Env: PY, TIME_LIMIT_S (14400)."""
import os, sys, json, time, glob, signal, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out"); LOGS = os.path.join(H, "logs"); MAN = os.path.join(H, "manifest_methods.jsonl")
TIME_LIMIT_S = os.environ.get("TIME_LIMIT_S", "14400"); SW = os.environ.get("PS_TEST_SWARM", "100x40"); P, I = (int(x) for x in SW.split("x"))
PDN = "pcb28full"; BASE = os.path.join(OUT, f"{PDN}_100x40"); LEVELS = ["T4", "T3", "T2", "T1"]; THR = {"T4": 140.0, "T3": 130.0, "T2": 125.0, "T1": 120.0}; NRUN = 4
EST = {"T1": {"iterative": 4800, "sm": 3500, "numpy": 3200}, "T2": {"iterative": 3000, "sm": 2200, "numpy": 2000}}      # seconds, only used to order the queue (longest first)
ADOPT = ["solve", "numpy", "sm", "iterative"]


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(): return [round(x, 2) for x in os.getloadavg()]
def put(obj):
    with open(MAN, "a") as fh: fh.write(json.dumps(obj) + "\n")
def alive(pid): return os.path.exists(f"/proc/{pid}") and "Z" not in open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
def nrec(method):
    f = os.path.join(BASE, method, "results_long.jsonl"); return sum(1 for l in open(f) if l.strip()) if os.path.exists(f) else 0


def topology():
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); socks = sorted({k[0] for k in keys}); s = [[phys[k][0] for k in keys if k[0] == sk] for sk in socks]
    M = s[0][:4] + s[1][:4]; Pp = s[0][4:12] + s[1][4:12]; spare = [c for c in s[0] + s[1] if c not in M and c not in Pp]
    return M, Pp, spare, phys


# ----------------------------------------------------------------------------------------------- worker: one (method, level, run)
def worker(method, lvl, run):
    out = os.environ["OUT"]; run = int(run)
    os.environ.update(N_PARTICLES=str(P), N_ITERATIONS=str(I), METHODS=method, TIME_LIMIT_S=TIME_LIMIT_S, STUDY_THREADS="1", NET="full", MAX_CAPS="14")
    sys.path.insert(0, os.path.join(ROOT, "PSO_PCB28", "full28_study")); d = __import__("pso_full28"); core = d.core; core.library(); prob = core.Problem(**d.PROB)
    os.makedirs(out, exist_ok=True)
    r = core.run_pso(prob, method, run, os.path.join(out, prob.name, method, lvl, f"run_{run}"), os.path.join(out, "results_long.jsonl"), pool=None, out_root=out,
                     threshold=d.T[lvl], level=lvl, max_caps=d.MAX_CAPS)
    print(f"RESULT {method} {lvl} run {run}: best {r['best_peak_ohm']['B3G']:.6g} @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.0f}s, success={r['success']} timed_out={r['timed_out']} evals={r['n_fitness_evals']}", flush=True)


def first_start(out):
    for f in sorted(glob.glob(os.path.join(out, "**", "run_*.log"), recursive=True)):
        try:
            for l in open(f, errors="replace"):
                if "RUN_START" in l: return l.strip(), (f"particles={P} iters={I}" in l)
        except OSError: pass
    return None, None


# ----------------------------------------------------------------------------------------------- main
def plan():
    st = {}
    for l in open(MAN):
        m = json.loads(l)
        if m.get("event") == "start" and m.get("pdn") == PDN and m.get("swarm") == "100x40" and m.get("method") in ADOPT: st[m["method"]] = m
    adopted = {me: dict(pid=st[me]["pid"], cpu=st[me]["cpu"], n0=nrec(me)) for me in ADOPT if me in st}
    seq = [(l, r) for l in LEVELS for r in range(1, NRUN + 1)]
    queue = []
    for me in ("numpy", "sm", "iterative"):
        a = adopted.get(me); n0 = a["n0"] if a else 0
        for (l, r) in seq[n0 + 1:] if (a and alive(a["pid"])) else seq[n0:]:
            queue.append(dict(method=me, lvl=l, run=r, est=EST.get(l, {}).get(me, 1500)))
    queue.sort(key=lambda j: -j["est"]); return adopted, queue


children = []
def stop_all(*_):
    for p in children:
        if p.poll() is None: p.terminate()
    sys.exit(1)
signal.signal(signal.SIGTERM, stop_all)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        a = sys.argv[sys.argv.index("--worker") + 1:]; worker(*a[:3]); sys.exit(0)
    M, Pp, spare, phys = topology(); adopted, queue = plan()
    print(f"[{utc()}] spare physical cores (not core 0, not pool M/P): {spare}; pool M: {M}", flush=True)
    for me, a in adopted.items(): print(f"   adopt {me:10} pid {a['pid']} cpu {a['cpu']} alive={alive(a['pid'])} records so far {a['n0']}/16", flush=True)
    print(f"   queue ({len(queue)} runs, longest first): " + ", ".join(f"{j['method']}/{j['lvl']}/r{j['run']}" for j in queue), flush=True)
    if "--dry" in sys.argv: sys.exit(0)
    free = list(spare) + [c for c in M if c not in {a["cpu"] for a in adopted.values() if alive(a["pid"])}]
    put(dict(event="adopt", utc=utc(), adopted={m: dict(a) for m, a in adopted.items()}, queued=len(queue), spare_cores=spare))
    solve_done = "solve" not in adopted
    running = []
    while queue or running or any(m != "solve" for m in adopted) or not solve_done:
        if not solve_done and not alive(adopted["solve"]["pid"]):                         # 1. launching starts only after the final solve run
            solve_done = True; free.append(adopted["solve"]["cpu"])
            put(dict(event="end", pdn=PDN, swarm="100x40", method="solve", note="driver exited after its final run", end_utc=utc(), loadavg_end=load()))
            print(f"[{utc()}] solve finished; launching {len(queue)} runs as separate processes", flush=True)
        for me in [m for m in list(adopted) if m != "solve"]:                              # 2. drain the old drivers after the run they are in
            a = adopted[me]
            if not alive(a["pid"]):
                free.append(a["cpu"]); del adopted[me]; put(dict(event="end", method=me, note="old driver exited on its own", end_utc=utc())); continue
            if nrec(me) > a["n0"]:
                try: os.kill(a["pid"], signal.SIGKILL)
                except ProcessLookupError: pass
                free.append(a["cpu"]); del adopted[me]
                put(dict(event="end", pdn=PDN, swarm="100x40", method=me, cpu=a["cpu"], end_utc=utc(), exit_code="SIGKILL after its in-progress run was recorded", loadavg_end=load()))
                print(f"[{utc()}] {me}: in-progress run recorded, old driver stopped (cpu {a['cpu']} freed)", flush=True)
        while queue and free and solve_done:                                               # 3. one process per remaining run
            j = queue.pop(0); cpu = free.pop(0); out = os.path.join(BASE, f"{j['method']}_par", j["lvl"], f"run_{j['run']}"); os.makedirs(out, exist_ok=True)
            env = dict(os.environ, OUT=out)
            for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): env[k + "_NUM_THREADS"] = "1"
            log = open(os.path.join(LOGS, f"methods_par_{j['method']}_{j['lvl']}_run{j['run']}.log"), "w")
            j["proc"] = subprocess.Popen(["taskset", "-c", str(cpu), PY, "-u", os.path.abspath(__file__), "--worker", j["method"], j["lvl"], str(j["run"])], env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(j["proc"]); j.update(cpu=cpu, out=out, t0=time.time(), start=utc(), load0=load(), verified=False); running.append(j)
            put(dict(event="start", phase=2, pdn=PDN, swarm="100x40", method=j["method"], target=THR[j["lvl"]], run=j["run"], cpu=cpu, pid=j["proc"].pid, start_utc=j["start"], loadavg_start=j["load0"]))
        time.sleep(5)
        for j in list(running):
            if not j["verified"]:
                line, ok = first_start(j["out"])
                if ok: j["verified"] = True; print(f"[{utc()}] {j['method']} {j['lvl']} run {j['run']}: {line}", flush=True)
                elif ok is False or (line is None and j["proc"].poll() is not None) or time.time() - j["t0"] > 1200:
                    print(f"[{utc()}] WRONG/MISSING RUN_START ({line!r}) -- killing all new jobs and stopping", flush=True)
                    put(dict(event="abort", method=j["method"], target=THR[j["lvl"]], run=j["run"], run_start=line, utc=utc())); stop_all()
            rc = j["proc"].poll()
            if rc is not None:
                put(dict(event="end", phase=2, pdn=PDN, swarm="100x40", method=j["method"], target=THR[j["lvl"]], run=j["run"], cpu=j["cpu"], start_utc=j["start"], end_utc=utc(), exit_code=rc,
                         loadavg_start=j["load0"], loadavg_end=load(), wall_s=round(time.time() - j["t0"])))
                print(f"[{utc()}] done {j['method']} {j['lvl']} run {j['run']} exit {rc} after {(time.time() - j['t0']) / 60:.0f} min", flush=True)
                free.append(j["cpu"]); running.remove(j)
    open(os.path.join(LOGS, "METHODS_DONE"), "w").write(utc() + "\n"); print(f"[{utc()}] all method runs finished", flush=True)
