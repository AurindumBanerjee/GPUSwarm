#!/usr/bin/env python3
"""Step 5. Idle reference for the methods. Waits for logs/METHODS_DONE (every 3A job has exited), then pauses the 3B jobs (SIGSTOP, pids from logs/pure_pids.json,
and logs/PAUSE_PURE keeps run_pure.py from launching new ones), runs ONE job at a time on one pinned core of pool M: for each swarm, PDN and method the easiest target,
run 1 (same seed as the loaded run), asserts n_fitness_evals equals the loaded run's, then resumes the 3B jobs (SIGCONT, in a finally block) and records the pause in manifest_pure.jsonl.
   python idle_reference.py            # waits for METHODS_DONE
   python idle_reference.py --now      # do not wait (only for a controlled test)
   python idle_reference.py --worker <pdn> <PxI> <method>      # ONE job in this process (used internally)
Output: out/idle_reference/<pdn>_<swarm>/<method>/..., out/idle_reference/idle_reference.json"""
import os, sys, json, time, glob, signal, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out"); LOGS = os.path.join(H, "logs"); MAN = os.path.join(H, "manifest_pure.jsonl")
REF = os.path.join(OUT, "idle_reference"); SWARMS = ["50x15", "100x40"]; PDNS = ["pcb28full", "ref21"]; METHODS = ["numpy", "solve", "sm", "iterative"]
EASIEST = {"pcb28full": "T4", "ref21": "T1"}


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def put(obj):
    with open(MAN, "a") as fh: fh.write(json.dumps(obj) + "\n")


def pool_m_cpu():
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core, sock in rows: phys.setdefault((sock, core), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); return phys[[k for k in keys if k[0] == sorted({k[0] for k in keys})[0]][0]][0]     # first core of pool M


def worker(pdn, sw, method):
    P, I = (int(x) for x in sw.split("x")); out = os.environ["OUT"]
    os.environ.update(N_PARTICLES=str(P), N_ITERATIONS=str(I), METHODS=method, TIME_LIMIT_S="14400", STUDY_THREADS="1")
    os.environ.update(dict(NET="full", MAX_CAPS="14") if pdn == "pcb28full" else dict(LIB="ref", TARGETS="0.06,0.05,0.045", FLOOR_RUNS="0"))
    if pdn == "pcb28full":
        sys.path.insert(0, os.path.join(ROOT, "PSO_PCB28", "full28_study")); d = __import__("pso_full28"); core = d.core; core.library(); prob = core.Problem(**d.PROB)
        lvl = "T4"; thr = d.T[lvl]; os.makedirs(out, exist_ok=True)
        r = core.run_pso(prob, method, 1, os.path.join(out, prob.name, method, lvl, "run_1"), os.path.join(out, "results_long.jsonl"), pool=None, out_root=out,
                         threshold=thr, level=lvl, max_caps=d.MAX_CAPS)
    else:
        sys.path.insert(0, os.path.join(ROOT, "PSO_REF21")); d = __import__("pso_ref21"); core = d.core; d.load("ref")
        prob = core.Problem("ref21", os.path.join(d.DATA, "y2.mat"), n_obs=1, obs_labels=["port0"]); lvl = "T1"; thr = d.TARGETS[0]
        o2 = os.path.join(out, "ref"); os.makedirs(o2, exist_ok=True)
        r = core.run_pso(prob, method, 1, os.path.join(o2, prob.name, method, lvl, "run_1"), os.path.join(o2, "results_long.jsonl"), pool=None, out_root=o2, threshold=thr, level=lvl)
    print("RESULT " + json.dumps({k: r[k] for k in ("n_fitness_evals", "time_to_target_s", "total_wall_time_s", "success", "best_n_caps", "caps_at_target")} | dict(level=lvl, target=thr)), flush=True)


def loaded_record(pdn, sw, method):
    for f in glob.glob(os.path.join(OUT, f"{pdn}_{sw}", method, "**", "results_long.jsonl"), recursive=True):
        for l in open(f):
            r = json.loads(l)
            if r["run_id"] == 1 and r["threshold_level"] == EASIEST[pdn] and r["method"] == method: return r
    return None


if __name__ == "__main__":
    if "--worker" in sys.argv:
        a = sys.argv[sys.argv.index("--worker") + 1:]; worker(*a[:3]); sys.exit(0)
    os.makedirs(REF, exist_ok=True)
    if "--now" not in sys.argv:
        while not os.path.exists(os.path.join(LOGS, "METHODS_DONE")): time.sleep(60)
    cpu = pool_m_cpu(); pids = []
    def read_pids():
        try: return [q["pid"] for q in json.load(open(os.path.join(LOGS, "pure_pids.json"))) if os.path.exists(f"/proc/{q['pid']}")]
        except (OSError, ValueError): return []
    open(os.path.join(LOGS, "PAUSE_PURE"), "w").write(utc())      # flag FIRST: run_pure.py launches nothing new from here on
    t_pause = time.time(); u0 = utc(); results = []
    time.sleep(12)                                                # a launch that began before the flag (scheduler loop is 5 s) has finished and written its pid
    try:
        for rnd in range(2):                                      # second pass catches any job that slipped in during the first
            for pid in read_pids():
                if pid in pids: continue
                pids.append(pid)
                try: os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError: pass
            time.sleep(3)
        print(f"[{u0}] paused {len(pids)} pure_python jobs; idle runs on cpu {cpu}", flush=True); time.sleep(10)
        for sw in SWARMS:
            for pdn in PDNS:
                for method in METHODS:
                    out = os.path.join(REF, f"{pdn}_{sw}", method); env = dict(os.environ, OUT=out)
                    for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): env[k + "_NUM_THREADS"] = "1"
                    t0 = time.time(); la0 = [round(x, 2) for x in os.getloadavg()]
                    p = subprocess.run(["taskset", "-c", str(cpu), PY, "-u", os.path.abspath(__file__), "--worker", pdn, sw, method], env=env, capture_output=True, text=True)
                    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
                    row = dict(pdn=pdn, swarm=sw, method=method, cpu=cpu, exit_code=p.returncode, start_utc=utc(), loadavg_start=la0, loadavg_end=[round(x, 2) for x in os.getloadavg()])
                    if line:
                        idle = json.loads(line[7:]); ld = loaded_record(pdn, sw, method)
                        row.update(idle=idle)
                        if ld:
                            row.update(loaded=dict(n_fitness_evals=ld["n_fitness_evals"], time_to_target_s=ld["time_to_target_s"], total_wall_time_s=ld["total_wall_time_s"], success=ld["success"]),
                                       evals_match=(ld["n_fitness_evals"] == idle["n_fitness_evals"]))
                            if not row["evals_match"]: print(f"   MISMATCH evals {pdn} {sw} {method}: loaded {ld['n_fitness_evals']} vs idle {idle['n_fitness_evals']}", flush=True)
                            lt = ld["time_to_target_s"] if ld["success"] else ld["total_wall_time_s"]; it = idle["time_to_target_s"] if idle["success"] else idle["total_wall_time_s"]
                            row.update(loaded_s=lt, idle_s=it, slowdown=(lt / it if it else None), loaded_s_per_eval=ld["total_wall_time_s"] / ld["n_fitness_evals"],
                                       idle_s_per_eval=idle["total_wall_time_s"] / idle["n_fitness_evals"])
                        else: row["loaded"] = None
                    results.append(row); json.dump(results, open(os.path.join(REF, "idle_reference.json"), "w"), indent=1)
                    print(f"[{utc()}] idle {pdn} {sw} {method}: exit {p.returncode} {line}", flush=True)
    finally:
        for pid in pids:
            try: os.kill(pid, signal.SIGCONT)
            except ProcessLookupError: pass
        try: os.remove(os.path.join(LOGS, "PAUSE_PURE"))
        except OSError: pass
        put(dict(event="pause", start_utc=u0, end_utc=utc(), pause_s=round(time.time() - t_pause), paused_pids=pids))
        print(f"[{utc()}] resumed {len(pids)} pure_python jobs after {(time.time() - t_pause) / 60:.1f} min", flush=True)
