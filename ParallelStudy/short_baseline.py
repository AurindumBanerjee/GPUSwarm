#!/usr/bin/env python3
"""Step 3C. Short IDLE pure_python baseline. Runs LAST: waits for logs/METHODS_DONE and logs/PURE_DONE and for our own scripts/drivers to be gone, then
 (1) times 20 pure_python evaluations (several different placements, one untimed warm-up) per PDN at the 50x15 and 100x40 settings on one pinned core and
     reports whether the per-evaluation cost agrees between the two settings;
 (2) runs one complete short pure_python run alone on one pinned core: 21-port, target 0.06 ohm, 50x15, run 1 (about 1-2 h).
Saves to out/short_baseline/ (timing.json, run folder). These per-evaluation times are the idle reference for the loaded pure_python timings.
   python short_baseline.py [--now]      # --now: do not wait
   python short_baseline.py --evals <pdn> <PxI>      # worker for (1)"""
import os, sys, json, time, subprocess, datetime

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
PY = os.environ.get("PY", sys.executable); OUT = os.path.join(H, "out", "short_baseline"); LOGS = os.path.join(H, "logs")
PDNS = {"pcb28full": dict(mod="pso_pcb28_pure", dir=os.path.join(ROOT, "PSO_PCB28", "baseline_pure_python")),
        "ref21": dict(mod="pso_ref21_pure", dir=os.path.join(ROOT, "PSO_REF21", "baseline_pure_python"))}
N_EVALS = 20; ORDER = ["50x15", "100x40"]


def utc(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def busy():
    me = os.getpid(); pats = ("run_methods.py", "run_pure.py", "idle_reference.py", "pso_full28.py", "pso_ref21.py", "pso_pcb28_pure.py", "pso_ref21_pure.py")
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != me:
            try: c = open(f"/proc/{p}/cmdline").read().replace("\0", " ")
            except OSError: continue
            if any(x in c for x in pats) and "short_baseline.py" not in c: out.append((int(p), c[:100]))
    return out


def evals(pdn, sw):
    import numpy as np
    P, I = (int(x) for x in sw.split("x"))
    os.environ.update(N_PARTICLES=str(P), N_ITERATIONS=str(I))
    os.environ.update(dict(NET="full", MAX_CAPS="14") if pdn == "pcb28full" else dict(LIB="ref"))
    cfg = PDNS[pdn]; sys.path.insert(0, cfg["dir"]); d = __import__(cfg["mod"]); core = d.core
    if pdn == "pcb28full": core.library(); prob = core.Problem(**d.PROB)
    else: d.base.load("ref"); prob = core.Problem("ref21", os.path.join(d.base.DATA, "y2.mat"), n_obs=1, obs_labels=["port0"])
    K = core.library()["d"].shape[0]; rng = np.random.RandomState(2026); times = []; placements = []
    core.peaks_all_bands(prob, [(int(rng.randint(K)), int(prob.pads[0]))], "pure_python")        # warm-up (builds the list copy of Y), not timed
    for k in range(N_EVALS):
        n = int(rng.randint(1, 9)); pads = rng.choice(prob.pads, n, replace=False); cfgk = [(int(rng.randint(K)), int(p)) for p in pads]
        t0 = time.perf_counter(); core.peaks_all_bands(prob, cfgk, "pure_python"); times.append(time.perf_counter() - t0); placements.append(n)
    print("RESULT " + json.dumps(dict(pdn=pdn, swarm=sw, times_s=times, n_caps=placements, mean_s=sum(times) / len(times), min_s=min(times), max_s=max(times))), flush=True)


if __name__ == "__main__":
    if "--evals" in sys.argv:
        a = sys.argv[sys.argv.index("--evals") + 1:]; evals(*a[:2]); sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    if "--now" not in sys.argv:
        while not (os.path.exists(os.path.join(LOGS, "METHODS_DONE")) and os.path.exists(os.path.join(LOGS, "PURE_DONE")) and not busy()): time.sleep(60)
    elif busy(): sys.exit(f"not idle: {busy()}")
    rows = [tuple(int(x) for x in l.split(",")) for l in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines() if l and not l.startswith("#")]
    phys = {}
    for cpu, core_, sock in rows: phys.setdefault((sock, core_), []).append(cpu)
    keys = sorted(phys); keys.remove(keys[0]); cpu = phys[keys[0]][0]
    env = dict(os.environ)
    for k in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"): env[k + "_NUM_THREADS"] = "1"
    env["STUDY_THREADS"] = "1"; timing = dict(cpu=cpu, start_utc=utc(), evals=[]); path = os.path.join(OUT, "timing.json")
    for pdn in PDNS:
        for sw in ORDER:
            p = subprocess.run(["taskset", "-c", str(cpu), PY, "-u", os.path.abspath(__file__), "--evals", pdn, sw], env=env, capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            timing["evals"].append(json.loads(line[7:]) if line else dict(pdn=pdn, swarm=sw, error=p.stderr[-500:]))
            json.dump(timing, open(path, "w"), indent=1); print(f"[{utc()}] evals {pdn} {sw}: {line}", flush=True)
    for pdn in PDNS:
        m = {e["swarm"]: e["mean_s"] for e in timing["evals"] if e["pdn"] == pdn and "mean_s" in e}
        if len(m) == 2: timing[f"{pdn}_agree"] = dict(mean_s=m, rel_diff=abs(m["50x15"] - m["100x40"]) / m["50x15"])
    json.dump(timing, open(path, "w"), indent=1)
    out = os.path.join(OUT, "ref21_50x15_T1_run1"); os.makedirs(out, exist_ok=True); env["OUT"] = out
    t0 = time.time(); p = subprocess.run(["taskset", "-c", str(cpu), PY, "-u", os.path.join(H, "run_pure.py"), "--job", "ref21", "50x15", "0.06", "1"], env=env, capture_output=True, text=True)
    timing["short_run"] = dict(exit_code=p.returncode, wall_s=round(time.time() - t0), out=out, stdout=p.stdout[-400:], stderr=p.stderr[-400:], end_utc=utc())
    json.dump(timing, open(path, "w"), indent=1); print(f"[{utc()}] short run exit {p.returncode} in {time.time() - t0:.0f} s", flush=True)
