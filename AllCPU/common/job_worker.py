#!/usr/bin/env python3
"""AllCPU worker: ONE process = ONE (PDN, method, swarm), single thread, loops over (band, level, run) sequentially. Resumable.
   python job_worker.py --pdn pdn3 --method numpy --swarm 50x15 --out DIR [--bands B20,B50] [--levels T1,T2] [--runs 4] [--dry]
Output: DIR/<pdn>/<swarm>/<method>/results_long.jsonl  and  DIR/<pdn>/<swarm>/<method>/<band>/<level>/run_<k>/ (log, plots).
Floor mode (used by floor_pass.py):  --floor --pdn pdn3 --rail mer1 --band B50 --run 1 --out DIR   (full capacitor budget, no target, patience stop)
Env: PATIENCE (PDN 3/4 default 6 capacitor counts, off for PDN 1/2), PATIENCE_TOL (1e-3), TIME_LIMIT_S (28800), WARM_START_PCT (20), GPUSWARM_ROOT."""
import os, sys, json, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
ap = argparse.ArgumentParser()
ap.add_argument("--pdn", required=True); ap.add_argument("--method", default="numpy"); ap.add_argument("--swarm", default="50x15"); ap.add_argument("--out", required=True)
ap.add_argument("--bands"); ap.add_argument("--levels"); ap.add_argument("--runs", type=int, default=int(os.environ.get("NUM_RUNS", 4)))
ap.add_argument("--dry", action="store_true"); ap.add_argument("--floor", action="store_true"); ap.add_argument("--rail"); ap.add_argument("--band"); ap.add_argument("--run", type=int, default=1)
a = ap.parse_args()
P, I = (int(x) for x in a.swarm.split("x"))
os.environ.update(STUDY_THREADS="1", WARM_START_PCT=os.environ.get("WARM_START_PCT", "20"), TIME_LIMIT_S=os.environ.get("TIME_LIMIT_S", "28800"))
if a.pdn in ("pdn3", "pdn4"): os.environ.setdefault("PATIENCE", "6")          # guard against unreachable levels walking through all 68-78 capacitor counts
import allcpu_pdns as R
cell = R.Cell(a.pdn, P, I, load_problem=not a.floor); core = cell.core; s = cell.spec


def log(*x): print(*x, flush=True)


if a.floor:
    import allcpu_core
    rail = a.rail
    path = R.RAIL_FILES[rail] if rail else s["path"]; name = f"floor_{a.pdn}_{rail or 'full'}"
    prob = core.Problem(name, path, n_obs=1); core.OBJ_WEIGHTS = None; core.RANK_BAND = a.band
    os.makedirs(a.out, exist_ok=True)
    rec = core.run_pso(prob, "numpy", a.run, os.path.join(a.out, "run"), os.path.join(a.out, "results_long.jsonl"), pool=None, out_root=a.out, threshold=None, level="floor", max_caps=None)
    log(f"FLOOR {name} {a.band} run {a.run}: best {rec['best_cost']:.6g} @ {rec['best_n_caps']} caps ({rec['n_fitness_evals']} evals, {rec['total_wall_time_s']:.0f}s, stagnated={rec['stagnated']}, timed_out={rec['timed_out']})")
    sys.exit(0)

base = os.path.join(a.out, a.pdn, a.swarm, a.method); res = os.path.join(base, "results_long.jsonl")
bands = a.bands.split(",") if a.bands else s["opt_bands"]
done = set()
if os.path.exists(res):
    for l in open(res):
        if l.strip(): r = json.loads(l); done.add((r["rank_band"], r["threshold_level"], r["run_id"]))
plan = []
for b in bands:
    levels, info = cell.select_band(b)
    for lv in cell.order(levels):
        if a.levels and lv not in a.levels.split(","): continue
        for k in range(1, a.runs + 1): plan.append((b, lv, k, levels[lv], info))
log(f"{s['label']} | {a.method} | {a.swarm} | warm {core.n_warm_count()}/{core.N_PARTICLES} | N={cell.prob.N} obs={cell.prob.n_obs} pads={len(cell.prob.pads)} F={cell.prob.F} | {len(plan)} runs ({len(done)} done) | objective {'ratio' if s['ratio'] else 'ohm'}")
if a.dry:
    for b, lv, k, thr, info in plan: log(f"   {b:7} {lv} run {k}  threshold {thr:.6g} {info['unit']} (floor {info['floor']:.6g}, bare {info['bare']:.6g})")
    sys.exit(0)
os.makedirs(base, exist_ok=True); cur = None
for b, lv, k, thr, info in plan:
    if (b, lv, k) in done: continue
    if cur != b: cell.select_band(b); cur = b                                  # (re)install RANK_BAND / weights for this band
    log(f"{s['name']} {a.method} {a.swarm} band {b} {lv}={thr:.6g} {info['unit']} run {k}/{a.runs}")
    try:
        r = core.run_pso(cell.prob, a.method, k, os.path.join(base, b, lv, f"run_{k}"), res, pool=None, out_root=base, threshold=thr, level=lv, max_caps=None)
        log(f"    best {r['best_cost']:.6g} {info['unit']} @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.1f}s, " + (f"hit in {r['time_to_target_s']:.1f}s" if r["success"] else f"target missed (stagnated={r['stagnated']} timed_out={r['timed_out']})") + f", evals {r['n_fitness_evals']}")
    except Exception as exc: log("    [ERROR]", repr(exc))
log("WORKER_DONE")
