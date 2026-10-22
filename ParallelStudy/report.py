#!/usr/bin/env python3
"""Step 6. Report for the parallel study; run after 3C.   python report.py [--no-plots]
Reads out/<pdn>_<swarm>/..../results_long.jsonl (methods + pure_python), manifest_pure.jsonl (pauses), out/idle_reference/idle_reference.json, out/short_baseline/timing.json.
Writes report.txt, report_*.csv and out/post_run/<pdn>_<swarm>/ (placement plots + tables through ../post_run.py, one call per PDN and swarm so the swarms never mix)."""
import os, sys, json, glob, csv, shutil, datetime, subprocess, collections
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H); OUT = os.environ.get("PS_OUT", os.path.join(H, "out")); PY = os.environ.get("PY", sys.executable)
LINE = "methods: 8 single-thread pinned jobs; pure_python: 16 single-thread pinned jobs on disjoint physical cores; BLAS=1"
rep = []
def say(s=""): rep.append(s); print(s)
def ts(s): return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

recs = []
for f in sorted(glob.glob(os.path.join(OUT, "*_*x*", "**", "results_long.jsonl"), recursive=True)):
    part = os.path.relpath(f, OUT).split(os.sep)[0]
    if part in ("short_baseline", "idle_reference"): continue
    pdn, sw = part.rsplit("_", 1)
    for r in jl(f): r["_pdn"], r["_swarm"], r["_file"] = pdn, sw, f; recs.append(r)
pauses = []
mp = os.path.join(H, "manifest_pure.jsonl")
if os.path.exists(mp): pauses = [(ts(m["start_utc"]), ts(m["end_utc"])) for m in jl(mp) if m.get("event") == "pause"]


def adjust(r):
    """wall / time-to-target with the SIGSTOP pause removed (only the part of a pause that lies inside the run, and before the target was hit)."""
    end = ts(r["timestamp_utc"]); start = end - r["total_wall_time_s"]; wall = r["total_wall_time_s"]; ttt = r["time_to_target_s"]
    for a, b in pauses:
        wall -= max(0.0, min(end, b) - max(start, a))
        if ttt is not None: ttt -= max(0.0, min(start + r["time_to_target_s"], b) - max(start, a))
    return wall, ttt
for r in recs: r["_wall"], r["_ttt"] = adjust(r) if r["method"] == "pure_python" else (r["total_wall_time_s"], r["time_to_target_s"])
band = lambda r: "B3G" if "B3G" in r["best_peak_ohm"] else "BFULL"
G = collections.defaultdict(list)
for r in recs: G[(r["_pdn"], r["_swarm"], r["threshold_ohm"], r["method"])].append(r)
ORDER = ["pure_python", "numpy", "solve", "sm", "iterative"]
def mean_ok(g, key): v = [r[key] for r in g if r["success"] and r[key] is not None]; return float(np.mean(v)) if v else None

say("PARALLEL STUDY REPORT   " + LINE); say()
# ------------------------------------------------------------------ 1. method tables per pdn and swarm
csv1 = [["pdn", "swarm", "target_ohm", "method", "hits", "runs", "mean_s", "median_s", "caps_at_target", "evals", "best_peak_ohm", "speedup_vs_numpy"]]
for pdn in sorted({k[0] for k in G}):
    for sw in ("50x15", "100x40"):
        ks = [k for k in G if k[0] == pdn and k[1] == sw]
        if not ks: continue
        say(f"== {pdn} {sw}: hits, time to target (successful runs), speedup vs numpy =="); say(f"{'target':>9} {'method':12} {'hit':>6} {'mean s':>10} {'median s':>10} {'caps':>5} {'evals':>8} {'best':>10} {'x numpy':>8}")
        for thr in sorted({k[2] for k in ks if k[2] is not None}, reverse=True):
            nb = mean_ok(G.get((pdn, sw, thr, "numpy"), []), "_ttt")
            for m in ORDER:
                g = G.get((pdn, sw, thr, m))
                if not g: continue
                ok = [r for r in g if r["success"]]; t = [r["_ttt"] for r in ok]
                mean = float(np.mean(t)) if t else float("nan"); med = float(np.median(t)) if t else float("nan")
                caps = float(np.mean([r["caps_at_target"] for r in ok])) if ok else float("nan"); ev = float(np.mean([r["n_fitness_evals"] for r in ok])) if ok else float("nan")
                best = min(min(r["best_peak_ohm"].values()) for r in g); sp = (nb / mean) if (nb and t and m != "pure_python") else None
                say(f"{thr:9.4g} {m:12} {len(ok):2d}/{len(g):<3d} {mean:10.2f} {med:10.2f} {caps:5.1f} {ev:8.0f} {best:10.5g} {('%.2fx' % sp) if sp else '-':>8}")
                csv1.append([pdn, sw, thr, m, len(ok), len(g), mean, med, caps, ev, best, sp])
        say()
csv.writer(open(os.path.join(H, "report_methods.csv"), "w", newline="")).writerows(csv1)

# ------------------------------------------------------------------ 2. pure_python: measured vs projected, slowdown
tp = os.path.join(OUT, "short_baseline", "timing.json"); idle_eval = {}
if os.path.exists(tp):
    T = json.load(open(tp))
    for pdn in ("pcb28full", "ref21"):
        v = [e["mean_s"] for e in T["evals"] if e["pdn"] == pdn and "mean_s" in e]
        if v: idle_eval[pdn] = float(np.mean(v))
    say("== idle pure_python cost per evaluation (short_baseline, 20 evals each, one pinned core) ==")
    for e in T["evals"]:
        if "mean_s" in e: say(f"   {e['pdn']:10} {e['swarm']:7} mean {e['mean_s']:.3f} s  min {e['min_s']:.3f}  max {e['max_s']:.3f}")
    for pdn in ("pcb28full", "ref21"):
        a = T.get(f"{pdn}_agree")
        if a: say(f"   {pdn}: 50x15 vs 100x40 agree within {100 * a['rel_diff']:.1f} %")
    if "short_run" in T: say(f"   short run (21-port, 0.06 ohm, 50x15, run 1, alone): exit {T['short_run']['exit_code']}, {T['short_run']['wall_s']} s"); say()
# the pure_python load changes over time: 24 jobs while the methods run, 16 afterwards. Only runs that overlapped the method window enter the slowdown.
mm = os.path.join(H, "manifest_methods.jsonl"); mwin = None
if os.path.exists(mm):
    ev = jl(mm); st = [ts(m["start_utc"]) for m in ev if m.get("event") == "start"]; en = [ts(m["end_utc"]) for m in ev if m.get("event") == "end"]
    if st and en: mwin = (min(st), max(en))
def overlap(r):
    if mwin is None: return 1.0
    end = ts(r["timestamp_utc"]); start = end - r["total_wall_time_s"]
    return max(0.0, min(end, mwin[1]) - max(start, mwin[0])) / max(1e-9, end - start)
loaded_eval, loaded_all, n_over = {}, {}, {}
for pdn in ("pcb28full", "ref21"):
    pr = [r for r in recs if r["_pdn"] == pdn and r["method"] == "pure_python"]
    if not pr: continue
    loaded_all[pdn] = sum(r["_wall"] for r in pr) / sum(r["n_fitness_evals"] for r in pr)
    ov = [r for r in pr if overlap(r) >= 0.9] or pr                  # fall back to all runs when none overlapped >= 90 %
    n_over[pdn] = (len(ov), len(pr)); loaded_eval[pdn] = sum(r["_wall"] for r in ov) / sum(r["n_fitness_evals"] for r in ov)
slow = {p: loaded_eval[p] / idle_eval[p] for p in loaded_eval if p in idle_eval}
say("== pure_python slowdown: loaded s/eval (pause-adjusted wall / evals; runs that overlapped the method window by >= 90 %) / idle s/eval (short_baseline) ==")
for p in loaded_eval: say(f"   {p:10} loaded {loaded_eval[p]:.3f} s/eval ({n_over[p][0]} of {n_over[p][1]} runs)   all runs {loaded_all[p]:.3f}   idle {idle_eval.get(p, float('nan')):.3f} s/eval   slowdown {slow.get(p, float('nan')):.2f}x")
# paired check: the 3B job ref21 50x15 0.06 run 1 and the idle 3C short run use the same seed, so their evaluation counts must match and only the speed differs
sp_ = os.path.join(OUT, "short_baseline", "ref21_50x15_T1_run1", "ref", "results_long.jsonl")
lo_ = [r for r in recs if r["_pdn"] == "ref21" and r["_swarm"] == "50x15" and r["method"] == "pure_python" and abs(r["threshold_ohm"] - 0.06) < 1e-12 and r["run_id"] == 1]
if os.path.exists(sp_) and lo_:
    idl, lod = jl(sp_)[0], lo_[0]; si = idl["total_wall_time_s"] / idl["n_fitness_evals"]; sl = lod["_wall"] / lod["n_fitness_evals"]
    say(f"   paired (ref21, 50x15, 0.06 ohm, run 1, same seed): evals loaded {lod['n_fitness_evals']} vs idle {idl['n_fitness_evals']} ({'match' if lod['n_fitness_evals'] == idl['n_fitness_evals'] else 'MISMATCH'}); "
        f"loaded {sl:.3f} s/eval, idle {si:.3f} s/eval, slowdown {sl / si:.2f}x; wall loaded {lod['_wall']:.0f} s vs idle {idl['total_wall_time_s']:.0f} s")
say("   3B ran concurrently with the methods, so the primary pure_python timings below are LOADED times; the idle-corrected value is loaded time / slowdown."); say()
csv2 = [["pdn", "swarm", "target_ohm", "status", "runs_done", "runs_timed_out_or_missing", "pure_s", "projected_s", "calibration_err_pct", "idle_corrected_s", "numpy_mean_evals"]]
proj = {}
say("== pure_python time to target per (pdn, swarm, target): measured vs projected ==")
say(f"{'pdn':10} {'swarm':7} {'target':>8} {'status':9} {'done':>5} {'pure s (loaded)':>16} {'projected s':>12} {'calib err':>10} {'idle-corr s':>12}")
for pdn in ("pcb28full", "ref21"):
    for sw in ("100x40", "50x15"):
        for thr in sorted({k[2] for k in G if k[0] == pdn and k[1] == sw and k[3] == "pure_python"}, reverse=True):
            g = G[(pdn, sw, thr, "pure_python")]; ok = [r for r in g if r["success"]]; ng = G.get((pdn, sw, thr, "numpy"), [])
            ev = float(np.mean([r["n_fitness_evals"] for r in ng])) if ng else None
            pj = ev * loaded_eval[pdn] if (ev and pdn in loaded_eval) else None; meas = float(np.mean([r["_ttt"] for r in ok])) if ok else None
            status = "measured" if ok and len(ok) == len(g) and len(g) >= 1 else ("partial" if ok else "projected")
            val = meas if meas is not None else pj
            cal = 100 * (pj - meas) / meas if (pj and meas) else None
            proj[(pdn, sw, thr)] = dict(measured=meas, projected=pj)
            ic = val / slow[pdn] if (val and pdn in slow) else None
            say(f"{pdn:10} {sw:7} {thr:8.4g} {status:9} {len(ok):2d}/{len(g)} {(('%.0f' % meas) if meas else '-'):>16} {(('%.0f' % pj) if pj else '-'):>12} {(('%+.1f %%' % cal) if cal is not None else '-'):>10} {(('%.0f' % ic) if ic else '-'):>12}")
            csv2.append([pdn, sw, thr, status, len(ok), len(g) - len(ok), meas, pj, cal, ic, ev])
say(); csv.writer(open(os.path.join(H, "report_pure.csv"), "w", newline="")).writerows(csv2)
# ------------------------------------------------------------------ 3. speedup of every method vs pure_python, measured and projected separately
for kind in ("measured", "projected"):
    say(f"== speedup vs pure_python ({kind} pure_python time / mean method time-to-target) ==")
    say(f"{'pdn':10} {'swarm':7} {'target':>8}  " + "  ".join(f"{m:>9}" for m in ORDER[1:]))
    for (pdn, sw, thr), v in sorted(proj.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
        pv = v[kind]
        if pv is None: continue
        cells = []
        for m in ORDER[1:]:
            mm = mean_ok(G.get((pdn, sw, thr, m), []), "_ttt"); cells.append(f"{pv / mm:8.0f}x" if mm else f"{'-':>9}")
        say(f"{pdn:10} {sw:7} {thr:8.4g}  " + "  ".join(cells))
    say()
# ------------------------------------------------------------------ 4. idle vs loaded slowdown for the methods
ip = os.path.join(OUT, "idle_reference", "idle_reference.json")
if os.path.exists(ip):
    say("== methods: loaded vs idle (easiest target, run 1, same seed; loaded = 8 method + 16 pure jobs, idle = 3B paused) =="); csv3 = [["pdn", "swarm", "method", "evals_match", "loaded_s", "idle_s", "slowdown", "loaded_s_per_eval", "idle_s_per_eval"]]
    say(f"{'pdn':10} {'swarm':7} {'method':10} {'evals ==':>8} {'loaded s':>10} {'idle s':>10} {'slowdown':>9} {'loaded s/eval':>14} {'idle s/eval':>12}")
    for row in json.load(open(ip)):
        if "idle" not in row or row.get("loaded") is None: say(f"{row['pdn']:10} {row['swarm']:7} {row['method']:10} (no loaded or idle result: exit {row.get('exit_code')})"); continue
        say(f"{row['pdn']:10} {row['swarm']:7} {row['method']:10} {str(row['evals_match']):>8} {row['loaded_s']:10.1f} {row['idle_s']:10.1f} {row['slowdown']:8.2f}x {row['loaded_s_per_eval']:14.4f} {row['idle_s_per_eval']:12.4f}")
        csv3.append([row["pdn"], row["swarm"], row["method"], row["evals_match"], row["loaded_s"], row["idle_s"], row["slowdown"], row["loaded_s_per_eval"], row["idle_s_per_eval"]])
    csv.writer(open(os.path.join(H, "report_slowdown_methods.csv"), "w", newline="")).writerows(csv3)
    if pauses: say(f"   pure_python paused for {sum(b - a for a, b in pauses) / 60:.1f} min (removed from the pure_python timings above)")
say(); say(LINE)
open(os.path.join(H, "report.txt"), "w").write("\n".join(rep) + "\n")
# ------------------------------------------------------------------ 5. placement plots + post_run tables, one call per pdn and swarm
if "--no-plots" not in sys.argv:
    for pdn, sw in sorted({(r["_pdn"], r["_swarm"]) for r in recs}):
        files = sorted({r["_file"] for r in recs if r["_pdn"] == pdn and r["_swarm"] == sw})
        subprocess.run([PY, os.path.join(ROOT, "post_run.py")] + files, cwd=ROOT, stdout=open(os.path.join(H, "logs", f"post_run_{pdn}_{sw}.log"), "w"), stderr=subprocess.STDOUT)
        dst = os.path.join(OUT, "post_run", f"{pdn}_{sw}"); os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(ROOT, "post_run_out", "summary_*")): shutil.move(f, os.path.join(dst, os.path.basename(f)))
        print("post_run done:", pdn, sw)
