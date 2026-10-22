#!/usr/bin/env python3
"""Study 2 (allowance 27) vs study 1 (allowance 14), run by run. Writes compare_runs.csv and compare_report.txt.
CONTROL: a run that hit within 14 caps in study 1 must reproduce (same evals, caps, peak). MISSED: study-1 misses; shows whether more capacitors reach the target and how many.
Env: PY not needed; run from anywhere."""
import os, glob, json, csv
H = os.path.dirname(os.path.abspath(__file__)); S1 = os.path.join(os.path.dirname(H), "ParallelStudy", "out")
def load(root):
    rec = {}
    for f in glob.glob(os.path.join(root, "pcb28full_*", "**", "results_long.jsonl"), recursive=True):
        if os.sep + "pure_python" + os.sep in f: continue
        sw = os.path.basename(os.path.dirname(f.split("pcb28full_")[0] + "pcb28full_" + f.split("pcb28full_")[1].split(os.sep)[0] + os.sep + "x")) if False else f.split("pcb28full_")[1].split(os.sep)[0]
        for l in open(f):
            if l.strip(): r = json.loads(l); rec[(sw, r["method"], r["threshold_level"], r["run_id"])] = r
    return rec
a, b = load(S1), load(os.path.join(H, "out")); rows = []; lines = []
for k in sorted(b):
    if k not in a: continue
    r1, r2 = a[k], b[k]; hit14 = r1["success"]
    same = (r1["n_fitness_evals"] == r2["n_fitness_evals"] and r1["best_n_caps"] == r2["best_n_caps"] and abs(r1["best_peak_ohm"]["B3G"] - r2["best_peak_ohm"]["B3G"]) < 1e-6)
    rows.append(dict(swarm=k[0], method=k[1], level=k[2], run=k[3], s1_success=r1["success"], s1_caps=r1["best_n_caps"], s1_peak=r1["best_peak_ohm"]["B3G"], s1_evals=r1["n_fitness_evals"],
                     s2_success=r2["success"], s2_caps=r2["best_n_caps"], s2_peak=r2["best_peak_ohm"]["B3G"], s2_evals=r2["n_fitness_evals"], s2_wall_s=round(r2["total_wall_time_s"]),
                     kind="control" if hit14 else "missed_in_s1", reproduced=same if hit14 else ""))
if rows:
    with open(os.path.join(H, "compare_runs.csv"), "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ctl = [r for r in rows if r["kind"] == "control"]; ms = [r for r in rows if r["kind"] != "control"]
lines.append(f"runs compared {len(rows)} | controls {len(ctl)}: reproduced {sum(1 for r in ctl if r['reproduced'])}, NOT reproduced {sum(1 for r in ctl if not r['reproduced'])}")
for r in ctl:
    if not r["reproduced"]: lines.append(f"  CONTROL MISMATCH {r['swarm']} {r['method']} {r['level']} run {r['run']}: s1 {r['s1_caps']}c/{r['s1_peak']:.4f}/{r['s1_evals']}ev vs s2 {r['s2_caps']}c/{r['s2_peak']:.4f}/{r['s2_evals']}ev")
lines.append(f"previously missed {len(ms)}: now hit {sum(1 for r in ms if r['s2_success'])}")
for r in ms: lines.append(f"  {r['swarm']} {r['method']:9} {r['level']} run {r['run']}: s1 {r['s1_peak']:.2f} ohm @ {r['s1_caps']} caps -> s2 {r['s2_peak']:.2f} ohm @ {r['s2_caps']} caps, success={r['s2_success']}, {r['s2_wall_s']} s")
open(os.path.join(H, "compare_report.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
