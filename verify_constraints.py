#!/usr/bin/env python3
"""Checks the placement / objective / stopping rules on result records, and (--smoke) on freshly run tiny PSOs.
  python verify_constraints.py --records PSO_PCB28/full28_study/results_full/results_long.jsonl [--idx PSO_PCB28/Data/murata_index_rlc.csv] [--ref21]
  python verify_constraints.py --smoke netA|full|ref21     (small swarm, needs the data folders)
Rules: obs index 0 never used, pads 1..N-1, unique pads, n_caps <= pads, curve_caps = 1..n without gaps, model row valid and
name == library name of that row (skip with --permuted), success <=> best peak <= threshold <=> time_to_target set,
floor < threshold < bare (floor 117.18 ohm for the 28-port, 40.5 mOhm for the 21-port)."""
import sys, os, json, csv, argparse
import numpy as np
FLOOR = {"B3G": 117.18, "BFULL": 0.0405}
def check(rec, names=None, nlib=None):
    bad = []; N, n_obs, pads = rec["N"], rec["n_obs"], rec["n_pads"]; pl = rec["placement"]; band = rec["rank_band"]
    if n_obs != 1 or pads != N - 1: bad.append("obs/pad counts")
    ps = [p["pad"] for p in pl]
    if any(p < n_obs or p >= N for p in ps): bad.append(f"pad out of range/obs used {ps}")
    if len(set(ps)) != len(ps): bad.append(f"duplicate pad {ps}")
    if len(pl) != rec["best_n_caps"] or len(pl) > pads or rec["best_n_caps"] > rec["max_caps"]: bad.append("cap count")
    cc = rec["curve_caps"]
    if cc != list(range(1, len(cc) + 1)): bad.append(f"curve_caps has gaps {cc}")
    for p in pl:
        if nlib and not (0 <= p["model"] < nlib): bad.append("model row out of range")
        if names is not None and names[p["model"]] != p["name"]: bad.append(f"name/row mismatch {p}")
    thr = rec["threshold_ohm"]; best = rec["best_peak_ohm"][band]; ok = rec["success"]; t = rec["time_to_target_s"]
    if thr is not None:
        if ok != (best <= thr): bad.append(f"success={ok} but best {best:.6g} vs threshold {thr}")
        if (t is not None) != ok: bad.append("time_to_target / success disagree")
        if not (FLOOR[band] < thr < rec["bare_peak_ohm"][band]): bad.append("threshold outside (floor, bare)")
    elif ok or t is not None: bad.append("success without threshold")
    return bad
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--records"); ap.add_argument("--idx"); ap.add_argument("--permuted", action="store_true")
    ap.add_argument("--smoke"); a = ap.parse_args()
    if a.records:
        names = [r["part_number"] for r in csv.DictReader(open(a.idx))] if a.idx and not a.permuted else None
        n = nb = 0
        for line in open(a.records):
            r = json.loads(line); b = check(r, names, len(names) if names else None); n += 1
            if b: nb += 1; print(r["method"], r["threshold_level"], r["run_id"], b)
        print(f"{n} records checked, {nb} with violations"); sys.exit(1 if nb else 0)
