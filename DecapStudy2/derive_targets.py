"""
Derive Exp 2 targets from Exp 1's results (READ-ONLY on Exp 1).

For every rail (mer1 ... mphyvdd, from the rail-wise A1 runs) and for each DDR3
PDN size (ddr21 from A4, ddr_full from A5), the *floor* is the lowest peak
|Z11| on the <=50 MHz band (B2) that any Exp 1 run reached while using its FULL
capacitor budget. Each multiplier m then gives one threshold level:

    target_ohm[key][m] = floor_ohm[key] * m

Written to targets.json next to this script (the only file this script writes).

A run counts as "full budget" when (a) its budget was the whole pad count
(max_caps == n_pads, i.e. not a reduced smoke budget) and (b) its search actually
went through the last capacitor count (curve_caps[-1] == max_caps) instead of
stopping early on a target or a time limit.

    python derive_targets.py                              # defaults
    python derive_targets.py --multipliers 1.05 1.2 1.4
    python derive_targets.py --floor-source any           # any optimisation band, scored on B2
    python derive_targets.py --exp1-results PATH          # Exp 1 results_long.jsonl
"""
import os
import sys
import json
import argparse
import time
from collections import defaultdict

import study_config as C

SCORE_BAND = "B2"            # the <=50 MHz band the floors are measured on


def level_key(m):
    return f"{m:g}"


def target_key(rec):
    """(target key, observation index) for an Exp 1 record, or None."""
    sc = rec["scope"]
    if sc == "A1":
        return rec["rail"], 0               # rail-wise run: one observation port
    if sc == "A4":
        return "ddr21", 0
    if sc == "A5":
        return "ddr_full", 0
    return None


def load_runs(path, stage):
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record_type") == "run" and r.get("grid") != "G2" and r["stage"] == stage:
                runs.append(r)
    return runs


def derive(runs, floor_source):
    floors, prov = {}, {}
    seen = defaultdict(int)
    skipped = defaultdict(lambda: defaultdict(int))
    for r in runs:
        tk = target_key(r)
        if tk is None:
            continue
        key, k = tk
        seen[key] += 1
        if floor_source == "b2" and r["band"] != SCORE_BAND:
            skipped[key]["other optimisation band"] += 1
            continue
        if r["n_pads"] != r["max_caps"]:
            skipped[key]["reduced budget"] += 1
            continue
        if not r["curve_caps"] or r["curve_caps"][-1] != r["max_caps"]:
            skipped[key]["stopped before the full budget"] += 1
            continue
        peak = r["rescored"][SCORE_BAND]["peaks_ohm"][k]
        if key not in floors or peak < floors[key]:
            floors[key] = peak
            prov[key] = {"scope": r["scope"], "optimised_band": r["band"], "run_id": r["run_id"],
                         "stage": r["stage"], "max_caps": r["max_caps"], "caps_used": r["caps_used"]}
    for key in prov:
        prov[key]["n_records_seen"] = seen[key]
        prov[key]["n_eligible"] = seen[key] - sum(skipped[key].values())
    return floors, prov, seen, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1-results", default=C.EXP1_RESULTS,
                    help="Exp 1 results_long.jsonl (read only)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=C.DEFAULT_MULTIPLIERS)
    ap.add_argument("--floor-source", choices=["b2", "any"], default="b2",
                    help="b2: only runs optimised on B2 (default); any: every optimisation "
                         "band, re-scored on B2")
    ap.add_argument("--stage", type=int, default=2, help="Exp 1 stage to read (default 2 = full budget runs)")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="derive even though Exp 1 has no report.md yet")
    ap.add_argument("--force", action="store_true", help="overwrite a targets.json that is already filled")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    mult = sorted(set(args.multipliers))
    if len(mult) != len(args.multipliers):
        sys.exit("duplicate multipliers")
    if any(m <= 0 for m in mult):
        sys.exit("multipliers must be > 0")
    if any(m < 1 for m in mult):
        print("[warn] a multiplier below 1 is a target below the best Exp 1 result", file=sys.stderr)

    src = os.path.abspath(args.exp1_results)
    if not os.path.exists(src):
        sys.exit(f"Exp 1 results not found: {src}")
    report = os.path.join(os.path.dirname(src), "report.md")
    if not os.path.exists(report) and not args.allow_unfinished:
        sys.exit(f"Exp 1 does not look finished ({report} is missing). Targets are meant to be "
                 f"derived after Exp 1 ends; use --allow-unfinished to override.")

    if os.path.exists(C.TARGETS_FILE) and not args.force and not args.dry_run:
        cur = json.load(open(C.TARGETS_FILE))
        if any(v is not None for v in cur.get("floors_ohm", {}).values()):
            sys.exit("targets.json is already filled; pass --force to overwrite")

    runs = load_runs(src, args.stage)
    floors, prov, seen, skipped = derive(runs, args.floor_source)

    keys = [r[0] for r in C.MPHY_RAILS] + C.DDR_TARGET_KEYS
    missing = [k for k in keys if k not in floors]
    if missing:
        msg = ["no eligible Exp 1 run for: " + ", ".join(missing)]
        for k in missing:
            msg.append(f"  {k}: {seen.get(k, 0)} records seen, skipped {dict(skipped.get(k, {}))}")
        sys.exit("\n".join(msg))

    levels = [level_key(m) for m in mult]
    out = {
        "_comment": "Derived by derive_targets.py from Exp 1. targets_ohm[key][level] = "
                    "floors_ohm[key] * level, in ohm of peak |Z_kk| on the <=50 MHz band.",
        "multipliers": mult,
        "floors_ohm": {k: floors[k] for k in keys},
        "targets_ohm": {k: {lv: floors[k] * m for lv, m in zip(levels, mult)} for k in keys},
        "source": {
            "exp1_results": src, "score_band": SCORE_BAND, "stage": args.stage,
            "floor_source": args.floor_source,
            "derived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": {k: prov[k] for k in keys},
        },
    }

    print(f"{'key':<10} {'floor [ohm]':>12}  " + "  ".join(f"x{lv:<6}" for lv in levels) + "  from")
    for k in keys:
        p = prov[k]
        print(f"{k:<10} {floors[k]:>12.5g}  "
              + "  ".join(f"{out['targets_ohm'][k][lv]:<7.4g}" for lv in levels)
              + f"  {p['scope']}/{p['optimised_band']}/run{p['run_id']} "
                f"({p['n_eligible']}/{p['n_records_seen']} runs eligible)")
    if args.dry_run:
        print("dry run: nothing written")
        return
    tmp = C.TARGETS_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, C.TARGETS_FILE)
    print(f"wrote {C.TARGETS_FILE}")


if __name__ == "__main__":
    main()
