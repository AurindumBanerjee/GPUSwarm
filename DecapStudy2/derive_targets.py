"""
Derive Exp 2 targets from Exp 1's results (READ-ONLY on Exp 1).

For every rail (mer1 ... mphyvdd, from the rail-wise A1 runs) and each DDR3 PDN
size (ddr21 from A4, ddr_full from A5), every Exp 1 stage-2 run that used its FULL
capacitor budget contributes one achieved peak |Z11| on the <=50 MHz band (B2).
The *floor* of a rail is anchored on that distribution:

    spread = (max - min) / min * 100 %
    spread <= threshold (10 %)  ->  floor = min      (the best any run reached)
    spread  > threshold         ->  floor = median   (the best run is an outlier)

(--anchor min|median overrides.) Each multiplier m then gives one threshold level:

    target_ohm[key][m] = floor_ohm[key] * m

A run counts as "full budget" when (a) its budget was the whole pad count
(max_caps == n_pads) and (b) its search went through the last capacitor count
(curve_caps[-1] == max_caps) instead of stopping early on a target or a time limit.

The report printed before anything is written has three tables: the floor
distribution per key (min / median / max / spread, the anchor chosen, the capacitor
count of the floor run), and a probe of how many capacitors Exp 1's runs needed to get
within m x floor for a range of m -- the evidence for choosing the multiplier list.
Nothing is written with --dry-run; targets.json is the only file this script writes.

    python derive_targets.py --dry-run                      # report only
    python derive_targets.py --multipliers 1.1 1.25 1.5     # write targets.json
    python derive_targets.py --anchor min                   # no median switching
    python derive_targets.py --floor-source any             # any optimisation band, scored on B2
    python derive_targets.py --exp1-results PATH            # Exp 1 results_long.jsonl
"""
import os
import sys
import json
import argparse
import time
import statistics
from collections import defaultdict

import study_config as C

SCORE_BAND = "B2"            # the <=50 MHz band the floors are measured on
SPREAD_THRESHOLD_PCT = 10.0
PROBE_MULTIPLIERS = [1.05, 1.1, 1.25, 1.5, 2.0, 3.0]


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


def eligible(runs, floor_source):
    """key -> [{peak, caps_used, max_caps, rec}] for every full-budget run; plus bookkeeping."""
    elig = defaultdict(list)
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
        elig[key].append({"peak": r["rescored"][SCORE_BAND]["peaks_ohm"][k],
                          "caps_used": r["caps_used"], "max_caps": r["max_caps"], "rec": r})
    return elig, seen, skipped


def anchor(elig, mode, thr_pct):
    """key -> stats incl. the anchored floor."""
    out = {}
    for key, lst in elig.items():
        peaks = sorted(x["peak"] for x in lst)
        mn, md, mx = peaks[0], statistics.median(peaks), peaks[-1]
        spread = 100.0 * (mx - mn) / mn
        chosen = mode if mode != "auto" else ("median" if spread > thr_pct else "min")
        best = min(lst, key=lambda x: x["peak"])
        caps = sorted(x["caps_used"] for x in lst)
        out[key] = {
            "n_runs": len(lst), "min": mn, "median": md, "max": mx, "spread_pct": spread,
            "anchor": chosen, "floor": md if chosen == "median" else mn,
            "max_caps": best["max_caps"], "caps_at_min_run": best["caps_used"],
            "caps_min": caps[0], "caps_median": statistics.median(caps), "caps_max": caps[-1],
            "n_at_full_budget": sum(1 for c in caps if c == best["max_caps"]),
            "min_run": {"scope": best["rec"]["scope"], "optimised_band": best["rec"]["band"],
                        "run_id": best["rec"]["run_id"], "stage": best["rec"]["stage"]},
        }
    return out


def probe(elig, stats, multipliers):
    """key -> m -> (median capacitors needed, runs that got there, runs).

    Uses each B2-optimised run's per-capacitor-count best peak (curve_ratio x the
    old single-observation target = ohm on B2): the first capacitor count whose best peak
    is within m x floor."""
    out = {}
    for key, lst in elig.items():
        runs = [x["rec"] for x in lst if x["rec"]["band"] == SCORE_BAND]
        row = {}
        for m in multipliers:
            need = []
            for r in runs:
                t = r["targets_ohm"][0]
                for n, c in zip(r["curve_caps"], r["curve_ratio"]):
                    if c * t <= m * stats[key]["floor"]:
                        need.append(n)
                        break
            row[m] = (statistics.median(need) if need else None, len(need), len(runs))
        out[key] = row
    return out


def report(keys, stats, prb, mult, multipliers_probe):
    print("\n[1] achieved floor per key: peak |Z11| on <=50 MHz (B2) of every full-budget Exp 1 run")
    print(f"    spread = (max-min)/min; anchor = median where spread > {SPREAD_THRESHOLD_PCT:g} %")
    print(f"{'key':<9}{'runs':>5}{'min':>10}{'median':>10}{'max':>10}{'spread%':>9}  {'anchor':<7}"
          f"{'floor [ohm]':>12}")
    for k in keys:
        s = stats[k]
        print(f"{k:<9}{s['n_runs']:>5}{s['min']:>10.5g}{s['median']:>10.5g}{s['max']:>10.5g}"
              f"{s['spread_pct']:>9.1f}  {s['anchor']:<7}{s['floor']:>12.5g}")
    print("\n[2] capacitors used by the runs that produced those peaks (budget = pads)")
    print(f"{'key':<9}{'budget':>7}{'floor run':>10}{'min':>5}{'med':>6}{'max':>5}"
          f"{'runs at full budget':>22}")
    for k in keys:
        s = stats[k]
        print(f"{k:<9}{s['max_caps']:>7}{s['caps_at_min_run']:>10}{s['caps_min']:>5}"
              f"{s['caps_median']:>6.4g}{s['caps_max']:>5}"
              f"{str(s['n_at_full_budget']) + '/' + str(s['n_runs']):>22}")
    print("\n[3] capacitors Exp 1 needed to get within m x floor "
          "(median over B2-optimised runs; '-' = never; (reached/runs))")
    print(f"{'key':<9}" + "".join(f"{'x' + format(m, 'g'):>13}" for m in multipliers_probe))
    for k in keys:
        cells = []
        for m in multipliers_probe:
            med, got, n = prb[k][m]
            cells.append(f"{('-' if med is None else format(med, 'g')) + f' ({got}/{n})':>13}")
        print(f"{k:<9}" + "".join(cells))
    if mult:
        print("\n[4] levels that would be written: " + ", ".join(f"x{level_key(m)}" for m in mult))
        print(f"{'key':<9}" + "".join(f"{'x' + level_key(m) + ' [ohm]':>14}" for m in mult))
        for k in keys:
            print(f"{k:<9}" + "".join(f"{stats[k]['floor'] * m:>14.5g}" for m in mult))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1-results", default=C.EXP1_RESULTS,
                    help="Exp 1 results_long.jsonl (read only)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=C.DEFAULT_MULTIPLIERS)
    ap.add_argument("--probe-multipliers", type=float, nargs="+", default=PROBE_MULTIPLIERS,
                    help="multipliers shown in table [3] (evidence for choosing --multipliers)")
    ap.add_argument("--anchor", choices=["auto", "min", "median"], default="auto")
    ap.add_argument("--spread-threshold", type=float, default=SPREAD_THRESHOLD_PCT,
                    help="percent; with --anchor auto the floor switches from min to median above it")
    ap.add_argument("--floor-source", choices=["b2", "any"], default="b2",
                    help="b2: only runs optimised on B2 (default); any: every optimisation "
                         "band, re-scored on B2")
    ap.add_argument("--stage", type=int, default=2, help="Exp 1 stage to read (default 2 = full budget runs)")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="derive even though Exp 1 has no report.md yet")
    ap.add_argument("--force", action="store_true", help="overwrite a targets.json that is already filled")
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    args = ap.parse_args()

    mult = sorted(set(args.multipliers))
    if len(mult) != len(args.multipliers):
        sys.exit("duplicate multipliers")
    if any(m <= 0 for m in mult):
        sys.exit("multipliers must be > 0")
    if any(m < 1 for m in mult):
        print("[warn] a multiplier below 1 is a target below the Exp 1 floor", file=sys.stderr)

    src = os.path.abspath(args.exp1_results)
    if not os.path.exists(src):
        sys.exit(f"Exp 1 results not found: {src}")
    report_md = os.path.join(os.path.dirname(src), "report.md")
    if not os.path.exists(report_md) and not args.allow_unfinished:
        sys.exit(f"Exp 1 does not look finished ({report_md} is missing). Targets are meant to be "
                 f"derived after Exp 1 ends; use --allow-unfinished to override.")

    if os.path.exists(C.TARGETS_FILE) and not args.force and not args.dry_run:
        cur = json.load(open(C.TARGETS_FILE))
        if any(v is not None for v in cur.get("floors_ohm", {}).values()):
            sys.exit("targets.json is already filled; pass --force to overwrite")

    runs = load_runs(src, args.stage)
    elig, seen, skipped = eligible(runs, args.floor_source)

    keys = [r[0] for r in C.MPHY_RAILS] + C.DDR_TARGET_KEYS
    missing = [k for k in keys if k not in elig]
    if missing:
        msg = ["no eligible Exp 1 run for: " + ", ".join(missing)]
        for k in missing:
            msg.append(f"  {k}: {seen.get(k, 0)} records seen, skipped {dict(skipped.get(k, {}))}")
        sys.exit("\n".join(msg))

    stats = anchor(elig, args.anchor, args.spread_threshold)
    prb = probe(elig, stats, args.probe_multipliers)
    report(keys, stats, prb, mult, args.probe_multipliers)
    switched = [k for k in keys if stats[k]["anchor"] == "median" and args.anchor == "auto"]
    print("\nrails switched from min to median anchoring (spread > "
          f"{args.spread_threshold:g} %): " + (", ".join(switched) if switched else "none"))

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    levels = [level_key(m) for m in mult]
    out = {
        "_comment": "Derived by derive_targets.py from Exp 1. targets_ohm[key][level] = "
                    "floors_ohm[key] * level, in ohm of peak |Z_kk| on the <=50 MHz band.",
        "multipliers": mult,
        "floors_ohm": {k: stats[k]["floor"] for k in keys},
        "targets_ohm": {k: {lv: stats[k]["floor"] * m for lv, m in zip(levels, mult)} for k in keys},
        "source": {
            "exp1_results": src, "score_band": SCORE_BAND, "stage": args.stage,
            "floor_source": args.floor_source, "anchor_mode": args.anchor,
            "spread_threshold_pct": args.spread_threshold,
            "derived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_key": {k: {kk: vv for kk, vv in stats[k].items()} for k in keys},
        },
    }
    tmp = C.TARGETS_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, C.TARGETS_FILE)
    print(f"\nwrote {C.TARGETS_FILE}")


if __name__ == "__main__":
    main()
