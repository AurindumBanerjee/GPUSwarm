"""
Derive Exp 2 targets from Exp 1's results (READ-ONLY on Exp 1), SEPARATELY FOR EACH BAND.

For every rail (mer1 ... mphyvdd, from the rail-wise A1 runs) and each DDR3 PDN size (ddr21 from A4,
ddr_full from A5), and for each band b in B1/B2/B3: every numpy run optimised ON b that used its FULL
capacitor budget contributes one achieved peak |Z11| on band b (the run's own re-scored peak). The
*floor* of (key, b) is anchored on that distribution:

    spread = (max - min) / min * 100 %
    spread <= threshold (10 %)  ->  floor = min      (the best any run reached)
    spread  > threshold         ->  floor = median   (the best run is an outlier)

(--anchor min|median overrides.) Each multiplier m then gives one threshold level of band b:

    target_ohm[b][key][m] = floor_ohm[b][key] * m

so a B1 cell is scored against a B1-derived target, a B2 cell against a B2-derived one, and so on.
Noise flags are computed per band: a level whose target lies BELOW the median achieved peak of that
band's runs is "inside the floor spread" (a typical run could not reach it).

A run counts as "full budget" when (a) its budget was the whole pad count (max_caps == n_pads) and (b)
its search went through the last capacitor count (curve_caps[-1] == max_caps) instead of stopping early
on a target or a time limit. Exp 1's B1 runs for mer2 / pll1v0 / mphyvdd / ddr21 stopped after 1-7
capacitors on Exp 1's old target (and ddr_full has no stage-2 B1 run), so they are NOT floors: those
(key, band) pairs are filled from floor_pass.py (floor_pass/floor_runs.jsonl: full budget, no early
stop, own code). Each band's source per key is recorded in targets.json ("exp1" | "floor_pass").
Only numpy runs feed the floors (an explicit filter; the approximate `iterative` method exists only in
Exp 1's Grid 2, which is never read here).

The report printed before anything is written has, per band, the floor distribution and a probe of
how many capacitors the runs needed to get within m x floor.

Nothing is written with --dry-run; targets.json is the only file this script writes.

    python derive_targets.py --dry-run                      # report only
    python derive_targets.py --force                        # (re)write targets.json
    python derive_targets.py --anchor min                   # no median switching
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


def load_floor_runs(path):
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("record_type") == "floor_run":
                        out.append(r)
    return out


def eligible(runs, band):
    """key -> [{peak, caps_used, max_caps, curve_caps, curve_peak, rec}] for every full-budget Exp 1
    numpy run optimised on `band` (peak = its own re-scored peak on `band`); plus bookkeeping."""
    elig = defaultdict(list)
    seen = defaultdict(int)
    skipped = defaultdict(lambda: defaultdict(int))
    for r in runs:
        tk = target_key(r)
        if tk is None or r["band"] != band:
            continue
        key, k = tk
        seen[key] += 1
        if r.get("method", "numpy") != "numpy":          # floors must come from exact numpy runs only
            skipped[key]["non-numpy method"] += 1
            continue
        if r["n_pads"] != r["max_caps"]:
            skipped[key]["reduced budget"] += 1
            continue
        if not r["curve_caps"] or r["curve_caps"][-1] != r["max_caps"]:
            skipped[key]["stopped before the full budget"] += 1
            continue
        t0 = r["targets_ohm"][0]            # Exp 1 curve_ratio is relative to this scalar target
        elig[key].append({"peak": r["rescored"][band]["peaks_ohm"][k],
                          "caps_used": r["caps_used"], "max_caps": r["max_caps"],
                          "curve_caps": r["curve_caps"], "curve_peak": [c * t0 for c in r["curve_ratio"]],
                          "run": {"scope": r["scope"], "optimised_band": r["band"],
                                  "run_id": r["run_id"], "stage": r["stage"]}})
    return elig, seen, skipped


def eligible_floor(frs, band):
    """Same shape as eligible(), from floor_pass.py records (full budget, no early stop)."""
    elig = defaultdict(list)
    for r in frs:
        if r["band"] != band or r.get("method", "numpy") != "numpy":
            continue
        if r["n_pads"] != r["max_caps"] or not r["curve_caps"] or r["curve_caps"][-1] != r["max_caps"]:
            continue
        elig[r["key"]].append({"peak": r["peak_ohm"], "caps_used": r["caps_used"],
                               "max_caps": r["max_caps"], "curve_caps": r["curve_caps"],
                               "curve_peak": r["curve_cost"],
                               "run": {"scope": "floor_pass", "optimised_band": r["band"],
                                       "run_id": r["run_id"], "stage": 2}})
    return elig


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
            "min_run": best["run"],
        }
    return out


def noise_flags(stats, mult):
    """key -> level label -> True when floor*m < median achieved peak (strictly below)."""
    return {k: {level_key(m): bool(s["floor"] * m < s["median"]) for m in mult}
            for k, s in stats.items()}


def probe(elig, stats, multipliers):
    """key -> m -> (median capacitors needed, runs that got there, runs): the first capacitor count whose
    best own-band peak is within m x floor."""
    out = {}
    for key, lst in elig.items():
        row = {}
        for m in multipliers:
            need = []
            for x in lst:
                for n, pk in zip(x["curve_caps"], x["curve_peak"]):
                    if pk <= m * stats[key]["floor"]:
                        need.append(n)
                        break
            row[m] = (statistics.median(need) if need else None, len(need), len(lst))
        out[key] = row
    return out


def report(band, keys, stats, srcs, prb, mult, multipliers_probe, flags):
    print(f"\n==================== band {band} ({C.BAND_LABEL[band]}) ====================")
    print(f"[1] achieved floor per key: peak |Z11| on band {band} of every full-budget numpy run optimised "
          f"on {band}")
    print(f"    spread = (max-min)/min; anchor = median where spread > {SPREAD_THRESHOLD_PCT:g} %")
    print(f"{'key':<9}{'source':<11}{'runs':>5}{'min':>11}{'median':>11}{'max':>11}{'spread%':>9}  "
          f"{'anchor':<7}{'floor [ohm]':>12}")
    for k in keys:
        s = stats[k]
        print(f"{k:<9}{srcs[k]:<11}{s['n_runs']:>5}{s['min']:>11.5g}{s['median']:>11.5g}{s['max']:>11.5g}"
              f"{s['spread_pct']:>9.1f}  {s['anchor']:<7}{s['floor']:>12.5g}")
    print(f"[2] capacitors used by those runs (budget = pads)")
    print(f"{'key':<9}{'budget':>7}{'floor run':>10}{'min':>5}{'med':>6}{'max':>5}{'runs at full budget':>22}")
    for k in keys:
        s = stats[k]
        print(f"{k:<9}{s['max_caps']:>7}{s['caps_at_min_run']:>10}{s['caps_min']:>5}"
              f"{s['caps_median']:>6.4g}{s['caps_max']:>5}"
              f"{str(s['n_at_full_budget']) + '/' + str(s['n_runs']):>22}")
    print(f"[3] capacitors needed to get within m x floor (median over the runs; '-' = never; (reached/runs))")
    print(f"{'key':<9}" + "".join(f"{'x' + format(m, 'g'):>13}" for m in multipliers_probe))
    for k in keys:
        cells = []
        for m in multipliers_probe:
            med, got, n = prb[k][m]
            cells.append(f"{('-' if med is None else format(med, 'g')) + f' ({got}/{n})':>13}")
        print(f"{k:<9}" + "".join(cells))
    print(f"[4] levels that would be written for band {band}: " + ", ".join(f"x{level_key(m)}" for m in mult))
    print(f"{'key':<9}" + "".join(f"{'x' + level_key(m) + ' [ohm]':>14}" for m in mult)
          + f"{'median/floor':>14}")
    for k in keys:
        cells = ""
        for m in mult:
            v = f"{stats[k]['floor'] * m:.5g}" + ("*" if flags[k][level_key(m)] else " ")
            cells += f"{v:>14}"
        print(f"{k:<9}{cells}{stats[k]['median'] / stats[k]['floor']:>14.4f}")
    print("    * = level target below the median achieved peak of this band's runs: inside the floor "
          "spread (noise-flagged)")
    for k in keys:
        fl = [f"x{lab}" for lab, v in flags[k].items() if v]
        if fl:
            print(f"    flagged {k}: {', '.join(fl)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1-results", default=C.EXP1_RESULTS,
                    help="Exp 1 results_long.jsonl (read only)")
    ap.add_argument("--floor-pass", default=C.FLOOR_PASS_FILE,
                    help="floor_pass.py output (used only for (key, band) pairs Exp 1 cannot supply)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=C.DEFAULT_MULTIPLIERS)
    ap.add_argument("--probe-multipliers", type=float, nargs="+", default=PROBE_MULTIPLIERS,
                    help="multipliers shown in table [3] (evidence for choosing --multipliers)")
    ap.add_argument("--anchor", choices=["auto", "min", "median"], default="auto")
    ap.add_argument("--spread-threshold", type=float, default=SPREAD_THRESHOLD_PCT,
                    help="percent; with --anchor auto the floor switches from min to median above it")
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
        print("[warn] a multiplier below 1 is a target below the floor", file=sys.stderr)

    src = os.path.abspath(args.exp1_results)
    if not os.path.exists(src):
        sys.exit(f"Exp 1 results not found: {src}")
    report_md = os.path.join(os.path.dirname(src), "report.md")
    if not os.path.exists(report_md) and not args.allow_unfinished:
        sys.exit(f"Exp 1 does not look finished ({report_md} is missing). Targets are meant to be "
                 f"derived after Exp 1 ends; use --allow-unfinished to override.")

    if os.path.exists(C.TARGETS_FILE) and not args.force and not args.dry_run:
        cur = json.load(open(C.TARGETS_FILE))
        if cur.get("bands") or any(v is not None for v in cur.get("floors_ohm", {}).values()):
            sys.exit("targets.json is already filled; pass --force to overwrite")

    runs = load_runs(src, args.stage)
    frs = load_floor_runs(args.floor_pass)
    keys = [r[0] for r in C.MPHY_RAILS] + C.DDR_TARGET_KEYS
    levels = [level_key(m) for m in mult]

    bands_out = {}
    problems = []
    for band in C.BANDS:
        elig1, seen, skipped = eligible(runs, band)
        eligf = eligible_floor(frs, band)
        elig, srcs = {}, {}
        for k in keys:
            if k in elig1:
                elig[k], srcs[k] = elig1[k], "exp1"
            elif k in eligf:
                elig[k], srcs[k] = eligf[k], "floor_pass"
            else:
                problems.append(f"{band} {k}: no valid full-budget run optimised on {band} "
                                f"({seen.get(k, 0)} Exp 1 records seen, skipped {dict(skipped.get(k, {}))}) "
                                f"-> run floor_pass.py")
        if any(k not in elig for k in keys):
            continue
        stats = anchor(elig, args.anchor, args.spread_threshold)
        prb = probe(elig, stats, args.probe_multipliers)
        flags = noise_flags(stats, mult)
        report(band, keys, stats, srcs, prb, mult, args.probe_multipliers, flags)
        bands_out[band] = {
            "floors_ohm": {k: stats[k]["floor"] for k in keys},
            "targets_ohm": {k: {lv: stats[k]["floor"] * m for lv, m in zip(levels, mult)} for k in keys},
            "flags": flags,
            "floor_source": srcs,
            "per_key": {k: dict(stats[k]) for k in keys},
        }
    if problems:
        sys.exit("\n" + "\n".join(problems))
    methods = sorted({"numpy"})
    print(f"\ninversion methods feeding the floors: {', '.join(methods)}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    out = {
        "_comment": "Derived by derive_targets.py PER BAND. bands[b].targets_ohm[key][level] = "
                    "bands[b].floors_ohm[key] * level, in ohm of peak |Z_kk| on band b, where the floor "
                    "is the best peak of full-budget numpy runs optimised on b and scored on b.",
        "multipliers": mult,
        "bands": bands_out,
        "flag_rule": "bands[b].flags[key][level] is true when bands[b].targets_ohm[key][level] < the "
                     "MEDIAN achieved peak of that band's full-budget runs (the level lies inside the "
                     "floor spread of that band)",
        "source": {
            "methods_feeding_floors": methods,
            "exp1_results": src, "floor_pass": os.path.abspath(args.floor_pass),
            "stage": args.stage, "anchor_mode": args.anchor,
            "spread_threshold_pct": args.spread_threshold,
            "derived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
