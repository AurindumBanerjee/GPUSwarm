"""
Floor pass: full-budget PSO runs that supply the per-band Exp 1 floors Exp 1 itself cannot.

Per-band targets need, for every (key, band), the best peak that full-budget runs OPTIMISED ON THAT
BAND reach. Exp 1 has those for B2 and B3 everywhere, but its B1 runs stopped after 1-7 capacitors
(or 0 for ddr_full: no stage-2 B1 run) because they met Exp 1's old assumed-voltage target; a peak at
the moment of stopping is not a floor. `derive_targets.py` reports which (key, band) pairs lack valid
Exp 1 runs; this script runs exactly those, with Exp 2's code, the Exp 1 stage-2 PSO budget, and NO
early stop (stop_level "none": the whole capacitor budget 1..max_caps).

    python floor_pass.py --plan       # list the (key, band) pairs that need runs, run nothing
    python floor_pass.py              # run them (resumable)

Writes only floor_pass/floor_runs.jsonl (under DecapStudy2/). The runs read no targets: dummy
targets are injected in memory so the shared PSO code can be reused, and only the achieved
peaks of the run's own band are used downstream.
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import study_config as C                                # noqa: E402
import study_core as S                                  # noqa: E402
import derive_targets as D                              # noqa: E402

C._inside(C.FLOOR_PASS_DIR)


def _dummy_targets():
    """In-memory placeholder targets so Problem() can be built before targets.json exists.
    Never written anywhere; the floor runs use only achieved peaks."""
    keys = [r[0] for r in C.MPHY_RAILS] + list(C.DDR_TARGET_KEYS)
    levels = [("1", 1.0)]
    tables = {b: {k: {"1": 1.0} for k in keys} for b in C.BANDS}
    S._CACHE["targets"] = (levels, tables)
    S._CACHE["flags"] = {}


def _problem_key(key):
    return key if key in C.DDR_TARGET_KEYS else f"mphy_{key}"


def load_done():
    done = set()
    if os.path.exists(C.FLOOR_PASS_FILE):
        with open(C.FLOOR_PASS_FILE) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["key"], r["band"], r["run_id"]))
    return done


def needed(exp1_path):
    """[(key, band)] with no valid full-budget numpy Exp 1 run optimised on that band."""
    runs = D.load_runs(exp1_path, 2)
    keys = [r[0] for r in C.MPHY_RAILS] + C.DDR_TARGET_KEYS
    out = []
    for band in C.BANDS:
        elig, _, _ = D.eligible(runs, band)
        out += [(k, band) for k in keys if k not in elig]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="list what would run, run nothing")
    ap.add_argument("--exp1-results", default=C.EXP1_RESULTS)
    args = ap.parse_args()

    todo = needed(args.exp1_results)
    print("(key, band) pairs without a valid full-budget Exp 1 run optimised on that band:")
    for k, b in todo:
        n = C.FLOOR_PASS_RUNS_BY_KEY.get(k, C.FLOOR_PASS_RUNS)
        print(f"  {k:<9} {b}   {n} runs")
    if not todo or args.plan:
        return
    os.makedirs(C.FLOOR_PASS_DIR, exist_ok=True)
    _dummy_targets()
    done = load_done()
    cfg_base = dict(C.STAGES[2])
    cfg_base.pop("runs")
    cfg_base["stop_level"] = "none"
    # small problems first, ddr_full last
    todo.sort(key=lambda kb: (kb[0] == "ddr_full", kb[0] == "ddr21"))
    for key, band in todo:
        prob = S.build_problem(_problem_key(key))
        n = C.FLOOR_PASS_RUNS_BY_KEY.get(key, C.FLOOR_PASS_RUNS)
        for run_id in range(1, n + 1):
            if (key, band, run_id) in done:
                continue
            t = time.perf_counter()
            res = S.run_pso(prob, band, run_id, dict(cfg_base))
            peak = res["rescored"][band]["peaks_ohm"][0]
            rec = {"record_type": "floor_run", "key": key, "band": band, "run_id": run_id,
                   "n_particles": cfg_base["n_particles"], "n_iters": cfg_base["n_iters"],
                   "method": "numpy", "problem": prob.name, "N": prob.N, "n_pads": len(prob.pads),
                   "max_caps": res["max_caps"], "caps_used": res["caps_used"],
                   "curve_caps": res["curve_caps"], "curve_cost": res["curve_cost"],
                   "peak_ohm": peak, "n_evals": res["n_evals"], "wall_s": res["wall_s"],
                   "placement": res["placement"], "rescored": res["rescored"],
                   "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with open(C.FLOOR_PASS_FILE, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{time.strftime('%H:%M:%S')} floor {key} {band} run{run_id}/{n}: "
                  f"peak={peak:.5g} ohm caps_used={res['caps_used']}/{res['max_caps']} "
                  f"evals={res['n_evals']} {time.perf_counter() - t:.0f}s", flush=True)
    print("floor pass done")


if __name__ == "__main__":
    main()
