"""
Runs the whole study sequentially: each experiment starts by itself when the
previous one finishes.

    rail responsiveness -> Stage 1 (screening) -> Stage 2 (survivors) ->
    Grid 2 (winning band) -> consistency check -> plots + report

No input is needed between steps: Stage 2 starts by itself when Stage 1 ends.
The Stage 1 table is still written (stage1_table.md / .csv, survivors.json) and
echoed to the log the moment Stage 1 finishes, so it can be read while Stage 2
runs; pass --pause-after-stage1 to stop there instead. Everything is
resumable: finished runs are read back from results_long.jsonl and skipped, so
after a crash or a kill just launch the same command again.

    nohup python run_all.py > run_all.out 2>&1 &
    STUDY_THREADS=16 nohup python run_all.py > run_all.out 2>&1 &
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause-after-stage1", action="store_true",
                    help="stop after writing the Stage 1 table instead of going straight to Stage 2")
    ap.add_argument("--only", choices=["response", "stage1", "stage2", "grid2",
                                       "consistency", "report"])
    ap.add_argument("--care-band", choices=["B1", "B2", "B3"],
                    help="band used to rank cells (default from study_config / CARE_BAND)")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop a run after this many capacitor counts without >0.1%% gain")
    ap.add_argument("--grid2-band", choices=["B1", "B2", "B3"],
                    help="override the winning band for Grid 2")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets, 2 runs, <=3 caps: checks the pipeline end to end")
    args = ap.parse_args()

    if args.care_band:
        os.environ["CARE_BAND"] = args.care_band
    import study_config as C
    if args.care_band:
        C.CARE_BAND = args.care_band
    if args.patience:
        C.PATIENCE = args.patience
    import experiments as X
    import analyze as A

    ov = {}
    if args.smoke:
        ov = {"n_particles": 6, "n_iters": 2, "max_caps": 3, "runs": 2}
        C.GRID2_RUNS, C.GRID2_PURE_RUNS = 1, 1
        C.EVAL_TIMING_N, C.EVAL_TIMING_CAPS = 3, 3
        C.GRID2_PSO = dict(n_particles=6, n_iters=2, max_caps=3)
        C.GRID2_TIME_LIMIT_S = 120
    want = lambda name: args.only in (None, name)

    if want("response"):
        X.rail_response(stride=50 if args.smoke else 7)

    if want("stage1"):
        X.run_grid1(1, X.all_cells(), ov)
        text = A.write_stage1(C.CARE_BAND)
        if text:
            X.log("Stage 1 screening table:\n" + text)
        if args.only is None and args.pause_after_stage1:
            X.log("Stage 1 done (--pause-after-stage1). Re-run without the flag for Stage 2.")
            return

    if want("stage2"):
        cells = A.survivors()
        if cells is None:
            A.write_stage1(C.CARE_BAND)
            cells = A.survivors()
        if not cells:
            X.log("No Stage 1 survivors -- run Stage 1 first."); return
        X.run_grid1(2, cells, ov)

    wb = A.winning_band(C.CARE_BAND)
    band = args.grid2_band or (wb[0] if wb else None)
    if want("grid2"):
        if band is None:
            X.log("No Grid 1 data -- cannot pick a band for Grid 2."); return
        X.log(f"Winning band for Grid 2: {band}")
        X.run_grid2(band, {k: v for k, v in ov.items() if k in ("max_caps",)})

    if want("consistency"):
        for b in C.BANDS:
            X.consistency_check(b)

    if want("report"):
        A.make_plots()
        A.build_report()
        X.log(f"Report: {os.path.join(C.OUT_ROOT, 'report.md')}")


if __name__ == "__main__":
    main()
