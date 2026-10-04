"""
Exp 2 driver. Runs the whole study sequentially: each experiment starts by
itself when the previous one finishes.

    rail responsiveness -> Stage 1 (screening) -> Stage 2 (survivors) ->
    Grid 2 (winning band) -> consistency check -> plots + report

Refuses to start (exit code 2) while any target in targets.json is null or
invalid: run derive_targets.py after Exp 1 has finished.

    python run_all.py --validate     # load data, check shapes + port rules, print targets
                                     # and the planned cells, exit (no PSO, writes nothing)
    ./run_exp2.sh                    # the real run, detached under nohup

No input is needed between steps: Stage 2 starts by itself when Stage 1 ends. The
Stage 1 table is written (results/stage1_table.md/.csv, survivors.json) and echoed to
the log when Stage 1 finishes; --pause-after-stage1 stops there instead. Everything
is resumable: finished runs are read back from results/results_long.jsonl and
skipped, so after a crash or a reboot just launch the same command again.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="load every dataset, check shapes and port rules, print the resolved "
                         "targets and the planned cell list, then exit (no PSO, no writes)")
    ap.add_argument("--pause-after-stage1", action="store_true",
                    help="stop after writing the Stage 1 table instead of going straight to Stage 2")
    ap.add_argument("--only", choices=["response", "stage1", "stage2", "grid2",
                                       "consistency", "report"])
    ap.add_argument("--care-band", choices=["B1", "B2", "B3"],
                    help="band used to rank cells (default from study_config / CARE_BAND)")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop a run after this many capacitor counts without >0.1%% gain")
    ap.add_argument("--grid2-band", choices=["B1", "B2", "B3", "B4"],
                    help="override the winning band for Grid 2")
    args = ap.parse_args()

    if args.care_band:
        os.environ["CARE_BAND"] = args.care_band
    import study_config as C           # first: pins BLAS threads before numpy is imported
    if args.care_band:
        C.CARE_BAND = args.care_band
    if args.patience:
        C.PATIENCE = args.patience

    if args.validate:
        import validate
        sys.exit(validate.run())

    import study_core as S
    try:
        S.load_targets()               # refuse to start on any null / invalid target
    except S.TargetsNotSet as e:
        sys.stderr.write(f"\nREFUSING TO START: {e}\n")
        sys.exit(2)

    import experiments as X
    import analyze as A

    want = lambda name: args.only in (None, name)

    if want("response"):
        X.rail_response()

    if want("stage1"):
        X.run_grid1(1, X.all_cells())
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
            X.log("No Stage 1 survivors -- run Stage 1 first.")
            return
        X.run_grid1(2, cells)

    wb = A.winning_band(C.CARE_BAND)
    band = args.grid2_band or (wb[0] if wb else None)
    if want("grid2"):
        if band is None:
            X.log("No Grid 1 data -- cannot pick a band for Grid 2.")
            return
        X.log(f"Winning band for Grid 2: {band}")
        X.run_grid2(band)

    if want("consistency"):
        for b in C.BANDS:
            X.consistency_check(b)

    if want("report"):
        A.make_plots()
        A.build_report()
        X.log(f"Report: {os.path.join(C.OUT_ROOT, 'report.md')}")


if __name__ == "__main__":
    main()
