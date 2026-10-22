#!/usr/bin/env python3
"""21-port reference PDN (Data/y2.mat, port 0 = observation, 20 pads, 100.99 Hz - 99.83 MHz): legacy decap library vs the
new Murata library, time-to-target study with thresholds 60, 50, 45 mOhm (floor 40.5 mOhm: the paper's 40/30 mOhm are unreachable) + a full-budget 'floor' run.

  LIB=ref     Data/decaps.mat (3348 legacy models, admittance)           names: ref_<row>
  LIB=murata  Data/murata_decaps_ref21.mat (16171 Murata parts, same grid; series-RLC low-side extrapolation)
  LIB=both    (default) both arms, one after the other, SAME seeds -> paired comparison

  python pso_ref21.py --validate
  NUM_RUNS=10 FLOOR_RUNS=3 python pso_ref21.py
env: LIB, NUM_RUNS (10), FLOOR_RUNS (3; 0 = skip), TARGETS (0.06,0.05,0.045), FLOOR_OHM (0.0405), METHODS (numpy), STUDY_THREADS (1),
     N_PARTICLES/N_ITERATIONS (default 50/15, SB.py's values), OUT, DATA
Objective: one band, peak |Z11| over the whole grid."""
import os, sys, csv, json
os.environ["RANK_BAND"] = "BFULL"
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import numpy as np, scipy.io as sio
import pso_sp_core as core
from concurrent.futures import ThreadPoolExecutor
core.N_PARTICLES = int(os.environ.get("N_PARTICLES", 50)); core.N_ITERATIONS = int(os.environ.get("N_ITERATIONS", 15))   # SB.py values
core.BANDS.clear(); core.BANDS["BFULL"] = None; core.EXTRAP_START_HZ = 1e9
DATA = os.environ.get("DATA", os.path.join(H, "Data")); OUT = os.environ.get("OUT", os.path.join(H, "results"))
FLOOR_OHM = float(os.environ.get("FLOOR_OHM", 0.0405))   # lowest peak |Z11| reached on the 21-port PDN (~40.5 mOhm); targets must lie above it
TARGETS = [float(x) for x in os.environ.get("TARGETS", "0.06,0.05,0.045").split(",")]
NUM_RUNS = int(os.environ.get("NUM_RUNS", "10")); FLOOR_RUNS = int(os.environ.get("FLOOR_RUNS", "3"))
METHODS = os.environ.get("METHODS", "numpy").split(","); LIB = os.environ.get("LIB", "both")

def load(lib):
    f = sio.loadmat(os.path.join(DATA, "freq2.mat"))["freq"].ravel()
    if lib == "ref":
        d = sio.loadmat(os.path.join(DATA, "decaps.mat"))["decaps"]; names = [f"ref_{i}" for i in range(d.shape[0])]
    else:
        m = sio.loadmat(os.path.join(DATA, "murata_decaps_ref21.mat")); d = m["decaps"]; assert np.allclose(m["freq"].ravel(), f, rtol=1e-9)
        names = [r["part_number"] for r in csv.DictReader(open(os.path.join(DATA, "murata_index_ref21.csv")))]
    assert d.shape[1] == len(f) and len(names) == d.shape[0] and np.isfinite(d).all()
    core._lib.clear(); core._lib.update(d=d, names=names, slope=np.ones(d.shape[0]), freq=f, lists=None)
core.library = lambda: core._lib

if __name__ == "__main__":
    validate = "--validate" in sys.argv; libs = ["ref", "murata"] if LIB == "both" else [LIB]
    pool = ThreadPoolExecutor(core.THREADS) if core.THREADS > 1 else None
    for lib in libs:
        load(lib); prob = core.Problem("ref21", os.path.join(DATA, "y2.mat"), n_obs=1, obs_labels=["port0"])
        core.selftest(prob, METHODS); bare = float(np.max(core.bare_peaks(prob)["BFULL"]))
        d = core._lib["d"]
        print(f"[{lib}] library {d.shape}, passive: {bool(d.real.min() >= -1e-12*np.abs(d).max())}, bare peak {bare:.5g} ohm, N={prob.N} pads={len(prob.pads)}")
        assert all(FLOOR_OHM < t < bare for t in TARGETS), f"a target is outside (floor {FLOOR_OHM}, bare {bare}) -- set FLOOR_OHM to override"
        if validate: continue
        out = os.path.join(OUT, lib); res = os.path.join(out, "results_long.jsonl"); os.makedirs(out, exist_ok=True)
        jobs = [(f"T{i+1}", t, NUM_RUNS) for i, t in enumerate(TARGETS)] + ([("floor", None, FLOOR_RUNS)] if FLOOR_RUNS else [])
        for meth in METHODS:
            for lvl, thr, nr in jobs:
                for run in range(1, nr + 1):
                    folder = os.path.join(out, prob.name, meth, lvl, f"run_{run}")
                    print(f"[{lib}] {meth} {lvl} thr={thr} run {run}/{nr}", flush=True)
                    try:
                        r = core.run_pso(prob, meth, run, folder, res, pool=pool, out_root=out, threshold=thr, level=lvl)
                        print(f"    best {r['best_peak_ohm']['BFULL']:.5g} ohm @ {r['best_n_caps']} caps, {r['total_wall_time_s']:.1f}s, "
                              + ("hit in %.1fs" % r["time_to_target_s"] if r["success"] else ("target missed" if thr else "floor run")), flush=True)
                    except Exception as exc: print("    [ERROR]", exc, flush=True)
    if validate: print("--validate: OK")
