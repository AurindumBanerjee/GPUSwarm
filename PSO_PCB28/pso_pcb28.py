#!/usr/bin/env python3
"""
PSO decap placement - PCB 28-port PDN (Dataset/PCB_27ports), 100 Hz - 3 GHz.

Reuses GPUSwarm/PSO_SP/pso_sp_core.py unchanged; only the data hooks are replaced:
  * library:  LIB=murata (default)  Data/murata_decaps_pcb_rlc.mat
                                     (stored as admittance Y in S, which the core adds to the diagonal)
  * bands:    B20 (<=20 MHz), B100 (<=100 MHz), B1G (<=1 GHz), B3G (whole grid)
  * board:    y_pcb_netA = port 1 (observation) + the 11 pads on its net, which is exact
              for Z11 (checked against the full 28-port network in --validate)

env: PCB_DATA (input folder, default ./Data), MURATA_MAT, MURATA_IDX, LIB, RANK_BAND (default B3G),
     NUM_RUNS, STUDY_THREADS, TIME_LIMIT_S, OUT
  python pso_pcb28.py --validate
  NUM_RUNS=3 STUDY_THREADS=2 python pso_pcb28.py
"""
import os, sys, csv
os.environ.setdefault("RANK_BAND", "B3G")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np, scipy.io as sio
import pso_sp_core as core

DATA = os.environ.get("PCB_DATA", os.path.join(H, "Data"))   # self-contained: all inputs live in ./Data
BOARD = DATA
LIB = os.environ.get("LIB", "murata")
NETA_PORTS = [1, 3, 5, 6, 10, 11, 12, 18, 19, 22, 23, 25]        # 1-based original port numbers

core.BANDS.clear(); core.BANDS.update({"B20": 20e6, "B100": 100e6, "B1G": 1e9, "B3G": None})
core.EXTRAP_START_HZ = 1e9      # library is measured to >=1 GHz for every part; beyond that only 373 parts extrapolate

def load_library():
    f = sio.loadmat(os.path.join(BOARD, "freq_pcb.mat"))["freq"].ravel()
    if LIB == "murata":
        mat = os.environ.get("MURATA_MAT", os.path.join(DATA, "murata_decaps_pcb_rlc.mat"))
        idx = os.environ.get("MURATA_IDX", os.path.join(DATA, "murata_index_rlc.csv"))
        m = sio.loadmat(mat); d = m["decaps"]; lf = m["freq"].ravel()   # stored as admittance Y (S), as the core expects
        names = [r["part_number"] for r in csv.DictReader(open(idx))]
    assert d.shape[1] == len(f) == len(lf) and np.allclose(f, lf, rtol=1e-9), "library grid != PDN grid"
    assert len(names) == d.shape[0]
    core._lib.update(d=d, names=names, slope=np.ones(d.shape[0]), freq=f, lists=None)
core.library = lambda: core._lib if core._lib else (load_library() or core._lib)

PROBLEMS = [dict(name="pcb28_netA", path=os.path.join(BOARD, "y_pcb_netA.mat"), n_obs=1,
                 obs_labels=["port1"])]

if __name__ == "__main__":
    core.library()
    out = os.environ.get("OUT", os.path.join(H, f"results_{LIB}"))
    if "--validate" in sys.argv:
        d = core._lib["d"]; f = core._lib["freq"]
        print("library", d.shape, d.dtype, f"{f[0]:.2f}-{f[-1]:.6g} Hz; finite:", bool(np.isfinite(d).all()))
        print("min Re(Y) over all models/freqs (passivity, should be >= 0):", float(d.real.min()))
        print("models with any Re(Y)<-1e-12*|Y|:", int((d.real < -1e-12*np.abs(d)).any(axis=1).sum()))
        print("|Y| range [S]:", float(np.abs(d).min()), float(np.abs(d).max()))
        y28 = np.transpose(sio.loadmat(os.path.join(BOARD, "y_pcb28.mat"))["y"], (2, 0, 1))
        yA = np.transpose(sio.loadmat(os.path.join(BOARD, "y_pcb_netA.mat"))["y"], (2, 0, 1))
        sel = [p-1 for p in NETA_PORTS]
        # random placement on net A pads: full-network vs reduced-network Z11
        rng = np.random.default_rng(3); pads = list(range(1, 12))
        cfg = [(int(rng.integers(0, d.shape[0])), int(p)) for p in rng.choice(pads, 6, replace=False)]
        A = yA.copy(); B = y28.copy()
        for c, p in cfg: A[:, p, p] += d[c]; B[:, sel[p], sel[p]] += d[c]
        zA = np.abs(np.linalg.inv(A)[:, 0, 0]); zB = np.abs(np.linalg.inv(B)[:, 0, 0])
        print("netA vs full-28 Z11 with 6 random decaps, max rel diff:", float(np.max(np.abs(zA-zB)/zB)))
    core.main("PCB28", PROBLEMS, out)
