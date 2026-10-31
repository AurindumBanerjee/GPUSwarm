#!/usr/bin/env python3
"""Builds the data PDN 3 / PDN 4 need beyond the existing 1 GHz sets (writes only into ALLCPU_DATA, default AllCPU/data_built; nothing in Dataset/ or Data/ is touched).
  boards  (grid 5g)    MPHY + DDR3 Touchstone files -> y_5g_{mphy_full,mer1,mer2,pll1v0,pll1v8,mphyvdd,ddr_full}.mat + freq_5g.mat, clip [101 Hz, 5 GHz].
                       Same port maps, same S->Y conversion and open-circuit rail reduction (through Z) as Dataset/Scripts/build_sp_1ghz.py (only the upper clip differs).
  library (grid 5g|1g) Murata RLC library (Dataset/Decaps/Murata/murata_decaps_raw.mat, 16171 parts, measured to 3-6 GHz) resampled on the board grid with the series-RLC low-side
                       extrapolation of Dataset/Scripts/resample_murata_library_rlc.py (run as is, grid and output set through its environment hooks)
                       -> murata_<grid>_rlc.mat + murata_<grid>_rlc_index.csv
   python build_data.py --grid 5g --part all [--dataset /DATA/Aurindum/Swarming/Dataset]
   python build_data.py --grid 1g --part library          (library on the existing 1 GHz grid, for the combined-vs-Murata pilot)
   python build_data.py --grid 5g --part boards --limit-blocks 20     (smoke test: first 20 frequency points only; use a scratch ALLCPU_DATA)
Time/memory: the two Touchstone files are 0.66 and 0.77 GB of text (several minutes each); the Murata library is ~0.9 GB on the 5g grid."""
import os, sys, argparse, subprocess, time
import numpy as np, scipy.io as sio
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import allcpu_pdns as R
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="5g", choices=["1g", "5g"]); ap.add_argument("--part", default="all", choices=["boards", "library", "all"])
ap.add_argument("--dataset", default=os.environ.get("ALLCPU_DATASET", os.path.join(os.path.dirname(R.ROOT), "Swarming", "Dataset"))); ap.add_argument("--limit-blocks", type=int)
a = ap.parse_args(); FLO, FHI = 101.0, 5e9
os.makedirs(os.path.join(R.DATA_BUILT, "boards_5g"), exist_ok=True)
T0 = time.time()


def blocks(path):
    buf = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip() or line[0] in "!#": continue
            if line[0] != " ":
                if buf is not None: yield buf
                buf = line.split()
            else: buf.extend(line.split())
    if buf is not None: yield buf


def sparam(tok, N):
    x = np.asarray(tok[1:], dtype=float).reshape(N, N, 2)                       # MA format
    return float(tok[0]), x[:, :, 0] * np.exp(1j * np.deg2rad(x[:, :, 1]))


def s_to_y(S, z0=50.0):
    I = np.eye(S.shape[0]); return np.linalg.solve(I + S, I - S) / z0


JOBS = {"MPHY": dict(src=os.path.join(a.dataset, "MPHY", "SP_Sep2026", "SP_Data_MPHY_68Caps.s73p"), N=73, full="mphy_full",
                     rails={"mer1": (1, range(6, 21)), "mer2": (2, range(21, 36)), "pll1v0": (3, range(36, 50)), "pll1v8": (4, range(50, 65)), "mphyvdd": (5, range(65, 74))}),
        "DDR": dict(src=os.path.join(a.dataset, "VDDQ_DDR3", "SP_Sep2026", "SP_Data_VDDQ_DDR3_78Caps.s79p"), N=79, full="ddr_full", rails={})}
OUTB = os.path.join(R.DATA_BUILT, "boards_5g")
if a.part in ("boards", "all"):
    assert a.grid == "5g", "boards are only built for the 5g grid (the 1g sets already exist in GPUSwarm/Data)"
    fgrid = {}
    for tag, J in JOBS.items():
        N = J["N"]; assert os.path.exists(J["src"]), J["src"]
        idx = {k: np.array([o - 1] + [q - 1 for q in pads]) for k, (o, pads) in J["rails"].items()}
        freqs, Yfull, Zsub = [], [], {k: [] for k in idx}; n = 0
        for tok in blocks(J["src"]):
            fq, S = sparam(tok, N)
            if not (FLO <= fq <= FHI): continue
            Y = s_to_y(S); Z = np.linalg.inv(Y); freqs.append(fq); Yfull.append(Y)
            for k, ix in idx.items(): Zsub[k].append(Z[np.ix_(ix, ix)])
            n += 1
            if a.limit_blocks and n >= a.limit_blocks: break
        f = np.array(freqs); fgrid[tag] = f; print(f"{tag}: {len(f)} points kept, {f[0]:.4g} .. {f[-1]:.4g} Hz  [{time.time() - T0:.0f}s]", flush=True)
        for k in list(Zsub):
            Y = np.linalg.inv(np.stack(Zsub.pop(k))); sio.savemat(os.path.join(OUTB, f"y_5g_{k}.mat"), {"y": np.transpose(Y, (1, 2, 0))})
            z = np.abs(np.linalg.inv(Y)[:, 0, 0]); print(f"  y_5g_{k:8s} N={Y.shape[1]:3d} |Z11| min={z.min():.4g} peak={z.max():.4g} @ {f[z.argmax()]:.4g} Hz", flush=True)
        Yf = np.stack(Yfull); del Yfull
        sio.savemat(os.path.join(OUTB, f"y_5g_{J['full']}.mat"), {"y": np.transpose(Yf, (1, 2, 0))})
        z = np.abs(np.linalg.inv(Yf)[:, 0, 0]); print(f"  y_5g_{J['full']:8s} N={N:3d} |Z11| min={z.min():.4g} peak={z.max():.4g} @ {f[z.argmax()]:.4g} Hz", flush=True); del Yf
    assert len(fgrid["MPHY"]) == len(fgrid["DDR"]) and np.allclose(fgrid["MPHY"], fgrid["DDR"], rtol=1e-12), "MPHY and DDR Touchstone grids differ"
    sio.savemat(os.path.join(OUTB, "freq_5g.mat"), {"freq": fgrid["MPHY"].reshape(1, -1)})
if a.part in ("library", "all"):
    grid_mat = os.path.join(OUTB, "freq_5g.mat") if a.grid == "5g" else os.path.join(R.ROOT, "Data", "MPHY", "freq_sp.mat")
    assert os.path.exists(grid_mat), f"{grid_mat} missing (run --part boards first)"
    script = os.path.join(a.dataset, "Scripts", "resample_murata_library_rlc.py"); assert os.path.exists(script), script
    out_mat = os.path.join(R.DATA_BUILT, f"murata_{a.grid}_rlc.mat"); out_idx = os.path.join(R.DATA_BUILT, f"murata_{a.grid}_rlc_index.csv")
    env = dict(os.environ, MURATA_GRID_MAT=grid_mat, MURATA_OUT_MAT=out_mat, MURATA_INDEX_OUT=out_idx)
    print("resampling the Murata library on", grid_mat, flush=True); subprocess.check_call([sys.executable, script], env=env)
    m = sio.loadmat(out_mat); f = sio.loadmat(grid_mat)["freq"].ravel(); d = m["decaps"]
    assert d.shape == (16171, len(f)) and np.allclose(m["freq"].ravel(), f) and np.isfinite(d).all(), (d.shape, len(f))
    print(f"library ok: {d.shape}, min Re(Y) {float(d.real.min()):.3g}", flush=True)
print(f"done in {time.time() - T0:.0f}s")
