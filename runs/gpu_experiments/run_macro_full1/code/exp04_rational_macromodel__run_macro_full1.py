"""
Experiment 4 -- rational macromodel on a dense / extended grid.

Dataset: VDDQ_DDR3 board 'ddr21' (21 ports, 20 pads, observation port 0) with the combined capacitor library (6695 models) on the
2588-point linear grid 110.9 Hz .. 997 MHz. The library is measured up to ~99.8 MHz and EXTRAPOLATED above (19.3 % of the grid).
Optimisation band here: 0 .. FMAX_OPT (default 300 MHz), i.e. it includes 99.8 .. 300 MHz where the tabulated capacitors are
extrapolated and where sharp PDN resonances can fall between grid points.

    PDN        Z0(f) = inv(Y(f)) (symmetric, 231 unique elements) vector-fitted with COMMON poles (Gustavsen-Semlyen VF, own numpy
               implementation: conjugate-pair real basis, pole relocation by QR-reduced LS, unstable poles flipped, constant + s*L
               terms). The pole count is increased until the max error criterion is met.
    capacitor  HYBRID: measured data (linearly interpolated between grid points) up to 99.8 MHz; above it a series R-L-C fitted on the
               top 30 % of the measured band (weights 1/|Z|^2) and scaled to be continuous at the junction. (A pure R-L-C fit of the whole
               measured band is too crude for this library: median |Z| error 10 %, 90th percentile 34 %; reported in macromodel.txt.)
               Above 99.8 MHz this replaces the tabulated log-log extrapolation by a physical R-L-C continuation.
    macro      PDN rational model + RLC capacitors evaluated on a grid RF times denser than the tabulated one.

Arms (PSO engine 'fast' = batched exact Woodbury + cache + pruning, complex128, identical seeds):
    tab    PSO on the tabulated model on the original grid (band <= FMAX_OPT)
    macro  PSO on the macromodel on the dense grid
Every final placement is re-scored with the EXACT original model on the original grid (exact_peak), and with the macromodel on the
dense grid (macro_dense_peak) and on the original grid (macro_grid_peak) -> peak between grid points, success_exact, fit fidelity.
Success in the tables is the PSO's own (model) success; success_exact is the honest one. Targets are given in EXP4_TARGETS.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.io as sio
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync

EXPERIMENT = "rational_macromodel"
BASE = "/DATA/Aurindum/GPUSwarm/Data/"
Y4 = os.environ.get("Y4_PATH", BASE + "VDDQ_DDR3/y_sp_ddr21.mat")
D4 = os.environ.get("D4_PATH", BASE + "Decaps/combined_decaps.mat")
F4 = os.environ.get("F4_PATH", BASE + "Decaps/sp_freq.mat")
FMAX_OPT = float(os.environ.get("FMAX_OPT", "300e6"))
EXP4_TARGETS = [float(t) for t in os.environ.get("EXP4_TARGETS", "0.5,0.4,0.35").split(",")]
RF = int(os.environ.get("RF", "2" if S["QUICK"] else "4"))                     # dense-grid refinement factor
VF_ORDERS = [int(x) for x in os.environ.get("VF_ORDERS", "24" if S["QUICK"] else "24,40,60,80").split(",")]   # numbers of poles
VF_ITERS = int(os.environ.get("VF_ITERS", "6" if S["QUICK"] else "12"))
VF_TOL = float(os.environ.get("VF_TOL", "2e-3"))                             # max |dZ| / max|Z_k| over all elements and frequencies
FMIN_FIT = float(os.environ.get("FMIN_FIT", "0"))


# ------------------------------------------------------------------------------------------------------ vector fitting
def vf_basis(s, a):
    d1 = 1.0 / (s[:, None] - a[None, :])
    d2 = 1.0 / (s[:, None] - np.conj(a)[None, :])
    return np.concatenate([d1 + d2, 1j * (d1 - d2)], axis=1)               # (n,2P): [c' (P) | c'' (P)]


def vf_design(s, a):
    return np.concatenate([vf_basis(s, a), np.ones((len(s), 1)), s[:, None]], axis=1)    # + d + s*e


def _stack(M):
    return np.concatenate([M.real, M.imag], axis=0)


def vector_fit(freq, F, P, n_iter, log=print):
    """F (n,K) complex responses on freq (Hz). Returns dict with normalised poles `a` (Im>0 members of conjugate pairs), real coefficient
    matrix `coef` (2P+2,K) (+scale), normalisation wn. Poles/s are in units of wn = 2*pi*fmax."""
    wn = 2 * np.pi * freq.max()
    s = 1j * 2 * np.pi * freq / wn
    n, K = F.shape
    sc = 1.0 / np.sqrt((np.abs(F) ** 2).mean(0))
    Fn = F * sc
    beta = np.linspace(max(s.imag.min(), 1e-3), s.imag.max(), P)
    a = -beta / 100.0 + 1j * beta
    unstable_hist = []
    for it in range(n_iter):
        A0 = vf_design(s, a)
        Phi = A0[:, :2 * P]
        R22s, rhs = [], []
        A0r = _stack(A0)
        for k in range(K):
            Ak = np.concatenate([A0r, _stack(-Fn[:, k:k + 1] * Phi)], axis=1)
            bk = np.concatenate([Fn[:, k].real, Fn[:, k].imag])
            Q, R = np.linalg.qr(Ak)
            n1 = 2 * P + 2
            R22s.append(R[n1:, n1:])
            rhs.append(Q[:, n1:].T @ bk)
        ct = np.linalg.lstsq(np.concatenate(R22s, axis=0), np.concatenate(rhs), rcond=None)[0]       # [c' | c'']
        Am = np.zeros((2 * P, 2 * P))
        bv = np.zeros(2 * P)
        cv = np.zeros(2 * P)
        for i in range(P):
            Am[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[a[i].real, a[i].imag], [-a[i].imag, a[i].real]]
            bv[2 * i] = 2.0
            cv[2 * i], cv[2 * i + 1] = ct[i], ct[P + i]
        ev = np.linalg.eigvals(Am - np.outer(bv, cv))
        up = ev[ev.imag > 1e-12 * np.abs(ev).max()]
        re = np.sort(ev[np.abs(ev.imag) <= 1e-12 * np.abs(ev).max()].real)
        extra = [complex(0.5 * (re[i] + re[i + 1]), 1e-3 * abs(re[i])) for i in range(0, len(re) - 1, 2)]
        new = np.concatenate([up, np.array(extra, dtype=complex)])
        if len(new) != P:                       # degenerate relocation: keep the old poles for the missing ones
            new = np.concatenate([new, a])[:P]
        unstable_hist.append(int((new.real > 0).sum()))
        a = np.where(new.real > 0, -new.real, new.real) + 1j * new.imag
    A0 = vf_design(s, a)
    coef = np.linalg.lstsq(_stack(A0), np.concatenate([Fn.real, Fn.imag], axis=0), rcond=None)[0]
    return {"a": a, "coef": coef, "sc": sc, "wn": wn, "P": P, "unstable_flips": unstable_hist}


def vf_eval(model, freq):
    s = 1j * 2 * np.pi * freq / model["wn"]
    return (vf_design(s, model["a"]) @ model["coef"]) / model["sc"]


# ------------------------------------------------------------------------------------------------------ capacitor R-L-C
def fit_rlc(d, omega):
    Z = 1.0 / d
    w = 1.0 / np.abs(Z) ** 2
    R = (w * Z.real).sum(1) / w.sum(1)
    a, b = omega[None, :], -1.0 / omega[None, :]
    saa, sab, sbb = (w * a * a).sum(1), (w * a * b).sum(1), (w * b * b).sum(1)
    sya, syb = (w * Z.imag * a).sum(1), (w * Z.imag * b).sum(1)
    det = saa * sbb - sab ** 2
    L = (sya * sbb - syb * sab) / det
    Ci = (saa * syb - sab * sya) / det
    return np.clip(R, 1e-4, None), np.clip(L, 1e-12, None), np.clip(Ci, 1e-3, None)


def rlc_admittance(R, L, Ci, freq):
    om = 2 * np.pi * freq
    return 1.0 / (R[:, None] + 1j * om[None, :] * L[:, None] + Ci[:, None] / (1j * om[None, :]))


# ------------------------------------------------------------------------------------------------------ build models
def build(run):
    t0 = time.perf_counter()
    y = np.transpose(sio.loadmat(Y4)["y"], (2, 0, 1))
    dall = sio.loadmat(D4)["decaps"]
    f = sio.loadmat(F4)["freq"].ravel()
    keep = (f <= FMAX_OPT) & (f >= FMIN_FIT)
    f, y, d = f[keep], y[keep], dall[:, keep]
    measured_end = 99.8e6
    n = y.shape[1]
    z0 = np.linalg.inv(y)
    run.lg.info("STAGE | data | n_ports=%d | grid_pts=%d (<= %.0f MHz) | models=%d | load+inv %.1fs", n, len(f), FMAX_OPT / 1e6, d.shape[0], time.perf_counter() - t0)
    iu = np.triu_indices(n)
    Fresp = z0[:, iu[0], iu[1]]                                              # (F,231)
    best, rows = None, []
    for P in VF_ORDERS:
        t1 = time.perf_counter()
        m = vector_fit(f, Fresp, P, VF_ITERS)
        fit = vf_eval(m, f)
        err = np.abs(fit - Fresp) / np.abs(Fresp).max(0)                     # per element, relative to its own peak
        z11e = np.abs(fit[:, 0] - Fresp[:, 0]) / np.abs(Fresp[:, 0])
        rows.append({"poles": 2 * P, "max_err": float(err.max()), "rms_err": float(np.sqrt((err ** 2).mean())),
                     "z11_max_rel_err": float(z11e.max()), "unstable_flips_last": m["unstable_flips"][-1],
                     "unstable_flips_total": int(sum(m["unstable_flips"])), "min_real_pole_ratio": float((-m["a"].real / np.abs(m["a"])).min()),
                     "fit_time_s": time.perf_counter() - t1})
        run.lg.info("STAGE | vf | %s", rows[-1])
        best = m
        if rows[-1]["max_err"] <= VF_TOL:
            break
    # capacitors: pure-RLC reference fit (reported only) and the hybrid tail model
    fm = f <= measured_end
    jf = int(np.nonzero(fm)[0].max())
    om = 2 * np.pi * f[fm]
    R0_, L0_, C0_ = fit_rlc(d[:, fm], om)
    rel0 = np.abs(np.abs(1.0 / rlc_admittance(R0_, L0_, C0_, f[fm])) - np.abs(1.0 / d[:, fm])) / np.abs(1.0 / d[:, fm])
    top = (f >= 0.7 * f[jf]) & fm
    R, L, Ci = fit_rlc(d[:, top], 2 * np.pi * f[top])
    zj = (1.0 / d[:, jf]) / (1.0 / rlc_admittance(R, L, Ci, f[jf:jf + 1])[:, 0])        # junction continuity factor (complex)
    reltop = np.abs(np.abs(1.0 / rlc_admittance(R, L, Ci, f[top])) - np.abs(1.0 / d[:, top])) / np.abs(1.0 / d[:, top])
    cap_fit = {"pure_rlc_whole_band_median_of_median_rel_err": float(np.median(np.median(rel0, 1))),
               "pure_rlc_whole_band_p90_of_median_rel_err": float(np.percentile(np.median(rel0, 1), 90)),
               "tail_rlc_top30pct_median_of_median_rel_err": float(np.median(np.median(reltop, 1))),
               "junction_scale_abs_median": float(np.median(np.abs(zj))), "junction_index": jf, "junction_MHz": float(f[jf] / 1e6)}
    if jf + 1 < len(f):
        Yt = 1.0 / (zj[:, None] / rlc_admittance(R, L, Ci, f[jf + 1:]))
        cap_fit["median_tab_vs_macro_|Y|_ratio_above_99.8MHz"] = float(np.median(np.abs(Yt) / np.abs(d[:, jf + 1:])))
    cap = {"R": R, "L": L, "Ci": Ci, "zj": zj, "jf": jf}
    # dense grid: RF-times refinement of the original grid (contains the original points)
    fd = np.concatenate([np.linspace(f[i], f[i + 1], RF, endpoint=False) for i in range(len(f) - 1)] + [f[-1:]])
    zd = np.empty((len(fd), n, n), dtype=np.complex128)
    fitd = vf_eval(best, fd)
    zd[:, iu[0], iu[1]] = fitd
    zd[:, iu[1], iu[0]] = fitd
    zg = np.empty((len(f), n, n), dtype=np.complex128)
    fitg = vf_eval(best, f)
    zg[:, iu[0], iu[1]] = fitg
    zg[:, iu[1], iu[0]] = fitg
    info = {"vf_rows": rows, "vf_poles": 2 * best["P"], "cap_fit": cap_fit, "n_grid": len(f), "n_dense": len(fd),
            "build_time_s": time.perf_counter() - t0, "dense_factor": RF}
    return f, fd, y, d, z0, zg, zd, cap, best, info


def main():
    run = C.Run(EXPERIMENT, __file__, {"Y4_PATH": Y4, "D4_PATH": D4, "F4_PATH": F4, "FMAX_OPT": FMAX_OPT, "EXP4_TARGETS": EXP4_TARGETS,
                                       "RF": RF, "VF_ORDERS": VF_ORDERS, "VF_ITERS": VF_ITERS, "VF_TOL": VF_TOL})
    C.init_device_log(run)
    f, fd, y, d, z0, zg, zd, cap, vfm, info = build(run)
    prob_tab = C.Problem(y=y, d=d, freq=f, label="tab")
    # macro capacitor admittances on the dense grid and on the original grid (hybrid model)
    jf, nC = cap["jf"], d.shape[0]
    idx = np.minimum(np.arange(len(fd)) // RF, len(f) - 1)
    wgt = (np.arange(len(fd)) % RF) / RF
    lo = fd <= f[jf]
    Dd = np.empty((nC, len(fd)), dtype=np.complex128)
    Dg = d.copy()
    for s0 in range(0, nC, 512):
        sl = slice(s0, s0 + 512)
        dd = (1 - wgt) * d[sl][:, idx] + wgt * d[sl][:, np.minimum(idx + 1, len(f) - 1)]
        dd[:, ~lo] = 1.0 / (cap["zj"][sl, None] / rlc_admittance(cap["R"][sl], cap["L"][sl], cap["Ci"][sl], fd[~lo]))
        Dd[sl] = dd
        if jf + 1 < len(f):
            Dg[sl, jf + 1:] = 1.0 / (cap["zj"][sl, None] / rlc_admittance(cap["R"][sl], cap["L"][sl], cap["Ci"][sl], f[jf + 1:]))
    prob_dense = C.Problem(z0=zd, d=Dd, freq=fd, label="macro_dense")
    del Dd
    prob_mgrid = C.Problem(z0=zg, d=Dg, freq=f, label="macro_grid")
    del Dg
    # fidelity: macro vs exact on the original grid for random configurations (cost level)
    rng = np.random.default_rng(11)
    devs, ratio = [], []
    for kk in (3, 8, 15):
        Ms, Ps = [], []
        for _ in range(30):
            m_, p_ = prob_tab.decode_particle(rng.random(2 * kk), kk)
            m_, p_ = prob_tab.resolve_adjacent_ports(m_, p_)
            Ms.append(m_)
            Ps.append(p_)
        mt, pt = torch.tensor(np.array(Ms), device=DEVICE), torch.tensor(np.array(Ps), device=DEVICE)
        ex = prob_tab.wb_peak(mt, pt)[0]
        mg = prob_mgrid.wb_peak(mt, pt)[0]
        md = prob_dense.wb_peak(mt, pt)[0]
        devs.append(((mg - ex).abs() / ex).cpu().numpy())
        ratio.append((md / mg).cpu().numpy())
    devs, ratio = np.concatenate(devs), np.concatenate(ratio)
    info["random_cfg_macro_vs_exact_grid_rel_dev"] = {"median": float(np.median(devs)), "max": float(devs.max())}
    info["random_cfg_dense_over_grid_peak_ratio"] = {"median": float(np.median(ratio)), "max": float(ratio.max()), "min": float(ratio.min())}
    bare_t = prob_tab.Z0[:, 0, 0].abs().cpu().numpy()
    bare_d = prob_dense.Z0[:, 0, 0].abs().cpu().numpy()
    info["bare_z11_peak"] = {"tabulated_grid": float(bare_t.max()), "macro_dense": float(bare_d.max()),
                             "macro_dense_f_MHz": float(fd[bare_d.argmax()] / 1e6)}
    json.dump(info, open(os.path.join(run.run_dir, "summary", "macromodel.json"), "w"), indent=2)
    with open(os.path.join(run.run_dir, "summary", "macromodel.txt"), "w") as fh:
        fh.write("PDN vector fit (common poles, 231 unique Z0 elements); error = |fit-data| / max|Z_k| per element\n")
        for r in info["vf_rows"]:
            fh.write(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()) + "\n")
        fh.write(f"chosen poles = {info['vf_poles']}\n")
        fh.write(f"capacitor model: {info['cap_fit']}\n")
        fh.write(f"random configurations: macro vs exact on the original grid, relative cost deviation: {info['random_cfg_macro_vs_exact_grid_rel_dev']}\n")
        fh.write(f"random configurations: dense-grid peak / original-grid peak (macro): {info['random_cfg_dense_over_grid_peak_ratio']}\n")
        fh.write(f"bare |Z11| peak: {info['bare_z11_peak']}\n")
        fh.write(f"grid points: original {info['n_grid']}, dense {info['n_dense']} (x{RF}); build time {info['build_time_s']:.1f}s\n")
    # ---- PSO arms
    arms = {"tab": (prob_tab, C.SwarmEvaluator(prob_tab, True, True, name="tab")),
            "macro": (prob_dense, C.SwarmEvaluator(prob_dense, True, True, name="macro"))}
    for pr, ev in arms.values():
        C.warmup(pr, [ev])
    for thr in EXP4_TARGETS:
        for name, (pr, ev) in arms.items():
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", name, thr, r)
                try:
                    rec = C.run_pso(pr, ev, [thr], r, label=name)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(name, thr, r, e)
                    continue
                if rec["best_models"] is not None:
                    mt = torch.tensor([rec["best_models"]], device=DEVICE)
                    pt = torch.tensor([rec["best_ports"]], device=DEVICE)
                    rec["exact_peak"] = float(prob_tab.wb_peak(mt, pt)[0][0])                 # original model, original grid
                    rec["macro_grid_peak"] = float(prob_mgrid.wb_peak(mt, pt)[0][0])
                    pk, am = prob_dense.wb_peak(mt, pt)
                    rec["macro_dense_peak"] = float(pk[0])
                    rec["macro_dense_peak_f_MHz"] = float(fd[int(am[0])] / 1e6)
                    rec["dense_over_grid"] = rec["macro_dense_peak"] / rec["macro_grid_peak"]
                    rec["success_exact"] = bool(rec["exact_peak"] <= thr)
                    rec["rescore_dev"] = abs(rec["exact_peak"] - rec["best_minz"]) / rec["best_minz"] if name == "tab" else None
                    rec["macro_vs_exact"] = (rec["macro_grid_peak"] - rec["exact_peak"]) / rec["exact_peak"]
                run.add(rec)
                run.result_line(rec, exact=f"{rec.get('exact_peak', float('nan')):.5f}", dense_over_grid=f"{rec.get('dense_over_grid', float('nan')):.3f}",
                                ok_exact=rec.get("success_exact"))
    run.summarise(extra_cols=("exact_peak", "macro_grid_peak", "macro_dense_peak", "dense_over_grid", "success_exact", "macro_vs_exact", "rescore_dev", "gpu_full_evals"))
    import pandas as pd
    df = pd.DataFrame(run.records)
    lines = []
    for m, g in df.groupby("method"):
        g = g.dropna(subset=["exact_peak"])
        if len(g):
            lines.append(f"{m}: success(own model) = {df[df.method == m].success.mean():.3f} | success(exact re-score) = {g.success_exact.mean():.3f} | "
                         f"exact_peak mean = {g.exact_peak.mean():.5f} | dense/grid peak ratio mean = {g.dense_over_grid.mean():.4f} max = {g.dense_over_grid.max():.4f}")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
    plots(run, f, fd, prob_tab, prob_dense, prob_mgrid, df)
    run.finish(notes={"vf_poles": info["vf_poles"]})


def plots(run, f, fd, prob_tab, prob_dense, prob_mgrid, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    zt = prob_tab.Z0[:, 0, 0].abs().cpu().numpy()
    zd = prob_dense.Z0[:, 0, 0].abs().cpu().numpy()
    ax[0].semilogy(f / 1e6, zt, ".", ms=2, label="data, original grid")
    ax[0].semilogy(fd / 1e6, zd, lw=.8, label=f"rational fit, dense grid (x{RF})")
    ax[0].axvline(99.8, color="gray", ls=":", label="caps extrapolated above")
    ax[0].set_xlabel("frequency (MHz)")
    ax[0].set_ylabel("bare |Z11| (ohm)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=.3)
    ok = df.dropna(subset=["exact_peak"])
    for m, g in ok.groupby("method"):
        ax[1].scatter(g.exact_peak, g.macro_dense_peak, s=12, label=m)
    lim = [ok.exact_peak.min() * 0.9, ok.exact_peak.max() * 1.1] if len(ok) else [0.1, 1]
    ax[1].plot(lim, lim, "k--", lw=.8)
    ax[1].set_xlabel("exact peak, original model on original grid (ohm)")
    ax[1].set_ylabel("macromodel peak on dense grid (ohm)")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "macromodel.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
