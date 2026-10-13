"""
r2_common.py -- helpers shared by the round-2 GPU experiments (speed / search_quality / validation / plots).

Builds on GPUExperiments/gpuexp_common.py (Problem, evaluators, run_pso, Run bookkeeping); nothing there is changed.

    DATASETS / load_problem   named PDNs (reference y2, DDR3 21/79 ports, MPHY rails) with the matching grid and SHUNT library
    targets_for               absolute targets, or targets relative to the bare peak (TARGET_MODE=relative)
    impedance_before_after    |Z11| without decaps and with the placed decaps on the same axes (+ target line)
    plot_best_runs            one before/after plot per (method, threshold) from the run records
    loop_methods              thresholds x methods x runs with paired seeds, trajectory match vs a reference arm, failures logged
    fisher_success            Fisher exact test on success counts
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.io as sio
import torch

import gpuexp_common as C
from gpuexp_common import S, DEVICE

ROOT = os.environ.get("GPUSWARM_ROOT", "/DATA/Aurindum/GPUSwarm")
_D = lambda *p: os.path.join(ROOT, "Data", *p)                       # noqa: E731
SP_LIB, SP_FREQ = _D("Decaps", "combined_decaps.mat"), _D("Decaps", "sp_freq.mat")
DATASETS = {
    "ref21":    (_D("Reference", "y2.mat"), _D("Reference", "decaps.mat"), _D("Reference", "freq2.mat")),
    "ddr21":    (_D("VDDQ_DDR3", "y_sp_ddr21.mat"), SP_LIB, SP_FREQ),
    "ddr79":    (_D("VDDQ_DDR3", "y_sp_ddr_full.mat"), SP_LIB, SP_FREQ),
    "mer1":     (_D("MPHY", "y_sp_mer1.mat"), SP_LIB, SP_FREQ),
    "mer2":     (_D("MPHY", "y_sp_mer2.mat"), SP_LIB, SP_FREQ),
    "pll1v0":   (_D("MPHY", "y_sp_pll1v0.mat"), SP_LIB, SP_FREQ),
    "pll1v8":   (_D("MPHY", "y_sp_pll1v8.mat"), SP_LIB, SP_FREQ),
    "mphyvdd":  (_D("MPHY", "y_sp_mphyvdd.mat"), SP_LIB, SP_FREQ),
}
# SP datasets: optimise below FMAX (band-edge problem, Data/Decaps/README_library.md). The reference set uses its full band.
SP_FMAX = float(os.environ.get("SP_FMAX", "50e6"))
# combined_decaps.mat holds 'avx_shunt' (3347) + 'legacy' (3348) models; LIB_SOURCE filters it (all | avx_shunt | legacy)
LIB_SOURCE = os.environ.get("LIB_SOURCE", "all")


def env_list(name, default, cast=str):
    v = os.environ.get(name, "")
    return [cast(x) for x in (v.split(",") if v else default.split(",")) if str(x).strip() != ""]


def load_problem(name_or_paths, fmax=None, fmin=None, lib_source=None, label=None):
    """Problem for a named dataset (or a (y, d, freq) tuple). Returns (prob, info dict)."""
    if isinstance(name_or_paths, str):
        name = name_or_paths
        y, d, f = DATASETS[name]
        if fmax is None and name != "ref21":
            fmax = SP_FMAX
    else:
        name = label or "custom"
        y, d, f = name_or_paths
    lib_source = lib_source or LIB_SOURCE
    yy = np.transpose(sio.loadmat(y)["y"], (2, 0, 1))
    dm = sio.loadmat(d)
    dd = dm["decaps"]
    keep_models = np.arange(dd.shape[0])
    if "source" in dm and lib_source != "all":
        src = np.array([str(np.asarray(x).ravel()[0]) for x in dm["source"].ravel()])
        keep_models = np.nonzero(src == lib_source)[0]
        dd = dd[keep_models]
    freq = sio.loadmat(f)["freq"].ravel()
    assert yy.shape[0] == dd.shape[1] == len(freq), (yy.shape, dd.shape, len(freq))
    keep = np.ones(len(freq), bool)
    if fmax is not None:
        keep &= freq <= fmax
    if fmin is not None:
        keep &= freq >= fmin
    prob = C.Problem(y=yy[keep], d=dd[:, keep], freq=freq[keep], label=name)
    prob.model_ids = keep_models                       # library row of every model index used by the PSO
    bare = prob.Z0[:, prob.target_port, prob.target_port].abs().cpu().numpy()
    info = {"dataset": name, "y": y, "d": d, "freq": f, "ports": prob.n, "models": prob.C, "lib_source": lib_source,
            "f_lo": float(prob.freq[0]), "f_hi": float(prob.freq[-1]), "n_freq": prob.F, "fmax": fmax,
            "bare_peak": float(bare.max()), "bare_peak_f": float(prob.freq[bare.argmax()])}
    return prob, info


def targets_for(info, default_abs="0.05,0.045,0.04", default_rel="0.6,0.5,0.4"):
    """TARGET_MODE=abs -> TARGETS (ohm); relative -> TARGET_FRACS x bare peak. Reference set defaults to abs, SP sets to relative."""
    mode = os.environ.get("TARGET_MODE", "abs" if info["dataset"] == "ref21" else "relative")
    if mode == "relative":
        return [round(fr * info["bare_peak"], 6) for fr in env_list("TARGET_FRACS", default_rel, float)]
    return env_list("TARGETS", default_abs, float)


# ------------------------------------------------------------------------------------------------------ plots
def z11_curve(prob, models, ports):
    if models is None or len(models) == 0:
        return prob.Z0[:, prob.target_port, prob.target_port].abs().cpu().numpy()
    v, _ = prob.wb_values(torch.tensor([list(models)], device=DEVICE), torch.tensor([list(ports)], device=DEVICE))
    return v[0].abs().cpu().numpy()


def impedance_before_after(prob, placements, target, path, title="", labels=None):
    """|Z11| of the bare PDN and of each placement (list of (models, ports)) on one log-log plot, with the target line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = prob.freq if prob.freq is not None else np.arange(prob.F)
    bare = z11_curve(prob, None, None)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.loglog(f, bare, color="0.45", lw=1.6, label=f"bare PDN (peak {bare.max():.4g} ohm)")
    for i, (m, p) in enumerate(placements):
        z = z11_curve(prob, m, p)
        lab = (labels[i] if labels else "with decaps") + f": {len(m)} caps, peak {z.max():.4g} ohm"
        ax.loglog(f, z, lw=1.4, label=lab)
        j = int(z.argmax())
        ax.plot(f[j], z[j], "o", ms=4, color=ax.lines[-1].get_color())
    if target:
        ax.axhline(target, color="crimson", ls="--", lw=1, label=f"target {target:.4g} ohm")
    ax.set_xlabel("frequency (Hz)" if prob.freq is not None else "frequency index")
    ax.set_ylabel("|Z11| (ohm)")
    ax.set_title(title or prob.label)
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_best_runs(prob, records, out_dir, tag=""):
    """For every (method, threshold): before/after plot of the best successful run (else the best run)."""
    import pandas as pd
    df = pd.DataFrame([r for r in records if r.get("best_models")])
    if df.empty:
        return
    for (m, thr), g in df.groupby(["method", "threshold"]):
        ok = g[g.success == True]                                             # noqa: E712
        r = (ok if len(ok) else g).sort_values("best_minz").iloc[0]
        name = f"before_after_{tag}{m}_T{thr:g}.png".replace("/", "_")
        impedance_before_after(prob, [(r.best_models, r.best_ports)], thr, os.path.join(out_dir, name),
                               title=f"{prob.label}: {m}, target {thr:g} ohm, run {int(r.run)}")


# ------------------------------------------------------------------------------------------------------ loops / stats
def loop_methods(run, prob, arms, targets, runs, reference="fast", on_stage_end=None, label_prefix="", extra=None):
    """arms: {name: evaluator}. Paired seeds; trajectory match against arm `reference` (if present). Returns records."""
    recs, ref = [], {}
    for thr in targets:
        for name, ev in arms.items():
            for r in range(1, runs + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", label_prefix + name, thr, r)
                try:
                    hook = on_stage_end(name, thr) if on_stage_end else None
                    rec = C.run_pso(prob, ev, [thr], r, label=label_prefix + name, on_stage_end=hook)
                except Exception as e:                                        # noqa: BLE001
                    run.fail(label_prefix + name, thr, r, e)
                    continue
                if name == reference:
                    ref[(thr, r)] = rec
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, ref.get((thr, r)))
                if extra:
                    rec.update(extra)
                run.add(rec)
                run.result_line(rec, match=rec["traj_match"])
                recs.append(rec)
    return recs


def fisher_success(a_succ, a_n, b_succ, b_n):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a_succ, a_n - a_succ], [b_succ, b_n - b_succ]])[1])


def time_per_success(g):
    """Total wall time of all runs / number of successes (inf if none)."""
    s = int(g.success.sum())
    t = float(g.total_time_s[g.total_time_s >= 0].sum())
    return t / s if s else float("inf")


def write_lines(run, name, lines):
    with open(os.path.join(run.run_dir, "summary", name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    for l in lines:
        run.lg.info("SUMMARY | %s", l)
