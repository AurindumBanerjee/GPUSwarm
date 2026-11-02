#!/usr/bin/env python3
"""
Predictive cost model -- how long does ONE fitness evaluation take, as a function of the PDN size N, the number of capacitors k,
the number of frequencies F, the observation ports n_obs and the swarm size B, for every method on CPU and GPU -- and which method
should be used for a given PDN?

Methods (all exact, all return |Z_oo| at every frequency):
    numpy     assemble Y, full N x N inverse per frequency                  ~ F * N^3            (one placement per call)
    solve     assemble Y, LU + n_obs right-hand sides per frequency         ~ F * (N^3/3 + N^2 n_obs)
    woodbury  gather the k x k block of Z0, LU + n_obs right-hand sides     ~ B * F * (k^3/3 + k^2 n_obs + k n_obs)   (B placements per call)
on device cpu (numpy, BLAS threads as configured) and gpu (torch, complex128).

1. MEASURE a grid of (N, k, F, B) on random reciprocal admittance matrices (timing does not depend on the data).
2. FIT per (method, device):   t_eval = c0 / B_call + c1 * work + c2 * bytes        (non-negative least squares, relative weighting)
       work  = the flop count above per placement,  bytes = data touched per placement,  B_call = placements per call (overhead share)
3. VALIDATE by leave-one-N-out cross-validation: fit without one N, predict it; report the median / 90th percentile relative error.
4. PREDICT the four AllCPU PDNs (their real N, F, n_obs) for k = 1..40 and swarm sizes 50 / 200, and write the DECISION RULE: the
   fastest method per (PDN, k, device), plus the break-even k where woodbury stops beating solve.

   python cost_model.py [--quick] [--devices cpu,gpu] [--reps 3]
Output: cost_model/out/{measurements.csv, fit.json, cv.csv, predictions.csv, decision.md, plots/*.png}
"""
import os
import sys
import json
import time
import argparse
import itertools

import numpy as np
try:
    import tabulate  # noqa: F401  -- DataFrame.to_markdown needs it; the server has no internet to install it, so fall back to a plain pipe-table writer
except ImportError:
    import pandas as _pd

    def _to_markdown(self, buf=None, mode="wt", index=True, **kwargs):
        df = self.reset_index() if index else self
        cols = [" | ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
        fmt = lambda v: format(v, "g") if isinstance(v, float) else str(v)
        rows = [[fmt(v) for v in r] for r in df.itertuples(index=False, name=None)]
        w = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
        line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
        out = "\n".join([line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])
        if buf is None:
            return out
        buf.write(out)
    _pd.DataFrame.to_markdown = _to_markdown

H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--devices", default="cpu,gpu")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default=os.path.join(H, "out"))
ap.add_argument("--pdn-dims", default="pdn1:21:1:1391,pdn2:28:1:2828,pdn3:73:5:3000,pdn4:79:1:3000",
                help="name:N:n_obs:F of the PDNs to predict (F of pdn3/4 depends on the built 5 GHz grid; edit if needed)")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)

import torch
HAS_GPU = torch.cuda.is_available()
devices = [d for d in a.devices.split(",") if d == "cpu" or (d == "gpu" and HAS_GPU)]
if "gpu" in a.devices and not HAS_GPU:
    print("[note] no CUDA device: GPU measurements skipped")

if a.quick:
    GRID_N, GRID_K, GRID_F, GRID_B = [11, 21, 41], [1, 5, 10], [200, 800], [10, 50]
else:
    GRID_N, GRID_K, GRID_F, GRID_B = [11, 21, 41, 79, 121], [1, 2, 5, 10, 20, 40], [500, 1500, 3000], [50, 200]
NOBS = [1, 5]
SKIPPED = []                                        # measurement cells that do not fit in GPU memory (listed in decision.md)


def rand_problem(N, F, C=256, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(F, N, N)) + 1j * rng.normal(size=(F, N, N))
    Y = (A + np.transpose(A, (0, 2, 1))) / 2 + N * np.eye(N)[None] * (1 + 1j)          # reciprocal, well conditioned
    D = (rng.normal(size=(C, F)) + 1j * rng.normal(size=(C, F))) * 0.1
    return Y, D


def work(method, N, k, F, no):
    if method == "numpy":
        return F * (8.0 / 3.0) * N ** 3 * 4                         # complex inverse ~ 8/3 N^3 complex flops ~ x4 real
    if method == "solve":
        return F * ((8.0 / 3.0) * N ** 3 / 2 * 4 / 2 + 8 * N ** 2 * no)
    return F * ((8.0 / 3.0) * k ** 3 + 8 * k ** 2 * no + 8 * k * no)


def nbytes(method, N, k, F, no):
    return 16 * F * (N * N if method in ("numpy", "solve") else (k * k + 2 * k * no + k))


def time_it(fn, dev, reps):
    fn()
    if dev == "gpu":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if dev == "gpu":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure():
    rows = []
    for N, F in itertools.product(GRID_N, GRID_F):
        Y, D = rand_problem(N, F)
        Z0 = np.linalg.inv(Y)
        for dev in devices:
            if dev == "gpu":
                Yd = torch.from_numpy(Y).cuda()
                Dd = torch.from_numpy(D).cuda()
                Z0d = torch.linalg.inv(Yd)
                DId = 1.0 / Dd
            for no in NOBS:
                if no >= N:
                    continue
                pads = np.arange(no, N)
                for k in [k for k in GRID_K if k <= len(pads)]:
                    rng = np.random.default_rng(k)
                    m = rng.integers(0, D.shape[0], size=k)
                    p = rng.choice(pads, size=k, replace=False)
                    # ---- numpy / solve: one placement per call (they are not batched over particles)
                    for meth in ("numpy", "solve"):
                        if dev == "cpu":
                            def fn(meth=meth):
                                A = Y.copy()
                                A[:, p, p] += D[m].T
                                if meth == "numpy":
                                    np.linalg.inv(A)
                                else:
                                    E = np.zeros((F, N, no), complex)
                                    E[:, np.arange(no), np.arange(no)] = 1
                                    np.linalg.solve(A, E)
                        else:
                            def fn(meth=meth):
                                A = Yd.clone()
                                A[:, p, p] += Dd[m].T
                                if meth == "numpy":
                                    torch.linalg.inv(A)
                                else:
                                    E = torch.zeros((F, N, no), dtype=A.dtype, device="cuda")
                                    E[:, torch.arange(no), torch.arange(no)] = 1
                                    torch.linalg.solve(A, E)
                        t = time_it(fn, dev, a.reps)
                        rows.append(dict(device=dev, method=meth, N=N, k=k, F=F, n_obs=no, B=1, t_call=t, t_eval=t))
                    # ---- woodbury: B placements per call
                    for B in GRID_B:
                        mm = rng.integers(0, D.shape[0], size=(B, k))
                        pp = np.stack([rng.choice(pads, size=k, replace=False) for _ in range(B)])
                        if dev == "cpu":
                            DI = 1.0 / D

                            def fn():
                                M = Z0[:, pp[:, :, None], pp[:, None, :]].transpose(1, 0, 2, 3).copy()
                                M[..., np.arange(k), np.arange(k)] += DI[mm].transpose(0, 2, 1)
                                Rr = Z0[:, pp, :no].transpose(1, 0, 2, 3)
                                X = np.linalg.solve(M, Rr)
                                Lr = Z0[:, :no, pp].transpose(2, 0, 1, 3)
                                (Lr * np.swapaxes(X, -1, -2)).sum(-1)
                        else:
                            est = 5 * 16 * F * B * k * k                  # gathered M + permuted copy + diag add + LU copy + result
                            free = torch.cuda.mem_get_info()[0]
                            if est > 0.8 * free:
                                SKIPPED.append(f"gpu woodbury N={N} k={k} F={F} B={B} n_obs={no}: needs ~{est / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free")
                                print("  [skip]", SKIPPED[-1], flush=True)
                                continue
                            mt = torch.from_numpy(mm).cuda()
                            pt = torch.from_numpy(pp).cuda()

                            def fn():
                                M = Z0d[:, pt.unsqueeze(2), pt.unsqueeze(1)].permute(1, 0, 2, 3) + torch.diag_embed(DId[mt].transpose(1, 2))
                                Rr = Z0d[:, pt, :no].permute(1, 0, 2, 3)
                                X = torch.linalg.solve(M, Rr)
                                Lr = Z0d[:, :no, pt].permute(2, 0, 1, 3)
                                (Lr * X.transpose(-1, -2)).sum(-1)
                        try:
                            t = time_it(fn, dev, a.reps)
                        except torch.OutOfMemoryError:
                            SKIPPED.append(f"gpu woodbury N={N} k={k} F={F} B={B} n_obs={no}: out of memory")
                            print("  [skip]", SKIPPED[-1], flush=True)
                            torch.cuda.empty_cache()
                            continue
                        if dev == "gpu":
                            torch.cuda.empty_cache()
                        rows.append(dict(device=dev, method="woodbury", N=N, k=k, F=F, n_obs=no, B=B, t_call=t, t_eval=t / B))
                print(f"  measured N={N} F={F} dev={dev}", flush=True)
    return rows


def design(df):
    w = np.array([work(r.method, r.N, r.k, r.F, r.n_obs) for r in df.itertuples()])
    b = np.array([nbytes(r.method, r.N, r.k, r.F, r.n_obs) for r in df.itertuples()])
    return np.stack([1.0 / df.B.values, w, b], 1)


def fit(df):
    from scipy.optimize import nnls
    X = design(df)
    y = df.t_eval.values
    wt = 1.0 / y                                                     # relative error weighting
    c, _ = nnls(X * wt[:, None], y * wt)
    return c


def predict(c, rows):
    import pandas as pd
    return design(pd.DataFrame(rows)) @ c


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(measure())
    df.to_csv(os.path.join(a.out, "measurements.csv"), index=False)
    coefs, cv = {}, []
    for (dev, meth), g in df.groupby(["device", "method"]):
        coefs[f"{dev}/{meth}"] = fit(g).tolist()
        for n_out in sorted(g.N.unique()):                           # leave-one-N-out cross-validation
            tr, te = g[g.N != n_out], g[g.N == n_out]
            if len(tr) < 3 or not len(te):
                continue
            c = fit(tr)
            pr = predict(c, te.to_dict("records"))
            rel = np.abs(pr - te.t_eval.values) / te.t_eval.values
            cv += [dict(device=dev, method=meth, N_held_out=n_out, rel_err=float(x)) for x in rel]
    json.dump({"columns": ["c0 (s per call, shared over B)", "c1 (s per flop)", "c2 (s per byte)"], "coefficients": coefs},
              open(os.path.join(a.out, "fit.json"), "w"), indent=1)
    CV = pd.DataFrame(cv)
    CV.to_csv(os.path.join(a.out, "cv.csv"), index=False)
    cvs = CV.groupby(["device", "method"]).rel_err.describe(percentiles=[.5, .9])[["count", "50%", "90%", "max"]] if len(CV) else None
    # ---- predictions for the PDNs + decision rule
    pdns = [dict(zip(["name", "N", "n_obs", "F"], [x.split(":")[0]] + [int(v) for v in x.split(":")[1:]])) for x in a.pdn_dims.split(",")]
    pr = []
    for p in pdns:
        for dev in devices:
            for k in range(1, min(40, p["N"] - p["n_obs"]) + 1):
                for B in (50, 200):
                    for meth in ("numpy", "solve", "woodbury"):
                        key = f"{dev}/{meth}"
                        if key not in coefs:
                            continue
                        r = dict(device=dev, method=meth, N=p["N"], k=k, F=p["F"], n_obs=p["n_obs"], B=(B if meth == "woodbury" else 1))
                        pr.append(dict(pdn=p["name"], swarm=B, **r, t_pred=float(predict(np.array(coefs[key]), [r])[0])))
    P = pd.DataFrame(pr)
    P.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    lines = [f"# Predictive cost model\n\nmeasured on {', '.join(devices)}; grid N={GRID_N} k={GRID_K} F={GRID_F} B={GRID_B} n_obs={NOBS}\n\n",
             "Model: t_eval = c0/B + c1*flops + c2*bytes per (device, method), NNLS with relative weighting.\n\n"]
    if SKIPPED:
        lines.append("Measurement cells skipped (too large for the GPU's free memory; the model extrapolates there):\n\n" + "\n".join(f"- {s}" for s in SKIPPED) + "\n\n")
    if cvs is not None:
        lines.append("## Leave-one-N-out validation (relative error of the predicted time per evaluation)\n\n" + cvs.round(3).to_markdown() + "\n\n")
    lines.append("## Decision rule: fastest method per PDN, device and k (swarm 50; numpy/solve are per placement)\n\n")
    for (pdn, dev), g in P[P.swarm == 50].groupby(["pdn", "device"]):
        best = g.loc[g.groupby("k").t_pred.idxmin()][["k", "method", "t_pred"]]
        seg, prev = [], None
        for r in best.itertuples():
            if r.method != prev:
                seg.append([r.k, r.k, r.method])
                prev = r.method
            else:
                seg[-1][1] = r.k
        ws = g[g.method == "woodbury"].set_index("k").t_pred
        so = g[g.method == "solve"].set_index("k").t_pred
        be = [k for k in ws.index if k in so.index and ws[k] > so[k]]
        lines.append(f"- **{pdn} / {dev}**: " + ", ".join(f"k={s}-{e}: {m}" for s, e, m in seg) +
                     (f"; woodbury loses to solve from k={be[0]}" if be else "; woodbury fastest for every k <= 40") +
                     f"; predicted speed-up woodbury vs numpy at k=10: {g[(g.method == 'numpy') & (g.k == 10)].t_pred.mean() / max(1e-30, g[(g.method == 'woodbury') & (g.k == 10)].t_pred.mean()):.1f}x\n")
    open(os.path.join(a.out, "decision.md"), "w").write("".join(lines))
    # ---- plots
    for dev, g in df.groupby("device"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for meth, x in g.groupby("method"):
            pv = predict(np.array(coefs[f"{dev}/{meth}"]), x.to_dict("records"))
            ax.loglog(x.t_eval, pv, "o", ms=3, label=meth)
        lim = [df.t_eval.min() * .5, df.t_eval.max() * 2]
        ax.plot(lim, lim, "k--", lw=.8)
        ax.set_xlabel("measured s / evaluation")
        ax.set_ylabel("predicted s / evaluation")
        ax.set_title(f"cost model fit ({dev})")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "plots", f"fit_{dev}.png"), dpi=150)
        plt.close(fig)
    for (pdn, dev), g in P[P.swarm == 50].groupby(["pdn", "device"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for meth, x in g.groupby("method"):
            ax.semilogy(x.k, x.t_pred, label=meth)
        ax.set_xlabel("capacitors k")
        ax.set_ylabel("predicted s / evaluation")
        ax.set_title(f"{pdn} on {dev}")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "plots", f"predict_{pdn}_{dev}.png"), dpi=150)
        plt.close(fig)
    print(open(os.path.join(a.out, "decision.md")).read())


if __name__ == "__main__":
    main()
