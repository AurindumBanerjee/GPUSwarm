# Experiment 04 -- rational macromodel fitness with grid-free peak search.
#
# Idea (GVF / TC-GVF, vector fitting): replace the tabulated data by
# rational models and evaluate the objective on the models.
#   * PDN: the bare port impedance Z0(s) = Y(s)^-1 is fitted ONCE with
#     common-pole Vector Fitting (all upper-triangle entries).
#   * Decaps: every SHUNT part is fitted as a series R-L-C,
#     Z_cap(s) = R + sL + 1/(sC).
#   * Fitness: Woodbury on the models. A coarse log grid (--ngrid points)
#     locates the candidate peaks, then each of the top --kpeaks local
#     maxima is refined by zooming (--zoom-rounds x --zoom-points) at
#     arbitrary frequencies, so peaks between samples are not missed.
#   * --fmax-ext lets the grid extend ABOVE the data band (the band-edge
#     problem: every SP dataset peaks at the last sample). This is model
#     EXTRAPOLATION and is reported as such.
#
# Exact H-infinity norms (Hamiltonian bisection) are not used: the
# state dimension (poles x ports) makes them slower than the grid+zoom
# search at our sizes. The zoomed peak is a lower bound of the true model
# peak that converges as zoom settings increase.
#
# Fitness is a different (model) objective, so trajectories differ from the
# table baseline. Every run's final best placement is therefore re-scored
# on the TABLE (in-band) and reported as best_minz_table.
#
# Methods
#   numpy   table baseline
#   fast    GPUBench2 fast (table)
#   macro   rational-model fitness (cache on)
#
# nohup python bench_macromodel.py --freq <freq.mat> > exp04.log 2>&1 &

import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import gbcore as gb  # noqa: E402
import macromodel as mm  # noqa: E402

p = gb.common_parser("Exp04 rational macromodel", "numpy,fast,macro",
                     "MinTime/Experiments/04_macromodel_rational_peak")
p.add_argument("--vf-pairs", type=int, default=30)
p.add_argument("--vf-iter", type=int, default=6)
p.add_argument("--ngrid", type=int, default=400)
p.add_argument("--kpeaks", type=int, default=3)
p.add_argument("--zoom-rounds", type=int, default=3)
p.add_argument("--zoom-points", type=int, default=9)
p.add_argument("--fmax-ext", type=float, default=None, help="extend the search band above the data (Hz)")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
if prob.freq is None:
    raise SystemExit("Exp04 needs --freq (the models are functions of frequency)")
dev, cd, T, N = prob.device, prob.cdtype, prob.TARGET_PORT, prob.N

# ------------------------------------------------------------ fitting
t0 = time.perf_counter()
Z0_np = np.linalg.inv(prob.y_np)
iu = np.triu_indices(N)
model, vf_err = mm.vector_fit(Z0_np[:, iu[0], iu[1]], prob.freq, args.vf_pairs, args.vf_iter)
Rc, Lc, Cc, rlc_err = mm.fit_rlc(prob.d_np, prob.freq)
fit_time = time.perf_counter() - t0
fit_info = {"vf_pairs": args.vf_pairs, "vf_iter": args.vf_iter, "n_poles": int(len(model.poles)),
            "vf_err_normalised": vf_err,
            "z11_fit_max_rel": float(np.max(np.abs(model.eval(1j * 2 * np.pi * prob.freq)[:, 0] - Z0_np[:, 0, 0])
                                            / np.abs(Z0_np[:, 0, 0]))),
            "rlc_err_median": float(np.median(rlc_err)), "rlc_err_p95": float(np.percentile(rlc_err, 95)),
            "rlc_err_max": float(rlc_err.max()), "fit_time_s": fit_time}
print("[fit]", json.dumps(fit_info))
with open(os.path.join(args.out, "fit_info.json"), "w") as f:
    json.dump(fit_info, f, indent=2)

# torch copies of the models
poles = torch.tensor(model.poles, dtype=cd, device=dev)
Res = torch.zeros(N, N, len(model.poles), dtype=cd, device=dev)
Dm = torch.zeros(N, N, dtype=cd, device=dev)
Em = torch.zeros(N, N, dtype=cd, device=dev)
ri, ci = torch.tensor(iu[0], device=dev), torch.tensor(iu[1], device=dev)
Res[ri, ci] = torch.tensor(model.residues, dtype=cd, device=dev)
Res[ci, ri] = torch.tensor(model.residues, dtype=cd, device=dev)
Dm[ri, ci] = torch.tensor(model.d, dtype=cd, device=dev)
Dm[ci, ri] = torch.tensor(model.d, dtype=cd, device=dev)
Em[ri, ci] = torch.tensor(model.e, dtype=cd, device=dev)
Em[ci, ri] = torch.tensor(model.e, dtype=cd, device=dev)
SCALE = model.scale
R_t = torch.tensor(Rc, dtype=torch.float64, device=dev)
L_t = torch.tensor(Lc, dtype=torch.float64, device=dev)
C_t = torch.tensor(Cc, dtype=torch.float64, device=dev)

f_lo = float(prob.freq.min())
f_hi = float(args.fmax_ext or prob.freq.max())
if args.fmax_ext:
    print(f"[band] searching up to {f_hi:.4g} Hz -- EXTRAPOLATION above {prob.freq.max():.4g} Hz")
logf_grid = np.linspace(np.log10(f_lo), np.log10(f_hi), args.ngrid)
s_grid = torch.tensor(1j * 2 * np.pi * 10 ** logf_grid, dtype=cd, device=dev)


def z0_at(sn):
    """Full Z0 model at normalised s (any shape S) -> S + (N,N)."""
    x = sn[..., None, None, None]
    return (Res / (x - poles)).sum(-1) + Dm + sn[..., None, None] * Em


def zcap(models, s):
    """Cap impedances, models (b,k), s (b,m) unnormalised -> (b,m,k)."""
    s_ = s.unsqueeze(-1)
    return R_t[models].unsqueeze(1) + s_ * L_t[models].unsqueeze(1) + 1.0 / (s_ * C_t[models].unsqueeze(1))


Zg = z0_at(s_grid / SCALE)                                   # (Ng,N,N)


def z11_at(mt, pt, s, Zfull=None):
    """|Z11| for particles (b,k) at per-particle frequencies s (b,m).
    Zfull: optional precomputed (b,m,N,N) table (coarse grid); otherwise the
    needed entries are built from the pole-residue model via einsum, so the
    cost is O(b*m*k^2*Np) and independent of the port count N."""
    b, k = pt.shape
    m = s.shape[1]
    if Zfull is not None:
        bi = torch.arange(b, device=dev)[:, None, None, None]
        mi = torch.arange(m, device=dev)[None, :, None, None]
        Zpp = Zfull[bi, mi, pt[:, None, :, None], pt[:, None, None, :]]          # (b,m,k,k)
        zc = Zfull[bi[..., 0], mi[..., 0], pt[:, None, :], T]                    # (b,m,k)
        zr = Zfull[bi[..., 0], mi[..., 0], T, pt[:, None, :]]
        ztt = Zfull[:, :, T, T]
    else:
        sn = s / SCALE
        G = 1.0 / (sn.unsqueeze(-1) - poles)                                     # (b,m,Np)
        Rpp = Res[pt[:, :, None], pt[:, None, :]]                               # (b,k,k,Np)
        Zpp = torch.einsum("bmp,bijp->bmij", G, Rpp) \
            + Dm[pt[:, :, None], pt[:, None, :]].unsqueeze(1) \
            + sn[..., None, None] * Em[pt[:, :, None], pt[:, None, :]].unsqueeze(1)
        zc = torch.einsum("bmp,bip->bmi", G, Res[pt, T]) + Dm[pt, T].unsqueeze(1) \
            + sn.unsqueeze(-1) * Em[pt, T].unsqueeze(1)
        zr = torch.einsum("bmp,bip->bmi", G, Res[T, pt]) + Dm[T, pt].unsqueeze(1) \
            + sn.unsqueeze(-1) * Em[T, pt].unsqueeze(1)
        ztt = G @ Res[T, T] + Dm[T, T] + sn * Em[T, T]
    Mx = Zpp + torch.diag_embed(zcap(mt, s))
    x, info = torch.linalg.solve_ex(Mx, zc.unsqueeze(-1))
    val = ztt - (zr.unsqueeze(-2) @ x)[..., 0, 0]
    a = val.abs()
    return torch.where(torch.isfinite(a) & (info == 0), a, torch.full_like(a, 1e200))


def macro_peak(mt, pt):
    out = []
    b_all = pt.shape[0]
    k = pt.shape[1]
    chunk = max(1, int(args.max_batch_elems // max(1, args.ngrid * k * k)))
    for s0 in range(0, b_all, chunk):
        m_, p_ = mt[s0:s0 + chunk], pt[s0:s0 + chunk]
        b = p_.shape[0]
        sg = s_grid.unsqueeze(0).expand(b, -1)
        curve = z11_at(m_, p_, sg, Zg.unsqueeze(0).expand(b, -1, -1, -1))       # (b,Ng)
        best = curve.max(dim=1).values
        # local maxima -> top K
        lm = torch.zeros_like(curve, dtype=torch.bool)
        lm[:, 1:-1] = (curve[:, 1:-1] >= curve[:, :-2]) & (curve[:, 1:-1] >= curve[:, 2:])
        lm[:, 0] = curve[:, 0] >= curve[:, 1]
        lm[:, -1] = curve[:, -1] >= curve[:, -2]
        kk = min(args.kpeaks, curve.shape[1])
        idx = torch.where(lm, curve, torch.full_like(curve, -1.0)).topk(kk, dim=1).indices     # (b,K)
        lg = torch.tensor(logf_grid, device=dev, dtype=torch.float64)
        step = float(logf_grid[1] - logf_grid[0])
        lo = lg[idx] - step
        hi = lg[idx] + step
        for _ in range(args.zoom_rounds):
            t = torch.linspace(0, 1, args.zoom_points, device=dev, dtype=torch.float64)
            lf = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * t                 # (b,K,P)
            lf = lf.clamp(np.log10(f_lo), np.log10(f_hi))
            s = (1j * 2 * np.pi * 10 ** lf).to(cd).reshape(b, -1)
            v = z11_at(m_, p_, s).reshape(b, kk, -1)
            best = torch.maximum(best, v.amax(dim=(1, 2)))
            j = v.argmax(dim=-1, keepdim=True)
            w = (hi - lo) / (args.zoom_points - 1)
            c = lf.gather(-1, j).squeeze(-1)
            lo, hi = c - w, c + w
        out.append(best)
    return torch.cat(out), None


fac = gb.standard_evaluators(prob)
fac["macro"] = lambda: gb.SwarmEvaluator(prob, True, False, full_fn=macro_peak)
methods = gb.select_methods(args, fac)

# verification: table methods strict; macro reported vs table
gb.verify(prob, {m: fac[m] for m in methods if m != "macro"}, args.out)
cfgs = gb.random_configs(prob, 10, seed=5, kmax=min(6, len(prob.VALID_PORTS)))
cmp = []
for m, pp in cfgs:
    mt = torch.tensor(m[None], device=dev)
    pt = torch.tensor(pp[None], device=dev)
    tab = prob.wb_peak(mt, pt)[0].item()
    mac = macro_peak(mt, pt)[0].item()
    cmp.append((mac - tab) / tab)
print(f"[verify] macro vs table peak rel diff: min {min(cmp):+.3e} max {max(cmp):+.3e} "
      f"(positive = macro found a higher in-between/extended peak or fit error)")
with open(os.path.join(args.out, "macro_vs_table.json"), "w") as f:
    json.dump({"rel_diff": cmp}, f, indent=2)

gb.warmup(prob, {m: fac[m] for m in methods})
recs = gb.run_benchmark(prob, fac, methods, args.out)
for r in recs:                                     # re-score final placements on the table
    if r.get("best_config"):
        m, pp = r["best_config"]
        mt = torch.tensor([m], device=dev)
        pt = torch.tensor([pp], device=dev)
        r["best_minz_table"] = float(prob.wb_peak(mt, pt)[0].item())
df, s = gb.analyse(recs, args.out, args.baseline)
tab = df.groupby(["threshold", "method"]).apply(
    lambda g: g["best_minz_table"].mean() if "best_minz_table" in g else np.nan)
tab.to_csv(os.path.join(args.out, "best_minz_table_by_method.csv"))
print("\nmean best |Z11| re-scored on the table:\n" + tab.to_string())
print("Done.")
