# Experiment 05 -- continuous relaxation + gradient polish + snap to parts.
#
# With each SHUNT decap described by a series R-L-C (fit_rlc), a capacitor
# becomes three continuous numbers (log R, log L, log C) instead of an index
# into the library. The Woodbury fitness is differentiable in PyTorch, so at
# the end of every PSO stage that has not met the target:
#   1. ports fixed at the stage gbest; caps initialised from their parts;
#   2. Adam minimises a smooth peak (p-norm of |Z11| over the TABLE grid)
#      with respect to (log R, log L, log C) of every cap;
#   3. snap: each cap is mapped to its --snap-k nearest real parts in
#      normalised (log R, log L, log C) space; a greedy coordinate pass
#      evaluates swaps EXACTLY on the table (batched) and keeps
#      improvements;
#   4. if the snapped placement beats gbest it replaces it (and seeds the
#      next stage's warm start).
# Every candidate that counts is scored exactly on the tabulated data, so
# reported |Z11| values are honest. Polish evaluations are added to n_evals
# and polish time is inside the timed run.
#
# Trajectories differ from the baseline after the first polish (by design).
#
# Methods
#   numpy        baseline
#   fast         GPUBench2 fast
#   fast_polish  fast + end-of-stage polish
#
# nohup python bench_relaxation_polish.py --freq <freq.mat> > exp05.log 2>&1 &

import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import gbcore as gb  # noqa: E402
import macromodel as mm  # noqa: E402

p = gb.common_parser("Exp05 relaxation polish", "numpy,fast,fast_polish",
                     "MinTime/Experiments/05_continuous_relaxation_polish")
p.add_argument("--steps", type=int, default=40, help="Adam steps per polish")
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--pnorm", type=float, default=32.0, help="smooth-max exponent")
p.add_argument("--snap-k", type=int, default=5, help="nearest parts tried per cap")
p.add_argument("--snap-passes", type=int, default=2)
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
if prob.freq is None:
    raise SystemExit("Exp05 needs --freq (RLC fit of the library)")
dev, cd, T = prob.device, prob.cdtype, prob.TARGET_PORT

Rc, Lc, Cc, rlc_err = mm.fit_rlc(prob.d_np, prob.freq)
feat = np.stack([np.log(Rc), np.log(Lc), np.log(Cc)], 1)
mu, sd = feat.mean(0), feat.std(0) + 1e-12
FEAT = torch.tensor((feat - mu) / sd, dtype=torch.float64, device=dev)     # (C,3) normalised
MU = torch.tensor(mu, dtype=torch.float64, device=dev)
SD = torch.tensor(sd, dtype=torch.float64, device=dev)
S = torch.tensor(1j * 2 * np.pi * prob.freq, dtype=cd, device=dev)        # (F,)
with open(os.path.join(args.out, "rlc_fit.json"), "w") as f:
    json.dump({"rlc_err_median": float(np.median(rlc_err)), "rlc_err_p95": float(np.percentile(rlc_err, 95)),
               "rlc_err_max": float(rlc_err.max())}, f, indent=2)


def smooth_peak(theta, ports):
    """theta (k,3) normalised log-RLC; differentiable p-norm peak of |Z11|."""
    lg = theta * SD + MU
    R, L, C = lg[:, 0].exp(), lg[:, 1].exp(), lg[:, 2].exp()
    zc = R[None] + S[:, None] * L[None] + 1.0 / (S[:, None] * C[None])            # (F,k)
    pt = ports
    M = prob.Z0[:, pt[:, None], pt[None, :]] + torch.diag_embed(zc)
    x = torch.linalg.solve(M, prob.Z0[:, pt, T].unsqueeze(-1))
    val = prob.Z0[:, T, T] - (prob.Z0[:, T, pt].unsqueeze(-2) @ x)[..., 0, 0]
    a = val.abs()
    sc = a.detach().max()
    return sc * ((a / sc) ** args.pnorm).mean() ** (1.0 / args.pnorm), a.max()


def polish_hook(prob_, ev, n, gbest, gpart):
    m, pp = prob.decode_config(gpart, n)
    pt = torch.tensor(pp, device=dev)
    theta = FEAT[torch.tensor(m, device=dev)].clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    for _ in range(args.steps):
        opt.zero_grad()
        loss, _ = smooth_peak(theta, pt)
        loss.backward()
        opt.step()
    # snap: candidate parts per cap
    with torch.no_grad():
        dist = torch.cdist(theta.detach(), FEAT)                                     # (k,C)
        cand = dist.topk(min(args.snap_k, prob.C), largest=False).indices.cpu().numpy()
    cur = cand[:, 0].copy()
    extra = 1
    cur_val = prob.wb_peak(torch.tensor(cur[None], device=dev), pt[None])[0].item()
    for _ in range(args.snap_passes):
        improved = False
        for j in range(n):
            trial = np.tile(cur, (cand.shape[1], 1))
            trial[:, j] = cand[j]
            v = prob.wb_peak(torch.tensor(trial, device=dev), pt.expand(len(trial), -1))[0].cpu().numpy()
            extra += len(trial)
            b = int(np.argmin(v))
            if v[b] < cur_val:
                cur, cur_val, improved = trial[b].copy(), float(v[b]), True
        if not improved:
            break
    hook_stats["polish_calls"] += 1
    if cur_val < gbest:
        hook_stats["improved"] += 1
        hook_stats["gain_rel"].append((gbest - cur_val) / gbest)
        newp = gpart.copy()
        newp[:n] = (cur + 0.5) / prob.C            # model genes -> chosen parts (ports unchanged)
        # ports are kept as decoded: re-encode port genes to the resolved ports
        vp = {int(v): i for i, v in enumerate(prob.VALID_PORTS)}
        newp[n:] = [(vp[int(q)] + 0.5) / len(prob.VALID_PORTS) for q in pp]
        return cur_val, newp, extra
    return gbest, gpart, extra


hook_stats = {"polish_calls": 0, "improved": 0, "gain_rel": []}
fac = gb.standard_evaluators(prob)
fac["fast_polish"] = fac["fast"]
methods = gb.select_methods(args, fac)
gb.verify(prob, {m: fac[m] for m in methods}, args.out)

# gradient sanity check (finite difference on one config)
m0, p0 = gb.random_configs(prob, 1, seed=2, kmax=3)[0]
th = FEAT[torch.tensor(m0, device=dev)].clone().requires_grad_(True)
l0, _ = smooth_peak(th, torch.tensor(p0, device=dev))
l0.backward()
eps = 1e-6
th2 = th.detach().clone()
th2[0, 2] += eps
fd = (smooth_peak(th2, torch.tensor(p0, device=dev))[0].item() - l0.item()) / eps
print(f"[verify] grad check d/dlogC: autograd {th.grad[0, 2].item():.4e} vs FD {fd:.4e}")

gb.warmup(prob, {m: fac[m] for m in methods})
_n = min(3, len(prob.VALID_PORTS))                 # warm up autograd/Adam kernels (untimed)
polish_hook(prob, None, _n, np.inf, np.random.default_rng(0).random(2 * _n))
hook_stats.update({"polish_calls": 0, "improved": 0, "gain_rel": []})
recs = gb.run_benchmark(prob, fac, methods, args.out, stage_hooks={"fast_polish": polish_hook})
gb.analyse(recs, args.out, args.baseline)
hs = dict(hook_stats)
hs["gain_rel_mean"] = float(np.mean(hs["gain_rel"])) if hs["gain_rel"] else 0.0
hs["gain_rel"] = hs["gain_rel"][:200]
with open(os.path.join(args.out, "polish_stats.json"), "w") as f:
    json.dump(hs, f, indent=2)
print(f"\n[polish] calls={hs['polish_calls']} improved={hs['improved']} mean gain={hs['gain_rel_mean']:.3%}")
print("Done.")
