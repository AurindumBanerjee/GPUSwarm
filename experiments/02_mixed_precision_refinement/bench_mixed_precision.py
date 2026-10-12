# Experiment 02 -- mixed-precision Woodbury with FP64 iterative refinement.
#
# On GA102 cards (RTX A5000) FP64 runs at 1/32 of the FP32 rate and there are
# no FP64 tensor cores, so a complex128 batched solve leaves most of the GPU
# idle. Following the TC-GVF idea (do the bulk in fast low precision, recover
# full accuracy cheaply):
#
#   full value : LU-factorise M in complex64, solve, then R steps of
#                iterative refinement with the residual r = z - M x formed in
#                complex128 (each step reuses the complex64 LU factors).
#                Converges to ~FP64 accuracy when cond(M)*eps32 << 1.
#   screen     : pruning lower bound computed entirely in complex64 and
#                deflated by a safety margin, so it stays a valid lower bound.
#
# Methods
#   numpy      baseline (complex128 inv)
#   fast       GPUBench2 fast (complex128, cache + prune)
#   wb_mp      batched Woodbury, mixed precision + refinement, no cache/prune
#   fast_mp    wb_mp + cache + complex64 screening
#
# The verification step reports the refinement accuracy and the observed
# complex64 screening error; if the observed error exceeds --screen-margin the
# script warns (raise the margin -- pruning stays exact only if margin >= error).
#
# nohup python bench_mixed_precision.py > exp02.log 2>&1 &

import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import gbcore as gb  # noqa: E402

p = gb.common_parser("Exp02 mixed precision", "numpy,fast,wb_mp,fast_mp",
                     "MinTime/Experiments/02_mixed_precision_refinement")
p.add_argument("--refine", type=int, default=2, help="FP64 refinement steps")
p.add_argument("--screen-margin", type=float, default=1e-2,
               help="relative deflation of the complex64 lower bound")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
assert prob.cdtype == torch.complex128, "run with --dtype complex128 (refinement target precision)"
Z0_32 = prob.Z0.to(torch.complex64)
DINV_32 = prob.DINV.to(torch.complex64)
T = prob.TARGET_PORT


def mp_peak(mt, pt, refine=args.refine):
    """Woodbury peak: complex64 LU + complex128 residual refinement."""
    B, k = pt.shape
    F = prob.F
    chunk = max(1, int(args.max_batch_elems // max(1, F * k * k)))
    out_pk, out_am = [], []
    for s in range(0, B, chunk):
        p_, m_ = pt[s:s + chunk], mt[s:s + chunk]
        M = prob.Z0[:, p_.unsqueeze(2), p_.unsqueeze(1)].permute(1, 0, 2, 3)
        M = M + torch.diag_embed(prob.DINV[m_].transpose(1, 2))           # complex128
        zc = prob.Z0[:, p_, T].permute(1, 0, 2).unsqueeze(-1)
        zr = prob.Z0[:, T, p_].permute(1, 0, 2).unsqueeze(-2)
        LU, piv, info = torch.linalg.lu_factor_ex(M.to(torch.complex64))
        x = torch.linalg.lu_solve(LU, piv, zc.to(torch.complex64)).to(torch.complex128)
        for _ in range(refine):
            r = zc - M @ x
            x = x + torch.linalg.lu_solve(LU, piv, r.to(torch.complex64)).to(torch.complex128)
        val = prob.Z0[:, T, T].unsqueeze(0) - (zr @ x)[..., 0, 0]
        pk, am = val.abs().max(dim=1)
        bad = (info != 0).any(dim=1) | ~torch.isfinite(pk)
        out_pk.append(torch.where(bad, torch.full_like(pk, 1e200), pk))
        out_am.append(am)
    return torch.cat(out_pk), torch.cat(out_am)


def screen32(mt, pt, fidx):
    return prob.wb_peak(mt, pt, fidx, Z0=Z0_32, DINV=DINV_32)[0]


fac = gb.standard_evaluators(prob)
fac["wb_mp"] = lambda: gb.SwarmEvaluator(prob, False, False, full_fn=mp_peak)
fac["fast_mp"] = lambda: gb.SwarmEvaluator(prob, True, True, full_fn=mp_peak, screen_fn=screen32,
                                           screen_margin=args.screen_margin)
methods = gb.select_methods(args, fac)

rep = gb.verify(prob, {m: fac[m] for m in methods}, args.out, tol=1e-9)

# accuracy study: refinement steps and complex64 screen error on random configs
cfgs = gb.random_configs(prob, 20, seed=7, kmax=min(prob.cfg.MAX_CAPS, len(prob.VALID_PORTS)))
acc = {"refine_steps": {}, "screen32_max_rel_err": 0.0}
for R in range(0, 4):
    e = 0.0
    for m, pp in cfgs:
        mt = torch.tensor(m[None], device=prob.device)
        pt = torch.tensor(pp[None], device=prob.device)
        ref = prob.wb_peak(mt, pt)[0].item()
        got = mp_peak(mt, pt, refine=R)[0].item()
        e = max(e, abs(got - ref) / ref)
    acc["refine_steps"][R] = e
for m, pp in cfgs:
    mt = torch.tensor(m[None], device=prob.device)
    pt = torch.tensor(pp[None], device=prob.device)
    full = torch.arange(prob.F, device=prob.device)
    ref = prob.wb_peak(mt, pt)[0].item()
    got = float(screen32(mt, pt, full)[0].item())
    acc["screen32_max_rel_err"] = max(acc["screen32_max_rel_err"], abs(got - ref) / ref)
print("[accuracy]", json.dumps(acc))
if acc["screen32_max_rel_err"] > args.screen_margin:
    print(f"[WARN] complex64 screen error {acc['screen32_max_rel_err']:.2e} > margin "
          f"{args.screen_margin:.0e}; raise --screen-margin to keep pruning exact")
with open(os.path.join(args.out, "accuracy.json"), "w") as f:
    json.dump(acc, f, indent=2)

gb.warmup(prob, {m: fac[m] for m in methods})
recs = gb.run_benchmark(prob, fac, methods, args.out)
gb.analyse(recs, args.out, args.baseline)
print("Done.")
