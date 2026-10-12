# Experiment 03 -- incremental (refactorisation-style) Woodbury updates.
#
# Sparse-LU papers (Lee & Achar, TVLSI'18 / TCAD'21) split work into a
# one-time symbolic step and cheap numeric refactorisations because the
# sparsity pattern never changes. Our analogue: for a particle whose PORT
# set is unchanged since the previous iteration, the Woodbury core
#     M = Z0[P,P] + diag(1/d)
# only changes in the diagonal entries of the capacitors whose MODEL
# changed. Each such change is a rank-1 update of M, so the stored inverse
# M^-1 is updated by Sherman-Morrison in O(k^2) per frequency instead of a
# fresh O(k^3) factorisation:
#     delta_j = 1/d_new - 1/d_old
#     M^-1 <- M^-1 - delta_j (M^-1 e_j)(e_j^T M^-1) / (1 + delta_j M^-1_jj)
# Z11 = Z0_tt - Z0[t,P] M^-1 Z0[P,t]  (O(k^2)).
#
# A particle gets a full (inverse) evaluation when: it has no state, its
# ports changed, more than --max-rank models changed, or it has had
# --refresh consecutive updates (bounds rounding drift).
#
# Methods
#   numpy           baseline
#   woodbury_batch  GPUBench2 batched solve (no cache/prune)
#   incr            incremental updates
#
# Reported extra stats: incremental vs full evaluation counts and the
# measured drift of updated inverses against fresh ones (--drift-check).
#
# nohup python bench_incremental.py > exp03.log 2>&1 &

import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import gbcore as gb  # noqa: E402

p = gb.common_parser("Exp03 incremental Woodbury", "numpy,woodbury_batch,incr",
                     "MinTime/Experiments/03_incremental_woodbury_updates")
p.add_argument("--max-rank", type=int, default=4, help="max changed models for an update")
p.add_argument("--refresh", type=int, default=8, help="full re-inverse after this many updates")
p.add_argument("--drift-check", type=int, default=50,
               help="every N evaluate() calls compare one updated particle with a fresh inverse")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
T = prob.TARGET_PORT


class IncrementalEvaluator:
    sequential = False

    def __init__(self):
        self.reset()

    def reset(self):
        self.k = None
        self.Minv = None
        self.n_full = self.n_incr = self.calls = 0
        self.drift = 0.0

    def _core(self, mt, pt):
        M = prob.Z0[:, pt.unsqueeze(2), pt.unsqueeze(1)].permute(1, 0, 2, 3)
        return M + torch.diag_embed(prob.DINV[mt].transpose(1, 2))

    def _value(self, Minv, pt):
        zc = prob.Z0[:, pt, T].permute(1, 0, 2).unsqueeze(-1)
        zr = prob.Z0[:, T, pt].permute(1, 0, 2).unsqueeze(-2)
        val = prob.Z0[:, T, T].unsqueeze(0) - (zr @ Minv @ zc)[..., 0, 0]
        pk = val.abs().max(dim=1).values
        return torch.where(torch.isfinite(pk), pk, torch.full_like(pk, 1e200))

    def evaluate(self, models, ports, pbest_val):
        dev = prob.device
        P, k = models.shape
        self.calls += 1
        mt = torch.from_numpy(models).to(dev)
        pt = torch.from_numpy(ports).to(dev)
        if self.k != k or self.Minv is None:              # new stage: no state
            self.k = k
            self.Minv = torch.zeros(P, prob.F, k, k, dtype=prob.cdtype, device=dev)
            self.prev_m = np.full_like(models, -1)
            self.prev_p = np.full_like(ports, -1)
            self.age = np.full(P, 10 ** 9)

        same_ports = (ports == self.prev_p).all(1)
        chg = models != self.prev_m
        nchg = chg.sum(1)
        upd = same_ports & (nchg <= args.max_rank) & (self.age < args.refresh)
        full = ~upd

        if full.any():
            idx = torch.from_numpy(np.nonzero(full)[0]).to(dev)
            Minv, info = torch.linalg.inv_ex(self._core(mt[idx], pt[idx]))
            self.Minv[idx] = Minv
            self.age[full] = 0
            self.n_full += int(full.sum())
        if upd.any():
            ui = np.nonzero(upd)[0]
            idx = torch.from_numpy(ui).to(dev)
            Mi = self.Minv[idx]                                        # (b,F,k,k)
            old = torch.from_numpy(self.prev_m[ui]).to(dev)
            new = mt[idx]
            cmask = torch.from_numpy(chg[ui]).to(dev)
            for j in np.nonzero(chg[ui].any(0))[0]:
                delta = (prob.DINV[new[:, j]] - prob.DINV[old[:, j]])   # (b,F)
                delta = delta * cmask[:, j].unsqueeze(1)
                u = Mi[:, :, :, j]                                     # (b,F,k)
                v = Mi[:, :, j, :]
                den = 1.0 + delta * Mi[:, :, j, j]
                Mi = Mi - (delta / den).unsqueeze(-1).unsqueeze(-1) * (u.unsqueeze(-1) * v.unsqueeze(-2))
            self.Minv[idx] = Mi
            self.age[upd] += 1
            self.n_incr += int(upd.sum())
            if args.drift_check and self.calls % args.drift_check == 0:
                j0 = ui[:1]
                t0 = torch.from_numpy(j0).to(dev)
                fresh = torch.linalg.inv(self._core(mt[t0], pt[t0]))
                a = self._value(fresh, pt[t0])
                b = self._value(self.Minv[t0], pt[t0])
                self.drift = max(self.drift, float(((a - b).abs() / a).max().item()))

        self.prev_m, self.prev_p = models.copy(), ports.copy()
        return self._value(self.Minv, pt).double().cpu().numpy()

    def stats(self):
        return {"gpu_full_evals": self.n_full, "incremental_evals": self.n_incr,
                "max_drift_rel": self.drift}


fac = gb.standard_evaluators(prob)
fac["incr"] = IncrementalEvaluator
methods = gb.select_methods(args, fac)
gb.verify(prob, {m: fac[m] for m in methods}, args.out)

# unit check: incremental path vs fresh after a sequence of random model swaps
rng = np.random.default_rng(3)
m, pp = gb.random_configs(prob, 1, seed=11, kmax=4)[0]
ev = IncrementalEvaluator()
M = np.tile(m, (4, 1))
Pp = np.tile(pp, (4, 1))
ev.evaluate(M, Pp, np.full(4, np.inf))
worst = 0.0
for step in range(6):
    M = M.copy()
    for r in range(4):
        M[r, rng.integers(0, M.shape[1])] = rng.integers(0, prob.C)
    got = ev.evaluate(M, Pp, np.full(4, np.inf))
    ref = np.array([prob.cpu_ref(list(zip(M[r].tolist(), Pp[r].tolist()))) for r in range(4)])
    worst = max(worst, float(np.max(np.abs(got - ref) / ref)))
print(f"[verify] incremental swap sequence max_rel={worst:.3e} (full={ev.n_full} incr={ev.n_incr})")
with open(os.path.join(args.out, "incremental_check.json"), "w") as f:
    json.dump({"swap_sequence_max_rel": worst}, f)

gb.warmup(prob, {m_: fac[m_] for m_ in methods})
recs = gb.run_benchmark(prob, fac, methods, args.out)
df, s = gb.analyse(recs, args.out, args.baseline)
if "incremental_evals" in df:
    g = df[df.method == "incr"]
    print(f"\n[incr] full={g.gpu_full_evals.mean():.0f} incremental={g.incremental_evals.mean():.0f} "
          f"per run, max drift {g.max_drift_rel.max():.2e}")
print("Done.")
