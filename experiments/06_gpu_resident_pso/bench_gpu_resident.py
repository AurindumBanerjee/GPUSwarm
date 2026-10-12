# Experiment 06 -- GPU-resident PSO.
#
# After batching the fitness (GPUBench2), the per-iteration cost is dominated
# by host work: NumPy RNG, per-particle decoding and the Python
# port-collision loop, and host<->device copies (DPCrypto's lesson: once the
# maths is fast, the bottleneck moves). Here the whole PSO state lives on the
# GPU as tensors:
#   * RNG: torch.Generator seeded per (run, n_caps) -> paired across methods
#     of THIS script, but NOT bit-identical to the NumPy stream of the
#     baseline, so trajectories differ (compare time, success, caps, evals).
#   * decode: floor(frac*K) on tensors (FIX 2/3).
#   * port resolution: vectorised over particles, looping only over the k
#     capacitor slots, using a precomputed candidate table
#     [p, p+1, p-1, p+2, p-2, ...] -- the exact search order of FIX 12.
#   * fitness: batched Woodbury (optionally with exact pruning).
#   * pbest/gbest: the sequential scan with immediate stop is reproduced
#     with cumulative minima: the first particle whose running best meets
#     the target ends the iteration; only particles up to it update.
#   * one host sync per iteration (the stop test).
#
# Methods
#   numpy             baseline (host PSO)
#   fast              GPUBench2 fast (host PSO)
#   gpu_pso           GPU-resident PSO, batched Woodbury
#   gpu_pso_prune     + exact lower-bound pruning
#
# nohup python bench_gpu_resident.py > exp06.log 2>&1 &

import os
import sys
import time
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import gbcore as gb  # noqa: E402

p = gb.common_parser("Exp06 GPU-resident PSO", "numpy,fast,gpu_pso,gpu_pso_prune",
                     "MinTime/Experiments/06_gpu_resident_pso")
args, _ = p.parse_known_args()
os.makedirs(args.out, exist_ok=True)

prob = gb.Problem(args)
cfg = prob.cfg
dev, T, N = prob.device, prob.TARGET_PORT, prob.N
VALID = torch.tensor(prob.VALID_PORTS, device=dev)
NV = len(prob.VALID_PORTS)

# candidate table in FIX-12 order: p, p+1, p-1, p+2, p-2, ... (-1 = invalid)
L = 2 * N - 1
cand = np.full((N, L), -1, np.int64)
for p0 in range(N):
    seq = [p0]
    for off in range(1, N):
        seq += [p0 + off, p0 - off]
    cand[p0] = [c if 0 <= c < N else -1 for c in seq]
CAND = torch.tensor(cand, device=dev)


def resolve_ports_gpu(ports):
    P, k = ports.shape
    used = torch.zeros(P, N, dtype=torch.bool, device=dev)
    used[:, T] = True
    out = ports.clone()
    ar = torch.arange(P, device=dev)
    for i in range(k):
        c = CAND[out[:, i]]                                   # (P,L)
        valid = c >= 0
        free = valid & ~used.gather(1, c.clamp(min=0))
        has = free.any(1)
        first = free.int().argmax(1)
        new = torch.where(has, c[ar, first], out[:, i])
        out[:, i] = new
        used[ar, new] = True
    return out


def decode_gpu(x, n):
    mi = (x[:, :n] * prob.C).floor().long().clamp(0, prob.C - 1)
    pi = (x[:, n:] * NV).floor().long().clamp(0, NV - 1)
    return mi, resolve_ports_gpu(VALID[pi])


def seed_gpu(n, prev_best, g):
    P = cfg.N_PARTICLES
    nw = min(cfg.N_WARM_START, P)
    rand = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float64)
    if prev_best is None or n == 1 or nw == 0:
        return rand(P, 2 * n)
    pn = n - 1
    warm = torch.cat([prev_best[:pn].expand(nw, -1), rand(nw, 1),
                      prev_best[pn:].expand(nw, -1), rand(nw, 1)], 1)
    if cfg.WARM_START_JITTER > 0 and nw > 1:
        noise = torch.randn(warm.shape, generator=g, device=dev, dtype=torch.float64) * cfg.WARM_START_JITTER
        noise[0] = 0
        noise[:, pn] = 0
        noise[:, -1] = 0
        warm = (warm + noise).clamp(0, 1)
    return torch.cat([warm, rand(P - nw, 2 * n)], 0) if P > nw else warm


class GPUPSO:
    """Marker evaluator: run_pso_gpu does the work."""
    sequential = False

    def __init__(self, prune):
        self.prune = prune
        self.reset()

    def reset(self):
        self.n_full = self.n_pruned = 0
        self.subset = torch.tensor(prob.BASE_SUBSET, device=dev)

    def costs(self, mi, pi, pbest_val):
        if not self.prune:
            self.n_full += mi.shape[0]
            return prob.wb_peak(mi, pi)[0]
        lb = prob.wb_peak(mi, pi, self.subset)[0] * (1 - 1e-9)
        need = lb < pbest_val
        c = lb.clone()
        idx = need.nonzero(as_tuple=True)[0]
        if idx.numel():
            pk, am = prob.wb_peak(mi[idx], pi[idx])
            c[idx] = pk
            self.subset = torch.unique(torch.cat([self.subset, am]))
        self.n_full += int(idx.numel())
        self.n_pruned += int(mi.shape[0] - idx.numel())
        return c

    def stats(self):
        return {"gpu_full_evals": self.n_full, "pruned": self.n_pruned}


def run_pso_gpu(ev, method, thresholds, run_id):
    thr_sorted = sorted(thresholds, reverse=True)
    final = thr_sorted[-1]
    ev.reset()
    P = cfg.N_PARTICLES
    cross = {t: None for t in thr_sorted}
    stage_best = []
    best_cfg = None
    prob.sync()
    start = time.perf_counter()
    n_evals = 0
    prev_best = None
    g = torch.Generator(device=dev)
    for n in range(1, cfg.MAX_CAPS + 1):
        g.manual_seed(gb.BASE_SEED + run_id * 10000 + n)
        x = seed_gpu(n, prev_best, g)
        v = torch.zeros_like(x)
        pbest = x.clone()
        pbest_val = torch.full((P,), float("inf"), dtype=torch.float64, device=dev)
        gbest = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
        gpart = None
        stop = False
        for it in range(cfg.N_ITERATIONS):
            mi, pi = decode_gpu(x, n)
            c = ev.costs(mi, pi, pbest_val).double()
            run = torch.minimum(torch.cummin(c, 0).values, gbest)
            hit = run <= final
            any_hit = bool(hit.any())                                     # one sync / iteration
            s = int(hit.int().argmax()) if any_hit else P - 1
            cons = torch.arange(P, device=dev) <= s
            n_evals += s + 1
            imp = (c < pbest_val) & cons
            pbest_val = torch.where(imp, c, pbest_val)
            pbest = torch.where(imp[:, None], x, pbest)
            cm = torch.where(cons, c, torch.full_like(c, float("inf")))
            j = int(cm.argmin())
            if cm[j] < gbest:
                gbest = cm[j].clone()
                gpart = x[j].clone()
                gv = float(gbest)
                for t in thr_sorted:
                    if cross[t] is None and gv <= t:
                        prob.sync()
                        cross[t] = (time.perf_counter() - start, n, n_evals, gv)
            if any_hit:
                stop = True
                break
            w = cfg.W_MAX - (cfg.W_MAX - cfg.W_MIN) * (it / cfg.N_ITERATIONS)
            r1 = torch.rand(x.shape, generator=g, device=dev, dtype=torch.float64)
            r2 = torch.rand(x.shape, generator=g, device=dev, dtype=torch.float64)
            v = w * v + cfg.C1 * r1 * (pbest - x) + cfg.C2 * r2 * (gpart - x)
            x = (x + v).clamp(0, 1)
        if gpart is None:
            prev_best = None
            continue
        prev_best = gpart
        mi, pi = decode_gpu(gpart[None], n)
        best_cfg = (mi[0].tolist(), pi[0].tolist())
        stage_best.append(float(gbest))
        if stop or float(gbest) <= final:
            break
    prob.sync()
    total = time.perf_counter() - start
    recs = []
    for t in thr_sorted:
        c = cross[t]
        r = {"threshold": t, "method": method, "run_id": run_id, "success": c is not None,
             "time_to_target_s": c[0] if c else None, "total_wall_time_s": c[0] if c else total,
             "caps_at_target": c[1] if c else None, "n_evals": c[2] if c else n_evals,
             "best_minz_ohm": c[3] if c else (min(stage_best) if stage_best else float("inf")),
             "init_time_s": prob.init_time, "shared_run": len(thr_sorted) > 1, "best_config": best_cfg}
        r.update(ev.stats())
        recs.append(r)
    return recs


# ---------------- checks ----------------
fac = gb.standard_evaluators(prob)
host_methods = [m for m in gb.select_methods(args, {**fac, "gpu_pso": 0, "gpu_pso_prune": 0})
                if m in fac]
gpu_methods = [m for m in args.methods.split(",") if m.strip() in ("gpu_pso", "gpu_pso_prune")]
gb.verify(prob, {m: fac[m] for m in host_methods}, args.out)

rng = np.random.default_rng(4)
mism = 0
for _ in range(200):                          # vectorised decode == reference decode
    n = int(rng.integers(1, min(cfg.MAX_CAPS, NV) + 3))
    xv = rng.random((4, 2 * n))
    mi, pi = decode_gpu(torch.tensor(xv, device=dev), n)
    for r in range(4):
        m_ref, p_ref = prob.decode_config(xv[r], n)
        mism += int(not (np.array_equal(mi[r].cpu().numpy(), m_ref) and np.array_equal(pi[r].cpu().numpy(), p_ref)))
print(f"[verify] GPU decode/port-resolution mismatches vs reference: {mism} / 800")
assert mism == 0

gb.warmup(prob, {m: fac[m] for m in host_methods})
for m in gpu_methods:                         # warm-up GPU PSO (untimed)
    run_pso_gpu(GPUPSO(m.endswith("prune")), m, [cfg.TARGETS[-1]], 999)

recs = gb.run_benchmark(prob, fac, host_methods, args.out) if host_methods else []
jl = os.path.join(args.out, "results_long.jsonl")
for t in cfg.TARGETS:
    for m in gpu_methods:
        print(f"\nMETHOD {m} TARGET {t}", flush=True)
        ev = GPUPSO(m.endswith("prune"))
        for run in range(1, cfg.NUM_RUNS + 1):
            rs = run_pso_gpu(ev, m, [t], run)
            recs.extend(rs)
            with open(jl, "a") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            r = rs[0]
            print(f"  {m:<16} T={t:<6} run {run:2d}: {r['total_wall_time_s']:.3f}s caps={r['caps_at_target']} "
                  f"minZ={r['best_minz_ohm']:.5f} evals={r['n_evals']}", flush=True)
gb.analyse(recs, args.out, args.baseline)
print("Done.")
