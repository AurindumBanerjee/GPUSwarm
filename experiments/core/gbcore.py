"""
gbcore -- shared engine for the GPUBench experiments.

Everything that GPUBench2.py does, refactored into importable pieces so each
experiment only has to define what is new:

  Problem        data loading (PDN Y(f), shunt decap library, optional freq
                 grid, band restriction), device tensors, base impedance
                 Z0 = Y^-1, discretisation (FIX 2/3/12), reference kernels.
  wb_peak        batched exact Woodbury fitness (GPUBench2 NEW 1/2).
  Evaluators     SeqEvaluator (numpy/solve/sm/woodbury, one particle at a
                 time, immediate stop) and SwarmEvaluator (whole swarm per
                 iteration, optional cache + exact lower-bound pruning).
  run_pso        the PSO of GPUBench/GPUBench2 (paired per-stage seeding,
                 hybrid warm start, immediate stop) extended to record the
                 crossing of SEVERAL thresholds in one run, plus an optional
                 end-of-stage hook.
  run_benchmark  loops thresholds x methods x runs, writes results.
  analyse        summary table, paired speedups, trajectory-match rate,
                 figures.

Decap library: the PSO adds the decap ADMITTANCE d(f) on the diagonal of
Y at the chosen pad, i.e. every capacitor is modelled as a SHUNT element
(pad to ground). Use the shunt-fixture library (decaps_sp.mat for the SP
datasets; decaps.mat for the reference y2 grid).
"""

import os
import time
import json
import math
import logging
import argparse

import numpy as np
import scipy.io as sio
import torch

DATA_DIR = "/DATA/Aurindum/Swarming/Data"
BASE_SEED = 12345
FOLDER_SEP = "__THR__"


# ============================================================
# ARGS / CONFIG
# ============================================================

def common_parser(description, default_methods, default_out):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--y", default=os.path.join(DATA_DIR, "y2.mat"),
                   help="PDN admittance .mat, variable y (N,N,F); port index 0 = observation")
    p.add_argument("--d", default=os.path.join(DATA_DIR, "decaps.mat"),
                   help="SHUNT decap library .mat, variable decaps (C,F) admittance")
    p.add_argument("--freq", default=None, help="frequency grid .mat, variable freq (1,F)")
    p.add_argument("--fmax", type=float, default=None, help="drop frequencies above (Hz)")
    p.add_argument("--fmin", type=float, default=None, help="drop frequencies below (Hz)")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    p.add_argument("--quick", action="store_true", help="tiny schedule for smoke tests")
    p.add_argument("--methods", default=default_methods)
    p.add_argument("--baseline", default="numpy")
    p.add_argument("--targets", default="0.05,0.045,0.04,0.03")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--out", default=default_out)
    p.add_argument("--stride", type=int, default=16, help="pruning subset stride")
    p.add_argument("--max-batch-elems", type=float, default=3e7,
                   help="cap on B*F*k*k elements per batched solve (memory)")
    p.add_argument("--max-caps", type=int, default=20)
    p.add_argument("--particles", type=int, default=50)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p


class Cfg:
    def __init__(self, args, synthetic):
        self.MAX_CAPS = args.max_caps
        self.N_PARTICLES = args.particles
        self.N_ITERATIONS = args.iterations
        self.W_MAX, self.W_MIN = 0.9, 0.4
        self.C1, self.C2 = 1.5, 1.5
        self.NUM_RUNS = args.runs
        self.TARGETS = [float(t) for t in args.targets.split(",")]
        if args.quick or synthetic:
            self.NUM_RUNS = min(self.NUM_RUNS, 3)
            self.TARGETS = self.TARGETS[:2]
            self.MAX_CAPS = min(self.MAX_CAPS, 5)
            self.N_PARTICLES = min(self.N_PARTICLES, 8)
            self.N_ITERATIONS = min(self.N_ITERATIONS, 4)
        self.N_WARM_START = int(round(0.4 * self.N_PARTICLES))
        self.WARM_START_JITTER = 0.02


# ============================================================
# PROBLEM
# ============================================================

def synthetic_pdn(n_nodes=8, n_freq=40, n_cap_models=150, seed=0):
    rng = np.random.default_rng(seed)
    freqs = np.logspace(4, 8, n_freq)
    w = 2.0 * np.pi * freqs
    R_sh = rng.uniform(5.0, 20.0, n_nodes)
    L_sh = rng.uniform(1e-9, 8e-9, n_nodes)
    C_sh = rng.uniform(1e-10, 5e-10, n_nodes)
    R_ln = rng.uniform(0.02, 0.08, n_nodes)
    L_ln = rng.uniform(0.5e-9, 3e-9, n_nodes)
    y = np.zeros((n_freq, n_nodes, n_nodes), dtype=np.complex128)
    Ysh = (1 / R_sh)[None] + 1 / (1j * w[:, None] * L_sh[None]) + 1j * w[:, None] * C_sh[None]
    Yln = 1 / (R_ln[None] + 1j * w[:, None] * L_ln[None])
    for k in range(n_nodes):
        j = (k + 1) % n_nodes
        y[:, k, k] += Ysh[:, k] + Yln[:, k]
        y[:, j, j] += Yln[:, k]
        y[:, k, j] -= Yln[:, k]
        y[:, j, k] -= Yln[:, k]
    y += 1e-6 * np.eye(n_nodes)[None]
    d = np.zeros((n_cap_models, n_freq), dtype=np.complex128)
    for m in range(n_cap_models):
        c = rng.integers(0, 5)
        C = 10 ** rng.uniform(-9 + c * 0.4, -6 + c * 0.5)
        L = 10 ** rng.uniform(-10.3 + c * 0.15, -9.6 + c * 0.15)
        R = 10 ** rng.uniform(-2.3, -0.7)
        d[m] = 1 / (R + 1j * w * L + 1 / (1j * w * C))      # shunt RLC admittance
    return y, d, freqs


class Problem:
    TARGET_PORT = 0

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.cdtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
        self.real_data = os.path.exists(args.y) and os.path.exists(args.d)
        print(f"[GPU] device={self.device} dtype={self.cdtype}")
        if self.device.type == "cuda":
            print(f"[GPU] {torch.cuda.get_device_name(0)} torch={torch.__version__}")

        t0 = time.perf_counter()
        if self.real_data:
            y = np.transpose(sio.loadmat(args.y)["y"], (2, 0, 1))
            dm = sio.loadmat(args.d)
            d = dm["decaps"]
            if "fixture" in dm:
                fx = {str(np.asarray(x).ravel()[0]) for x in dm["fixture"].ravel()}
                if fx != {"shunt"}:
                    print(f"[WARN] library fixtures {fx}: the PSO models caps as SHUNT elements; "
                          f"use the shunt entries (decaps_sp.mat).")
            freq = sio.loadmat(args.freq)["freq"].ravel() if args.freq else None
        else:
            print(f"[WARN] {args.y} / {args.d} not found -> synthetic PDN")
            y, d, freq = synthetic_pdn()
        assert y.shape[0] == d.shape[1], f"PDN has {y.shape[0]} freqs, library {d.shape[1]}"
        if args.fmax is not None or args.fmin is not None:
            assert freq is not None, "--fmax/--fmin need --freq"
            keep = np.ones(len(freq), bool)
            if args.fmax is not None:
                keep &= freq <= args.fmax
            if args.fmin is not None:
                keep &= freq >= args.fmin
            y, d, freq = y[keep], d[:, keep], freq[keep]
            print(f"[band] {freq[0]:.4g}-{freq[-1]:.4g} Hz ({keep.sum()} pts)")
        self.y_np = np.ascontiguousarray(y, dtype=np.complex128)
        self.d_np = np.ascontiguousarray(d, dtype=np.complex128)
        self.freq = None if freq is None else np.asarray(freq, float)
        self.F, self.N, _ = self.y_np.shape
        self.C = self.d_np.shape[0]
        self.load_time = time.perf_counter() - t0
        print(f"N_FREQS={self.F} N_NODES={self.N} N_CAP_MODELS={self.C} real={self.real_data}")

        self.cfg = Cfg(args, not self.real_data)
        self.VALID_PORTS = np.array([p for p in range(self.N) if p != self.TARGET_PORT])

        t0 = time.perf_counter()
        dev, cd = self.device, self.cdtype
        self.Y = torch.from_numpy(self.y_np).to(dev, cd)
        self.D = torch.from_numpy(self.d_np).to(dev, cd)
        self.E1 = torch.zeros(self.F, self.N, 1, dtype=cd, device=dev)
        self.E1[:, self.TARGET_PORT, 0] = 1
        self.Z0 = torch.linalg.inv(self.Y)
        Ds = torch.where(self.D.abs() < 1e-300, torch.full_like(self.D, 1e-300), self.D)
        self.DINV = 1.0 / Ds                       # decap impedance table (C,F)
        self.sync()
        self.init_time = self.load_time + time.perf_counter() - t0

        bare = np.abs(np.linalg.inv(self.y_np)[:, 0, 0])
        lmax = [i for i in range(1, self.F - 1) if bare[i] >= bare[i - 1] and bare[i] >= bare[i + 1]]
        lmax = sorted(lmax, key=lambda i: -bare[i])[:16]
        self.BASE_SUBSET = sorted(set(range(0, self.F, max(1, args.stride)))
                                  | {0, self.F - 1} | set(lmax))

    # ---------------- utilities ----------------
    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def frac_to_index(self, frac, K):                       # FIX 2
        return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)

    def decode_particle(self, vec, n):                      # FIX 3
        return (self.frac_to_index(vec[:n], self.C),
                self.VALID_PORTS[self.frac_to_index(vec[n:], len(self.VALID_PORTS))])

    def resolve_ports(self, models, ports):                 # FIX 12
        used = {self.TARGET_PORT}
        for i in range(len(ports)):
            if ports[i] not in used:
                used.add(ports[i])
                continue
            placed = False
            for off in range(1, self.N):
                for c in (ports[i] + off, ports[i] - off):
                    if 0 <= c < self.N and c not in used:
                        ports[i] = c
                        used.add(c)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                used.add(ports[i])
        return models, ports

    def decode_config(self, vec, n):
        m, p = self.decode_particle(vec, n)
        m, p = self.resolve_ports(m, p)
        return m, p

    def decode_swarm(self, particles, n):
        P = particles.shape[0]
        M = np.empty((P, n), np.int64)
        Pt = np.empty((P, n), np.int64)
        for i in range(P):
            M[i], Pt[i] = self.decode_config(particles[i], n)
        return M, Pt

    # ---------------- kernels ----------------
    def assemble(self, config):
        A = self.Y.clone()
        for c, p in config:
            A[:, p, p] += self.D[c, :]
        return A

    def cpu_ref(self, config):
        peak = 0.0
        for f in range(self.F):
            A = self.y_np[f].copy()
            for c, p in config:
                A[p, p] += self.d_np[c, f]
            try:
                v = np.linalg.inv(A)[0, 0]
            except np.linalg.LinAlgError:
                return 1e200
            peak = max(peak, abs(v))
        return peak

    def wb_peak(self, models_t, ports_t, fidx=None, Z0=None, DINV=None):
        """Batched exact Woodbury peak |Z11|. models_t, ports_t (B,k) long.
        Returns (peak (B,), argmax_freq_global (B,)). Z0/DINV can be
        overridden (e.g. complex64 copies)."""
        Z0 = self.Z0 if Z0 is None else Z0
        DINV = self.DINV if DINV is None else DINV
        t = self.TARGET_PORT
        Z0f = Z0 if fidx is None else Z0.index_select(0, fidx)
        Dv = DINV if fidx is None else DINV.index_select(1, fidx)
        B, k = ports_t.shape
        Fp = Z0f.shape[0]
        chunk = max(1, int(self.args.max_batch_elems // max(1, Fp * k * k)))
        pk_all, am_all = [], []
        for s in range(0, B, chunk):
            pt, mt = ports_t[s:s + chunk], models_t[s:s + chunk]
            M = Z0f[:, pt.unsqueeze(2), pt.unsqueeze(1)].permute(1, 0, 2, 3)
            M = M + torch.diag_embed(Dv[mt].transpose(1, 2))
            zc = Z0f[:, pt, t].permute(1, 0, 2).unsqueeze(-1)
            zr = Z0f[:, t, pt].permute(1, 0, 2).unsqueeze(-2)
            x, info = torch.linalg.solve_ex(M, zc)
            val = Z0f[:, t, t].unsqueeze(0) - (zr @ x)[..., 0, 0]
            pk, am = val.abs().max(dim=1)
            bad = (info != 0).any(dim=1) | ~torch.isfinite(pk)
            big = 1e200 if pk.dtype == torch.float64 else 3e38
            pk_all.append(torch.where(bad, torch.full_like(pk, big), pk))
            am_all.append(am)
        pk, am = torch.cat(pk_all), torch.cat(am_all)
        if fidx is not None:
            am = fidx[am]
        return pk.real if pk.is_complex() else pk, am

    def seq_eval(self, config, method):
        if method == "cpu_numpy":
            return self.cpu_ref(config)
        t = self.TARGET_PORT
        try:
            if method == "woodbury":
                m = torch.tensor([[c for c, _ in config]], device=self.device)
                p = torch.tensor([[q for _, q in config]], device=self.device)
                return float(self.wb_peak(m, p)[0][0].item())
            if method == "numpy":
                v = torch.linalg.inv(self.assemble(config))[:, t, t]
            elif method == "solve":
                v = torch.linalg.solve(self.assemble(config), self.E1)[:, t, 0]
            elif method == "sm":
                Z = self.Z0.clone()
                for c, p in config:
                    val = self.D[c, :]
                    den = 1.0 + val * Z[:, p, p]
                    if torch.any(den.abs() < 1e-12):
                        raise torch.linalg.LinAlgError("SM denominator ~ 0")
                    Z = Z - (Z[:, :, p].unsqueeze(2) * (val.unsqueeze(1) * Z[:, p, :]).unsqueeze(1)) \
                        / den.view(-1, 1, 1)
                v = Z[:, t, t]
            else:
                raise ValueError(method)
            return float(v.abs().max().item())
        except torch.linalg.LinAlgError:
            return 1e200


# ============================================================
# EVALUATORS
# ============================================================

class SeqEvaluator:
    """One particle at a time (numpy / solve / sm / woodbury / cpu_numpy)."""
    sequential = True

    def __init__(self, prob, method):
        self.prob, self.method = prob, method

    def reset(self):
        pass

    def eval_one(self, models, ports):
        return self.prob.seq_eval(list(zip(models.tolist(), ports.tolist())), self.method)

    def stats(self):
        return {}


class SwarmEvaluator:
    """Whole swarm per iteration; optional cache and exact lower-bound
    pruning (GPUBench2 NEW 2-4). full_fn/screen_fn may be overridden:
    full_fn(mt, pt) -> (peak, argmax); screen_fn(mt, pt, fidx) -> lower bound
    peak (must be a valid lower bound of the full peak)."""
    sequential = False

    def __init__(self, prob, use_cache=False, use_prune=False, full_fn=None, screen_fn=None,
                 screen_margin=1e-9):
        self.prob = prob
        self.use_cache, self.use_prune = use_cache, use_prune
        self.full_fn = full_fn or (lambda mt, pt: prob.wb_peak(mt, pt))
        self.screen_fn = screen_fn or (lambda mt, pt, f: prob.wb_peak(mt, pt, f)[0])
        self.margin = screen_margin
        self.reset()

    def reset(self):
        self.cache = {}
        self.subset = set(self.prob.BASE_SUBSET)
        self._sub_t = None
        self.n_full = self.n_screen = self.n_hits = self.n_pruned = 0

    def _fidx(self):
        if self._sub_t is None:
            self._sub_t = torch.tensor(sorted(self.subset), device=self.prob.device)
        return self._sub_t

    def evaluate(self, models, ports, pbest_val):
        P = models.shape[0]
        cost = np.empty(P)
        keys = [None] * P
        todo = []
        for i in range(P):
            if self.use_cache:
                keys[i] = tuple(sorted(zip(ports[i].tolist(), models[i].tolist())))
                h = self.cache.get(keys[i])
                if h is not None:
                    cost[i] = h
                    self.n_hits += 1
                    continue
            todo.append(i)
        if not todo:
            return cost
        idx = np.array(todo)
        dev = self.prob.device
        mt = torch.from_numpy(models[idx]).to(dev)
        pt = torch.from_numpy(ports[idx]).to(dev)
        need = np.ones(len(idx), bool)
        if self.use_prune:
            lb = self.screen_fn(mt, pt, self._fidx()).double().cpu().numpy() * (1 - self.margin)
            self.n_screen += len(idx)
            need = lb < pbest_val[idx]
            cost[idx[~need]] = lb[~need]          # >= pbest: cannot update anything
            self.n_pruned += int((~need).sum())
        if need.any():
            sel = np.nonzero(need)[0]
            st = torch.from_numpy(sel).to(dev)
            pk, am = self.full_fn(mt.index_select(0, st), pt.index_select(0, st))
            pk = pk.double().cpu().numpy()
            self.n_full += len(sel)
            cost[idx[sel]] = pk
            if self.use_cache:
                for j, v in zip(idx[sel], pk):
                    self.cache[keys[j]] = float(v)
            if self.use_prune and am is not None:
                new = set(am.cpu().numpy().tolist()) - self.subset
                if new:
                    self.subset |= new
                    self._sub_t = None
        return cost

    def stats(self):
        return {"gpu_full_evals": self.n_full, "gpu_screen_evals": self.n_screen,
                "cache_hits": self.n_hits, "pruned": self.n_pruned}


def standard_evaluators(prob):
    """Factories for the GPUBench2 roster."""
    f = {m: (lambda m=m: SeqEvaluator(prob, m)) for m in ("numpy", "solve", "sm", "woodbury", "cpu_numpy")}
    f["woodbury_batch"] = lambda: SwarmEvaluator(prob, False, False)
    f["wb_cache"] = lambda: SwarmEvaluator(prob, True, False)
    f["wb_prune"] = lambda: SwarmEvaluator(prob, False, True)
    f["fast"] = lambda: SwarmEvaluator(prob, True, True)
    return f


# ============================================================
# PSO
# ============================================================

def seed_particles(cfg, n, prev_best):                        # FIX 11
    P, nw = cfg.N_PARTICLES, min(cfg.N_WARM_START, cfg.N_PARTICLES)
    if prev_best is None or n == 1 or nw == 0:
        return np.random.rand(P, 2 * n)
    pn = n - 1
    warm = np.hstack([np.tile(prev_best[:pn], (nw, 1)), np.random.rand(nw, 1),
                      np.tile(prev_best[pn:], (nw, 1)), np.random.rand(nw, 1)])
    if cfg.WARM_START_JITTER > 0 and nw > 1:
        carried = np.ones(2 * n, bool)
        carried[pn] = False
        carried[-1] = False
        noise = np.random.normal(0.0, cfg.WARM_START_JITTER, warm.shape)
        noise[0, :] = 0.0
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    if P - nw > 0:
        warm = np.vstack([warm, np.random.rand(P - nw, 2 * n)])
    return warm


def _logger(path):
    lg = logging.getLogger(path)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter('%(asctime)s INFO: %(message)s'))
    lg.addHandler(h)
    return lg


def run_pso(prob, ev, method, thresholds, run_id, log_dir=None, stage_hook=None):
    """
    PSO identical to GPUBench2 when len(thresholds)==1. With several
    thresholds the run stops at the LOWEST one and records the moment each
    higher one is first crossed. Because seeding depends only on
    (run, n_caps), the trajectory up to each crossing is exactly the one a
    separate run for that threshold would follow.

    stage_hook(prob, ev, n_caps, gbest, gbest_particle) -> (gbest,
    gbest_particle, extra_evals): optional end-of-stage improvement step
    (experiment 05).
    Returns one record per threshold.
    """
    cfg = prob.cfg
    thr_sorted = sorted(thresholds, reverse=True)
    final_thr = thr_sorted[-1]
    lg = _logger(os.path.join(log_dir, f"run_{run_id}.log")) if log_dir else None
    ev.reset()
    cross = {t: None for t in thr_sorted}       # t -> (time, n_caps, n_evals, gbest)
    stage_best = []
    best_cfg = None
    prob.sync()
    start = time.perf_counter()
    n_evals = 0
    prev_best = None

    def check_cross(g, n):
        for t in thr_sorted:
            if cross[t] is None and g <= t:
                prob.sync()
                cross[t] = (time.perf_counter() - start, n, n_evals, g)

    for n in range(1, cfg.MAX_CAPS + 1):
        np.random.seed(BASE_SEED + run_id * 10000 + n)                    # FIX 1
        particles = seed_particles(cfg, n, prev_best)
        vel = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_val = np.full(cfg.N_PARTICLES, np.inf)
        gbest, gpart = np.inf, None
        stop = False
        for it in range(cfg.N_ITERATIONS):
            if not ev.sequential:
                M, Pt = prob.decode_swarm(particles, n)
                costs = ev.evaluate(M, Pt, pbest_val)
            for i in range(cfg.N_PARTICLES):
                if ev.sequential:
                    m, p = prob.decode_config(particles[i], n)
                    cost = ev.eval_one(m, p)
                else:
                    cost = costs[i]
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]
                if cost < gbest:
                    gbest = cost
                    gpart = particles[i].copy()
                    check_cross(gbest, n)
                if gbest <= final_thr:
                    stop = True
                    break
            if stop:
                break
            w = cfg.W_MAX - (cfg.W_MAX - cfg.W_MIN) * (it / cfg.N_ITERATIONS)
            vel = w * vel + cfg.C1 * np.random.rand(*particles.shape) * (pbest - particles) \
                + cfg.C2 * np.random.rand(*particles.shape) * (gpart - particles)
            particles = np.clip(particles + vel, 0, 1)

        if gpart is None:                                                  # FIX 13
            prev_best = None
            continue
        if stage_hook is not None and gbest > final_thr:
            gbest, gpart, extra = stage_hook(prob, ev, n, gbest, gpart)
            n_evals += extra
            check_cross(gbest, n)
        prev_best = gpart
        bm, bp = prob.decode_config(gpart, n)
        best_cfg = (bm.tolist(), bp.tolist())
        stage_best.append(gbest)
        if lg:
            lg.info("n_caps=%d minZ=%.6f placement=%s", n, gbest,
                    {int(p): int(c) for c, p in zip(bm, bp)})
        if gbest <= final_thr:
            break

    prob.sync()
    total = time.perf_counter() - start
    recs = []
    for t in thr_sorted:
        c = cross[t]
        r = {"threshold": t, "method": method, "run_id": run_id,
             "success": c is not None,
             "time_to_target_s": c[0] if c else None,
             "total_wall_time_s": c[0] if c else total,
             "caps_at_target": c[1] if c else None,
             "n_evals": c[2] if c else n_evals,
             "best_minz_ohm": c[3] if c else (min(stage_best) if stage_best else float("inf")),
             "init_time_s": prob.init_time,
             "shared_run": len(thr_sorted) > 1,
             "best_config": best_cfg}
        r.update(ev.stats())
        recs.append(r)
        if lg:
            lg.info("RESULT %s", json.dumps({k: v for k, v in r.items() if k != "best_config"}))
    return recs


# ============================================================
# VERIFY / WARM-UP / BENCHMARK
# ============================================================

def random_configs(prob, n_trials=6, seed=999, kmax=6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        n = int(rng.integers(1, min(kmax, len(prob.VALID_PORTS)) + 1))
        m, p = prob.decode_config(rng.random(2 * n), n)
        out.append((m, p))
    return out


def eval_config_with(ev, m, p):
    if ev.sequential:
        return ev.eval_one(m, p)
    ev.reset()
    return ev.evaluate(np.array([m, m]), np.array([p, p]), np.full(2, np.inf))[0]


def verify(prob, factories, out, tol=1e-6, strict=True):
    cfgs = random_configs(prob)
    refs = [prob.cpu_ref(list(zip(m.tolist(), p.tolist()))) for m, p in cfgs]
    rep = {}
    for name, fac in factories.items():
        if name == "cpu_numpy":
            continue
        ev = fac()
        errs = [abs(eval_config_with(ev, m, p) - r) / max(abs(r), 1e-30) for (m, p), r in zip(cfgs, refs)]
        rep[name] = float(max(errs))
        print(f"[verify] {name:<20} max_rel={rep[name]:.3e}")
        if strict and prob.cdtype == torch.complex128 and rep[name] > tol:
            raise AssertionError(f"{name} disagrees with CPU reference ({rep[name]:.2e})")
    with open(os.path.join(out, "verification.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


def warmup(prob, factories):
    m, p = random_configs(prob, 1)[0]
    P = prob.cfg.N_PARTICLES
    for _ in range(2):
        for name, fac in factories.items():
            if name == "cpu_numpy":
                continue
            ev = fac()
            if ev.sequential:
                ev.eval_one(m, p)
            else:
                ev.reset()
                ev.evaluate(np.tile(m, (P, 1)), np.tile(p, (P, 1)), np.zeros(P))
                ev.reset()
                ev.evaluate(np.tile(m, (P, 1)), np.tile(p, (P, 1)), np.full(P, np.inf))
    prob.sync()


def run_benchmark(prob, factories, methods, out, shared=False, stage_hooks=None, tag=""):
    """Runs thresholds x methods x runs. shared=True: one run per (method,
    run) at the lowest threshold, recording all crossings (experiment 01)."""
    cfg = prob.cfg
    stage_hooks = stage_hooks or {}
    jl = os.path.join(out, f"results_long{tag}.jsonl")
    open(jl, "w").close()
    recs = []

    def emit(rs):
        with open(jl, "a") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        recs.extend(rs)
        for r in rs:
            print(f"  {r['method']:<16} T={r['threshold']:<6} run {r['run_id']:2d}: "
                  f"{r['total_wall_time_s']:.3f}s caps={r['caps_at_target']} "
                  f"minZ={r['best_minz_ohm']:.5f} evals={r['n_evals']}", flush=True)

    if shared:
        for m in methods:
            print(f"\nMETHOD {m} (shared run, targets {cfg.TARGETS})", flush=True)
            ev = factories[m]()
            for run in range(1, cfg.NUM_RUNS + 1):
                d = os.path.join(out, f"{m}{FOLDER_SEP}shared", f"run_{run}")
                os.makedirs(d, exist_ok=True)
                emit(run_pso(prob, ev, m, cfg.TARGETS, run, d, stage_hooks.get(m)))
    else:
        for t in cfg.TARGETS:
            for m in methods:
                print(f"\nMETHOD {m} TARGET {t}", flush=True)
                ev = factories[m]()
                for run in range(1, cfg.NUM_RUNS + 1):
                    d = os.path.join(out, f"{m}{FOLDER_SEP}{t}", f"run_{run}")
                    os.makedirs(d, exist_ok=True)
                    emit(run_pso(prob, ev, m, [t], run, d, stage_hooks.get(m)))
    return recs


# ============================================================
# ANALYSIS
# ============================================================

def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


def analyse(records, out, baseline, tag=""):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(records).drop(columns=["best_config"], errors="ignore")
    df.to_csv(os.path.join(out, f"results_long{tag}.csv"), index=False)
    if baseline in set(df.method):
        b = df[df.method == baseline].set_index(["threshold", "run_id"])
        df["speedup"] = [b.loc[(r.threshold, r.run_id), "total_wall_time_s"] / r.total_wall_time_s
                         if (r.threshold, r.run_id) in b.index else np.nan for r in df.itertuples()]
        df["traj_match"] = [
            (abs(b.loc[(r.threshold, r.run_id), "best_minz_ohm"] - r.best_minz_ohm)
             <= 1e-9 * max(1.0, abs(r.best_minz_ohm)))
            and (b.loc[(r.threshold, r.run_id), "caps_at_target"] == r.caps_at_target
                 or (pd.isna(b.loc[(r.threshold, r.run_id), "caps_at_target"]) and pd.isna(r.caps_at_target)))
            if (r.threshold, r.run_id) in b.index else np.nan for r in df.itertuples()]
    else:
        df["speedup"] = np.nan
        df["traj_match"] = np.nan
    rows = []
    for (t, m), g in df.groupby(["threshold", "method"]):
        row = {"threshold": t, "method": m,
               "time_s_mean": g.total_wall_time_s.mean(), "time_s_ci95": ci95(g.total_wall_time_s),
               "speedup_mean": g.speedup.mean(), "speedup_ci95": ci95(g.speedup.dropna()),
               "caps_mean": g.caps_at_target.dropna().mean() if g.caps_at_target.notna().any() else np.nan,
               "best_minz_mean": g.best_minz_ohm.mean(), "success_rate": g.success.mean(),
               "n_evals_mean": g.n_evals.mean(),
               "trajectory_match_rate": g.traj_match.astype(float).mean(), "n_runs": len(g)}
        for c in ("gpu_full_evals", "cache_hits", "pruned"):
            if c in g:
                row[c + "_mean"] = g[c].mean()
        rows.append(row)
    s = pd.DataFrame(rows).sort_values(["threshold", "method"])
    s.to_csv(os.path.join(out, f"summary_table{tag}.csv"), index=False)
    with open(os.path.join(out, f"statistics{tag}.txt"), "w") as f:
        f.write(s.to_string(index=False))
    with open(os.path.join(out, f"timeouts{tag}.txt"), "w") as f:
        for (t, m), g in df[~df.success.astype(bool)].groupby(["threshold", "method"]):
            f.write(f"{m} T={t}: runs {g.run_id.tolist()} best_minZ "
                    f"{np.round(g.best_minz_ohm.values, 5).tolist()}\n")
    figs = os.path.join(out, "figs")
    os.makedirs(figs, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for m, g in s.groupby("method"):
        ax[0].errorbar(g.threshold, g.time_s_mean, yerr=g.time_s_ci95, marker="o", capsize=3, label=m)
        ax[1].errorbar(g.threshold, g.speedup_mean, yerr=g.speedup_ci95, marker="o", capsize=3, label=m)
    ax[0].set_ylabel("Wall time per run (s)")
    ax[1].set_ylabel(f"Speedup vs {baseline}")
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel("Target |Z11| (Ω)")
        a.grid(True, which="both", alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, f"runtime_and_speedup{tag}.png"), dpi=200)
    plt.close(fig)
    print("\n" + s[["threshold", "method", "time_s_mean", "speedup_mean",
                    "trajectory_match_rate", "success_rate"]].to_string(index=False))
    return df, s


def select_methods(args, factories):
    ms = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in ms:
        if m not in factories:
            raise ValueError(f"unknown method {m!r}; available: {sorted(factories)}")
    return ms
