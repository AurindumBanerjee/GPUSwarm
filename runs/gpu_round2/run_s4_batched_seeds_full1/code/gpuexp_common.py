"""
gpuexp_common.py -- shared building blocks of the GPUExperiments (copied next to every experiment script by run_all.sh).

Contents
    SETTINGS     fixed experimental settings, read from the environment (QUICK=1 -> tiny schedule); recorded in every manifest
    Problem      PDN + capacitor library on the device (Z0 = Y^-1, 1/D), exact batched Woodbury peak |Z11|, decoding
    Evaluators   SeqNumpyEvaluator (full inverse per particle: GPUBench2 'numpy' method), SwarmEvaluator (batched Woodbury,
                 optional cache + exact lower-bound pruning: GPUBench2 'woodbury_batch' / 'fast'; peak function is injectable)
    run_pso      the GPUBench2 PSO (paired per-stage seeding, floor(frac*K) decode, immediate stop), several thresholds at once
    Run          run-folder bookkeeping in the repo output style (run.log, results_long.jsonl, summary.*, statistics.txt, manifest)

The PSO / decode / cost code is a copy of GPUBench2.py (which cannot be imported: it parses argv at import).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import hashlib
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CD = torch.complex128


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


QUICK = _env("QUICK", "0") == "1"
SETTINGS = {
    "QUICK": QUICK,
    "NUM_RUNS": _env("NUM_RUNS", 5 if QUICK else 20, int),
    "TARGETS": [float(t) for t in _env("TARGETS", "0.05,0.045,0.04").split(",")],
    "N_PARTICLES": _env("N_PARTICLES", 20 if QUICK else 50, int),
    "N_ITERATIONS": _env("N_ITERATIONS", 10 if QUICK else 40, int),
    "MAX_CAPS": _env("MAX_CAPS", 8 if QUICK else 20, int),
    "TARGET_PORT": 0,
    "W_MAX": 0.9, "W_MIN": 0.4, "C1": 1.5, "C2": 1.5,
    "WARM_START_FRACTION": _env("WARM_START_FRACTION", 1.0, float),   # identical in every arm; default = CPU reference run08
    "WARM_START_JITTER": _env("WARM_START_JITTER", 0.0, float),       # (GPUBench2 hard-codes 0.4 / 0.02)
    "BASE_SEED": _env("BASE_SEED", 12345, int),
    "DTYPE": "complex128",
    "STRIDE": _env("STRIDE", 16, int),
    "MAX_BATCH_ELEMS": _env("MAX_BATCH_ELEMS", 3e7, float),
    "Y_PATH": _env("Y_PATH", "/DATA/Aurindum/GPUSwarm/Data/Reference/y2.mat"),
    "D_PATH": _env("D_PATH", "/DATA/Aurindum/GPUSwarm/Data/Reference/decaps.mat"),
    "FREQ_PATH": _env("FREQ_PATH", "/DATA/Aurindum/GPUSwarm/Data/Reference/freq2.mat"),
    "BASELINE_JSONL": _env("BASELINE_JSONL", ""),
}
S = SETTINGS
TARGET_PORT = S["TARGET_PORT"]
BASELINE_LABEL = "GPU numpy (GPUBench2 'numpy' method: full inverse per particle, same GPU)"


def cuda_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------------ problem
class Problem:
    """PDN (impedance form) + capacitor library on the device. Build from files (Y -> Z0) or from arrays (macromodels)."""

    def __init__(self, z0=None, y=None, d=None, freq=None, target_port=TARGET_PORT, label=""):
        assert d is not None
        self.label, self.target_port = label, target_port
        self.d_np = np.ascontiguousarray(d, dtype=np.complex128)
        self.freq = None if freq is None else np.asarray(freq, float).ravel()
        if z0 is None:
            self.y_np = np.ascontiguousarray(y, dtype=np.complex128)
            self.Y = torch.from_numpy(self.y_np).to(DEVICE, CD)
            self.Z0 = torch.linalg.inv(self.Y)
        else:
            self.y_np = None
            self.Y = None
            self.Z0 = torch.as_tensor(z0).to(DEVICE, CD).contiguous()
        self.F, self.n, _ = self.Z0.shape
        self.C = self.d_np.shape[0]
        assert self.d_np.shape[1] == self.F, (self.d_np.shape, self.F)
        self.D = torch.from_numpy(self.d_np).to(DEVICE, CD)
        Ds = torch.where(self.D.abs() < 1e-300, torch.full_like(self.D, 1e-300), self.D)
        self.DINV = 1.0 / Ds
        self.valid_ports = np.array([p for p in range(self.n) if p != target_port], dtype=int)
        self.E1 = None
        cuda_sync()
        # initial pruning subset: stride grid + band edges + local maxima of bare |Z11|
        bare = self.Z0[:, target_port, target_port].abs().cpu().numpy()
        lmax = [i for i in range(1, self.F - 1) if bare[i] >= bare[i - 1] and bare[i] >= bare[i + 1]]
        lmax = sorted(lmax, key=lambda i: -bare[i])[:16]
        self.base_subset = sorted(set(range(0, self.F, max(1, S["STRIDE"]))) | {0, self.F - 1} | set(lmax))

    @classmethod
    def from_files(cls, y_path, d_path, freq_path=None, fmax=None, fmin=None, label=""):
        y = np.transpose(sio.loadmat(y_path)["y"], (2, 0, 1))
        d = sio.loadmat(d_path)["decaps"]
        freq = sio.loadmat(freq_path)["freq"].ravel() if freq_path else None
        assert y.shape[0] == d.shape[1], f"PDN has {y.shape[0]} freqs, library {d.shape[1]}"
        if fmax is not None or fmin is not None:
            keep = np.ones(len(freq), bool)
            if fmax is not None:
                keep &= freq <= fmax
            if fmin is not None:
                keep &= freq >= fmin
            y, d, freq = y[keep], d[:, keep], freq[keep]
        return cls(y=y, d=d, freq=freq, label=label)

    # ---- decoding (GPUBench2 FIX 2, 3, 12)
    def frac_to_index(self, frac, K):
        return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)

    def decode_particle(self, vec, n_caps):
        return (self.frac_to_index(vec[:n_caps], self.C),
                self.valid_ports[self.frac_to_index(vec[n_caps:], len(self.valid_ports))])

    def resolve_adjacent_ports(self, models, ports):
        used = {self.target_port}
        for i in range(len(ports)):
            if ports[i] not in used:
                used.add(ports[i])
                continue
            placed = False
            for off in range(1, self.n):
                for cand in (ports[i] + off, ports[i] - off):
                    if 0 <= cand < self.n and cand not in used:
                        ports[i] = cand
                        used.add(cand)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                used.add(ports[i])
        return models, ports

    def decode_swarm(self, particles, n_caps):
        P = particles.shape[0]
        M = np.empty((P, n_caps), dtype=np.int64)
        Pt = np.empty((P, n_caps), dtype=np.int64)
        for i in range(P):
            m, p = self.decode_particle(particles[i], n_caps)
            m, p = self.resolve_adjacent_ports(m, p)
            M[i], Pt[i] = m, p
        return M, Pt

    def gather(self, models_t, ports_t, fidx=None):
        """Pieces of the Woodbury system for B configurations: M = diag(1/d)+Z0[P,P] (b,F',k,k), zc (b,F',k,1), zr (b,F',1,k), z00 (F',)."""
        Z0f = self.Z0 if fidx is None else self.Z0.index_select(0, fidx)
        Dv = self.DINV if fidx is None else self.DINV.index_select(1, fidx)
        t = self.target_port
        M = Z0f[:, ports_t.unsqueeze(2), ports_t.unsqueeze(1)].permute(1, 0, 2, 3)
        M = M + torch.diag_embed(Dv[models_t].transpose(1, 2))
        zc = Z0f[:, ports_t, t].permute(1, 0, 2).unsqueeze(-1)
        zr = Z0f[:, t, ports_t].permute(1, 0, 2).unsqueeze(-2)
        return M, zc, zr, Z0f[:, t, t]

    def chunk(self, F, k):
        return max(1, int(S["MAX_BATCH_ELEMS"] // max(1, F * k * k)))

    def wb_values(self, models_t, ports_t, fidx=None):
        """Exact Z11(f) of B configurations, (B,F') complex, plus a bad flag (B,). Chunked."""
        B, k = ports_t.shape
        Fp = self.F if fidx is None else len(fidx)
        out, bads = [], []
        for s in range(0, B, self.chunk(Fp, k)):
            M, zc, zr, z00 = self.gather(models_t[s:s + self.chunk(Fp, k)], ports_t[s:s + self.chunk(Fp, k)], fidx)
            x, info = torch.linalg.solve_ex(M, zc)
            out.append(z00.unsqueeze(0) - (zr @ x)[..., 0, 0])
            bads.append((info != 0).any(dim=1))
        return torch.cat(out), torch.cat(bads)

    def wb_peak(self, models_t, ports_t, fidx=None):
        """Batched Woodbury fitness (GPUBench2 wb_peak): (peak (B,), argmax global freq (B,)). Singular systems score 1e200."""
        B, k = ports_t.shape
        Fp = self.F if fidx is None else len(fidx)
        peaks, argm = [], []
        ch = self.chunk(Fp, k)
        for s in range(0, B, ch):
            M, zc, zr, z00 = self.gather(models_t[s:s + ch], ports_t[s:s + ch], fidx)
            x, info = torch.linalg.solve_ex(M, zc)
            val = z00.unsqueeze(0) - (zr @ x)[..., 0, 0]
            pk, am = val.abs().max(dim=1)
            bad = (info != 0).any(dim=1) | ~torch.isfinite(pk)
            peaks.append(torch.where(bad, torch.full_like(pk, 1e200), pk))
            argm.append(am)
        peak, am = torch.cat(peaks), torch.cat(argm)
        if fidx is not None:
            am = fidx[am]
        return peak, am

    def exact_peak_np(self, config):
        """Reference: full inverse of the assembled Y at every frequency, on the CPU (config = [(model, port), ...])."""
        assert self.y_np is not None
        peak = 0.0
        for f in range(self.F):
            A = self.y_np[f].copy()
            for cap, port in config:
                A[port, port] += self.d_np[cap, f]
            try:
                v = np.linalg.inv(A)[self.target_port, self.target_port]
            except np.linalg.LinAlgError:
                return 1e200
            peak = max(peak, abs(v))
        return peak

    def k_numpy(self, config):
        A = self.Y.clone()
        for cap, port in config:
            A[:, port, port] += self.D[cap, :]
        return torch.linalg.inv(A)[:, self.target_port, self.target_port]


# ------------------------------------------------------------------------------------------------------ evaluators
class SeqNumpyEvaluator:
    """GPUBench2 'numpy' method: one particle at a time, full inverse of the assembled Y at every frequency (complex128)."""
    sequential = True
    name = "numpy"

    def __init__(self, prob):
        self.prob = prob
        self.reset()

    def reset(self):
        self.n_full = self.n_screen = self.n_hits = self.n_pruned = 0

    def begin_stage(self, n_caps):
        pass

    def cost_one(self, models, ports):
        try:
            v = self.prob.k_numpy(list(zip(models.tolist(), ports.tolist())))
            self.n_full += 1
            return float(v.abs().max().item())
        except torch.linalg.LinAlgError:
            return 1e200

    def stats(self):
        return {"gpu_full_evals": self.n_full, "gpu_screen_evals": 0, "cache_hits": 0, "pruned": 0, "subset_size": 0}


class SwarmEvaluator:
    """Batched Woodbury evaluation of a whole swarm with optional cache + exact lower-bound pruning (GPUBench2 SwarmEvaluator).
    peak_fn(models_t, ports_t, fidx) -> (peak, argmax); default = exact complex128 Woodbury."""
    sequential = False

    def __init__(self, prob, use_cache, use_prune, peak_fn=None, name=None):
        self.prob, self.use_cache, self.use_prune = prob, use_cache, use_prune
        self.peak_fn = peak_fn or prob.wb_peak
        self.name = name or ("fast" if use_cache and use_prune else "woodbury_batch")
        self.reset()

    def reset(self):
        self.cache = {}
        self.subset = set(self.prob.base_subset)
        self._subset_t = None
        self.n_full = self.n_screen = self.n_hits = self.n_pruned = 0

    def begin_stage(self, n_caps):
        pass

    def _fidx(self):
        if self._subset_t is None:
            self._subset_t = torch.tensor(sorted(self.subset), device=DEVICE)
        return self._subset_t

    def evaluate(self, models, ports, pbest_val):
        P = models.shape[0]
        cost = np.empty(P)
        todo, keys = [], [None] * P
        for i in range(P):
            if self.use_cache:
                keys[i] = tuple(sorted(zip(ports[i].tolist(), models[i].tolist())))
                hit = self.cache.get(keys[i])
                if hit is not None:
                    cost[i] = hit
                    self.n_hits += 1
                    continue
            todo.append(i)
        if not todo:
            return cost
        idx = np.array(todo)
        mt = torch.from_numpy(models[idx]).to(DEVICE)
        pt = torch.from_numpy(ports[idx]).to(DEVICE)
        need_full = np.ones(len(idx), bool)
        if self.use_prune:
            lb, _ = self.peak_fn(mt, pt, self._fidx())
            lb = lb.cpu().numpy()
            self.n_screen += len(idx)
            need_full = lb < pbest_val[idx] * (1 + 1e-9)
            cost[idx[~need_full]] = lb[~need_full]
            self.n_pruned += int((~need_full).sum())
        if need_full.any():
            sel = np.nonzero(need_full)[0]
            sel_t = torch.from_numpy(sel).to(DEVICE)
            pk, am = self.peak_fn(mt.index_select(0, sel_t), pt.index_select(0, sel_t))
            pk = pk.cpu().numpy()
            self.n_full += len(sel)
            cost[idx[sel]] = pk
            if self.use_cache:
                for j, v in zip(idx[sel], pk):
                    self.cache[keys[j]] = float(v)
            if self.use_prune:
                new = set(am.cpu().numpy().tolist()) - self.subset
                if new:
                    self.subset |= new
                    self._subset_t = None
        return cost

    def stats(self):
        return {"gpu_full_evals": self.n_full, "gpu_screen_evals": self.n_screen, "cache_hits": self.n_hits,
                "pruned": self.n_pruned, "subset_size": len(self.subset)}


# ------------------------------------------------------------------------------------------------------ PSO
def seed_particles(n_caps, prev_best, rng_rand=None):
    """GPUBench2 seed_particles (FIX 11) with explicit warm-start fraction / jitter. Uses the global numpy stream like the original."""
    NP = S["N_PARTICLES"]
    n_warm = min(int(round(S["WARM_START_FRACTION"] * NP)), NP)
    jit = S["WARM_START_JITTER"]
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(NP, 2 * n_caps)
    pn = n_caps - 1
    warm = np.hstack([np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
                      np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1)])
    if jit > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, bool)
        carried[pn] = False
        carried[-1] = False
        noise = np.random.normal(0.0, jit, warm.shape)
        noise[0, :] = 0.0
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    if NP - n_warm > 0:
        warm = np.vstack([warm, np.random.rand(NP - n_warm, 2 * n_caps)])
    return warm


def run_pso(prob, ev, thresholds, run_id, label=None, on_stage_end=None, particles_hook=None):
    """One PSO run (GPUBench2 run_pso). thresholds: list; the run stops at min(thresholds). For every threshold the first
    crossing (stage, iteration, logical evals, time, cost) is recorded -> prefix property used by experiment 1.

    on_stage_end(n_caps, gbest, gbest_particle, prev_best) -> (gbest, gbest_particle) | None may refine the stage best (exp 5).
    Returns the result record."""
    thresholds = sorted(thresholds, reverse=True)
    stop_thr = thresholds[-1]
    NP, NI, MAXC = S["N_PARTICLES"], S["N_ITERATIONS"], S["MAX_CAPS"]
    ev.reset()
    stage_best, crossings = [], {}
    cuda_sync()
    start = time.perf_counter()
    time_to_target = caps_at_target = None
    n_evals = 0
    prev_best = None
    best_cfg = None
    final_n_caps = None
    extra = {}
    for n_caps in range(1, MAXC + 1):
        np.random.seed(S["BASE_SEED"] + run_id * 10000 + n_caps)
        ev.begin_stage(n_caps)
        particles = seed_particles(n_caps, prev_best)
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_val = np.full(NP, np.inf)
        gbest, gbest_particle = np.inf, None
        stop = False
        it = 0
        for it in range(NI):
            if not ev.sequential:
                M, Pp = prob.decode_swarm(particles, n_caps)
                costs = ev.evaluate(M, Pp, pbest_val)
            for i in range(NP):
                if ev.sequential:
                    m, p = prob.decode_particle(particles[i], n_caps)
                    m, p = prob.resolve_adjacent_ports(m, p)
                    cost = ev.cost_one(m, p)
                else:
                    cost = costs[i]
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]
                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()
                for thr in thresholds:
                    if thr not in crossings and gbest <= thr:
                        cuda_sync()
                        crossings[thr] = {"stage_caps": n_caps, "iteration": it + 1, "n_evals": n_evals,
                                          "time_s": time.perf_counter() - start, "cost": float(gbest)}
                if gbest <= stop_thr:
                    stop = True
                    break
            if stop:
                cuda_sync()
                time_to_target = time.perf_counter() - start
                caps_at_target = n_caps
                break
            w = S["W_MAX"] - (S["W_MAX"] - S["W_MIN"]) * (it / NI)
            velocities = w * velocities \
                + S["C1"] * np.random.rand(*particles.shape) * (pbest - particles) \
                + S["C2"] * np.random.rand(*particles.shape) * (gbest_particle - particles)
            particles = np.clip(particles + velocities, 0, 1)
        if gbest_particle is None:
            prev_best = None
            continue
        if on_stage_end is not None and not stop:
            r = on_stage_end(n_caps, gbest, gbest_particle, prev_best)
            if r is not None:
                gbest, gbest_particle = r[0], r[1]
                if gbest <= stop_thr:
                    stop = True
                    cuda_sync()
                    time_to_target = time.perf_counter() - start
                    caps_at_target = n_caps
        prev_best = gbest_particle
        bm, bp = prob.decode_particle(gbest_particle, n_caps)
        bm, bp = prob.resolve_adjacent_ports(bm, bp)
        best_cfg = (bm.tolist(), bp.tolist())
        stage_best.append([n_caps, float(gbest)])
        final_n_caps = n_caps
        if stop or gbest <= stop_thr:
            break
    cuda_sync()
    total = time.perf_counter() - start
    best = min(b for _, b in stage_best) if stage_best else float("inf")
    rec = {"method": label or ev.name, "run": run_id, "seed": S["BASE_SEED"] + run_id * 10000,
           "threshold": stop_thr, "success": time_to_target is not None,
           "time_s": time_to_target if time_to_target is not None else -1.0, "total_time_s": total,
           "caps_at_target": caps_at_target, "final_n_caps": final_n_caps, "best_minz": best, "n_evals": n_evals,
           "stage_best": stage_best, "best_models": best_cfg[0] if best_cfg else None,
           "best_ports": best_cfg[1] if best_cfg else None,
           "crossings": {str(k): v for k, v in crossings.items()}}
    rec.update(ev.stats())
    rec.update(extra)
    return rec


def match(rec, base, rtol=1e-9):
    """Trajectory match with a baseline record: equal (best_minZ, caps_at_target) -- plus evals/final stage as a stricter flag."""
    if base is None:
        return None, None
    ok_val = abs(rec["best_minz"] - base["best_minz"]) <= rtol * max(1.0, abs(rec["best_minz"]))
    ok = bool(ok_val and rec["caps_at_target"] == base["caps_at_target"])
    strict = bool(ok and rec["n_evals"] == base["n_evals"] and rec["final_n_caps"] == base["final_n_caps"])
    return ok, strict


def load_baseline(path=None, method="numpy"):
    """{(threshold, run): record} of the baseline arm (numpy method) from a results_long.jsonl, or {}."""
    path = path or S["BASELINE_JSONL"]
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("method") == method:
                out[(round(r["threshold"], 6), r["run"])] = r
    return out


# ------------------------------------------------------------------------------------------------------ run bookkeeping
def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def md5_file(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


class Run:
    def __init__(self, experiment, script_file, extra_settings=None):
        self.experiment = experiment
        self.run_dir = os.path.abspath(os.environ.get("RUN_DIR", os.path.join(HERE, "run_local_" + experiment)))
        self.run_id = os.environ.get("RUN_ID", "run_local")
        for d in ("code", "output", "logs", "summary", "plots"):
            os.makedirs(os.path.join(self.run_dir, d), exist_ok=True)
        self.script = os.path.abspath(script_file)
        self.extra = extra_settings or {}
        self.start = time.time()
        self.records = []
        lg = logging.getLogger(experiment + self.run_id)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        h = logging.FileHandler(os.path.join(self.run_dir, "logs", "run.log"))
        h.setFormatter(logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
        lg.addHandler(logging.StreamHandler(sys.stdout))
        self.lg = lg
        self.jsonl = os.path.join(self.run_dir, "output", "results_long.jsonl")
        open(self.jsonl, "w").close()
        self.manifest("running")
        kv = " | ".join(f"{k}={v}" for k, v in {**S, **self.extra}.items())
        self.lg.info("RUN_START | experiment=%s | run_id=%s | %s", experiment, self.run_id, kv)

    def manifest(self, status, end=None, notes=None):
        gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
        try:
            git = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:                                         # noqa: BLE001
            git = None
        common = os.path.join(os.path.dirname(self.script), "gpuexp_common.py")
        m = {"experiment": self.experiment, "run_id": self.run_id, "status": status, "start": iso(self.start),
             "end": iso(end) if end else None, "host": platform.node(), "platform": platform.platform(),
             "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
             "cuda": torch.version.cuda, "gpu": gpu, "cpu_count": os.cpu_count(), "timer": "time.perf_counter + cuda synchronize",
             "code": {"file": os.path.relpath(self.script, self.run_dir).replace(os.sep, "/"), "md5": md5_file(self.script),
                      "common_md5": md5_file(common) if os.path.exists(common) else None, "git": git},
             "settings": {**S, **self.extra},
             "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
             "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
        if notes:
            m["notes"] = notes
        json.dump(m, open(os.path.join(self.run_dir, "manifest.json"), "w"), indent=2, default=str)

    def add(self, rec):
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        self.records.append(rec)

    def result_line(self, rec, **kv):
        self.lg.info("RESULT | " + " | ".join(f"{k}={v}" for k, v in {
            "method": rec["method"], "threshold": rec["threshold"], "run": rec["run"], "time_s": f"{rec['time_s']:.4f}",
            "total_time_s": f"{rec['total_time_s']:.4f}", "caps": rec["caps_at_target"], "best_minz": f"{rec['best_minz']:.6f}",
            "n_evals": rec["n_evals"], "success": rec["success"], **kv}.items()))

    def fail(self, method, thr, run_id, err):
        self.lg.error("RESULT | method=%s | threshold=%s | run=%d | time_s=-1 | best_minz=nan | success=False | error=%r",
                      method, thr, run_id, err)
        rec = {"method": method, "threshold": thr, "run": run_id, "success": False, "time_s": -1.0, "total_time_s": -1.0,
               "caps_at_target": None, "final_n_caps": None, "best_minz": float("nan"), "n_evals": 0, "error": repr(err)}
        self.add(rec)

    def summarise(self, group=("method", "threshold"), extra_cols=()):
        """summary.txt / summary.csv / statistics.txt: one row per (method, threshold)."""
        import pandas as pd
        df = pd.DataFrame(self.records)
        rows = []
        for key, g in df.groupby(list(group), sort=True):
            key = key if isinstance(key, tuple) else (key,)
            ok = g[g.success == True]                              # noqa: E712
            tt = g.total_time_s[g.total_time_s >= 0]
            row = dict(zip(group, key))
            row.update({"runs": len(g), "successes": int(g.success.sum()), "success_rate": float(g.success.mean()),
                        "time_to_target_mean_s": float(ok.time_s.mean()) if len(ok) else None,
                        "total_time_mean_s": float(tt.mean()) if len(tt) else None, "total_time_ci95_s": ci95(tt) if len(tt) else None,
                        "caps_mean": float(ok.caps_at_target.mean()) if len(ok) else None,
                        "best_minz_mean": float(g.best_minz.mean()), "n_evals_mean": float(g.n_evals.mean())})
            for c in extra_cols:
                if c in g:
                    v = pd.to_numeric(g[c], errors="coerce")
                    row[c] = float(v.mean()) if v.notna().any() else None
            rows.append(row)
        with open(os.path.join(self.run_dir, "summary", "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(self.run_dir, "summary", "summary.txt"), "w") as fh:
            for r in rows:
                fh.write(" | ".join(f"{k}={('-' if v is None else format(v, '.6g') if isinstance(v, float) else v)}" for k, v in r.items()) + "\n")
        with open(os.path.join(self.run_dir, "summary", "statistics.txt"), "w") as fh:
            fh.write(f"records = {len(df)}\nsuccesses = {int(df.success.sum())}\nfailures_or_timeouts = {int((~df.success).sum())}\n")
            fh.write(f"best_overall = {df.best_minz.min():.6g}\n")
        return rows

    def finish(self, notes=None):
        self.manifest("finished", time.time(), notes)
        readme = os.path.join(self.run_dir, "README.txt")
        if not os.path.exists(readme):
            open(readme, "w").write(f"RUN {self.experiment}/{self.run_id}\n")
        self.lg.info("DONE | outputs under %s", self.run_dir)


def concurrent_jobs_note():
    """Snapshot of the GPU and load at the start of a run (goes to the log; copy into the README if non-empty)."""
    try:
        g = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                    stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                             # noqa: BLE001
        g = "n/a"
    return f"gpu_compute_apps=[{g or 'none'}] loadavg={os.getloadavg() if hasattr(os, 'getloadavg') else 'n/a'}"


def init_device_log(run):
    torch.zeros(1, device=DEVICE)
    cuda_sync()
    run.lg.info("ENV | device=%s | torch=%s | cuda=%s | %s", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
                torch.__version__, torch.version.cuda, concurrent_jobs_note())


def warmup(prob, evs):
    """Touch every kernel once so first-call costs (cuSOLVER/cuBLAS init) are outside the timed runs."""
    cfg = [(0, int(prob.valid_ports[i % len(prob.valid_ports)])) for i in range(min(2, len(prob.valid_ports)))]
    for _ in range(3):
        for ev in evs:
            if ev.sequential:
                ev.cost_one(np.array([c for c, _ in cfg]), np.array([p for _, p in cfg]))
            else:
                n = S["N_PARTICLES"]
                M = np.array([[c for c, _ in cfg]] * n)
                Pp = np.array([[p for _, p in cfg]] * n)
                ev.evaluate(M, Pp, np.full(n, np.inf))
                ev.reset()
    cuda_sync()


def verify_against_cpu(prob, evs, n_trials=6):
    """Max relative deviation of every evaluator vs the CPU full-inverse reference on random configurations."""
    rng = np.random.default_rng(999)
    cfgs = []
    for _ in range(n_trials):
        n = int(rng.integers(1, min(6, len(prob.valid_ports)) + 1))
        m, p = prob.decode_particle(rng.random(2 * n), n)
        m, p = prob.resolve_adjacent_ports(m, p)
        cfgs.append(list(zip(m.tolist(), p.tolist())))
    refs = [prob.exact_peak_np(c) for c in cfgs]
    rep = {}
    for ev in evs:
        errs = []
        for c, r in zip(cfgs, refs):
            if ev.sequential:
                got = ev.cost_one(np.array([x for x, _ in c]), np.array([q for _, q in c]))
            else:
                ev.reset()
                got = ev.evaluate(np.array([[x for x, _ in c]] * 2), np.array([[q for _, q in c]] * 2), np.full(2, np.inf))[0]
            errs.append(abs(got - r) / max(abs(r), 1e-30))
        rep[ev.name] = float(max(errs))
        ev.reset()
    return rep
