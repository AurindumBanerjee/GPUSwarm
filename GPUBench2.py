# nohup python GPUBench2.py > GPUBench2.log 2>&1 &
# ps -u $USER | grep GPUBench2
# pkill -9 -f GPUBench2.py
#
# ============================================================
# GPUBench2 -- GPUBench.py + the speed-up reformulations
# ============================================================
# Keeps every correctness fix of GPUBench.py (FIX 1-13: paired per-stage
# seeding, floor(frac*K) discretisation, TARGET_PORT exclusion, hybrid warm
# start, full-offset port resolution, narrow exception handling, timeout
# reporting) and the same PSO, objective (peak |Z11| over frequency) and
# immediate-stop rule. What is new:
#
# NEW 1  "woodbury": exact low-rank fitness. Capacitors only add d_k(f) to
#        k diagonal entries, so with Z0 = Y^-1 precomputed
#            Z11 = Z0[t,t] - Z0[t,P] (diag(1/d_P) + Z0[P,P])^-1 Z0[P,t]
#        i.e. one k x k solve per frequency instead of an n x n one, and no
#        Y_eq assembly. Exact (Woodbury identity), valid even with repeated
#        ports. This is the closed-form multi-rank version of "sm".
# NEW 2  "woodbury_batch": the whole swarm (P particles x F freqs) is
#        evaluated in one batched solve per PSO iteration with ONE
#        device->host copy, instead of P launches + P syncs. The
#        immediate-stop rule is preserved exactly: costs are scanned in
#        particle order and the scan stops at the first particle that meets
#        the target, so pbest/gbest/RNG trajectories are identical to the
#        sequential methods. (Some trailing evaluations are wasted; the
#        logical n_evals still counts only the particles the scan consumed.)
# NEW 3  Exact lower-bound frequency pruning ("fast", "wb_prune"). The peak
#        over any frequency subset is a lower bound on the full peak. A
#        particle is evaluated on the full grid only if that bound is below
#        its pbest; otherwise it cannot update pbest or gbest, so its exact
#        value is never needed. The subset (stride grid + band edge + bare
#        PDN maxima) grows adaptively with the arg-max frequency of every
#        full evaluation. Trajectory-preserving by construction.
# NEW 4  Configuration cache ("fast", "wb_cache"). Discretised (port,model)
#        sets repeat heavily once a swarm converges; exact costs are cached
#        per run and reused without touching the GPU.
# NEW 5  --fmax: restrict the optimisation band (e.g. 50e6) to avoid the
#        band-edge problem documented in Dataset/Decaps/README.md. Needs
#        --freq. --y/--d/--freq select any dataset (SP, DDR3, MPHY rails).
# NEW 6  Trajectory check: every method's (best_minZ, caps_at_target) is
#        compared with the baseline on the same (threshold, run) and the
#        match rate is reported, so speedups are shown to be like-for-like.
#
# "iterative" (2nd-order Neumann) is dropped: verification showed ~30%
# relative error because Y_eq is far from Y once caps are added, and it was
# the slowest method. "woodbury" is the exact way to reuse the base inverse.
#
# Default roster: numpy (baseline), solve, sm, woodbury, woodbury_batch,
# fast. Optional ablations: wb_cache, wb_prune. Optional CPU references:
# cpu_numpy.
# ============================================================

import os
import time
import json
import math
import logging
import argparse

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

# ============================================================
# CONFIG / ARGS
# ============================================================

BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
parser.add_argument("--quick", action="store_true", help="tiny schedule for smoke tests")
parser.add_argument("--y", default=os.path.join(DATA_DIR, "y2.mat"), help="PDN admittance .mat, var y (N,N,F)")
parser.add_argument("--d", default=os.path.join(DATA_DIR, "decaps.mat"), help="decap library .mat, var decaps (C,F)")
parser.add_argument("--freq", default=None, help="frequency grid .mat, var freq (1,F); needed for --fmax and plots")
parser.add_argument("--fmax", type=float, default=None, help="drop frequencies above this (Hz)")
parser.add_argument("--fmin", type=float, default=None, help="drop frequencies below this (Hz)")
parser.add_argument("--methods", default="numpy,solve,sm,woodbury,woodbury_batch,fast")
parser.add_argument("--baseline", default="numpy")
parser.add_argument("--targets", default="0.05,0.045,0.04,0.03")
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--out", default="MinTime/GPUBench2")
parser.add_argument("--stride", type=int, default=16, help="pruning subset stride")
parser.add_argument("--max-batch-elems", type=float, default=3e7,
                    help="cap on B*F*k*k complex elements per batched solve (memory)")
args, _ = parser.parse_known_args()

ROOT_OUT = args.out
os.makedirs(ROOT_OUT, exist_ok=True)

TARGET_PORT = 0
MAX_CAPS = 20
N_PARTICLES = 50
N_ITERATIONS = 40
W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5
NUM_RUNS = args.runs
TARGETS = [float(t) for t in args.targets.split(",")]
METHODS = [m.strip() for m in args.methods.split(",") if m.strip()]
BASELINE_METHOD = args.baseline
assert BASELINE_METHOD in METHODS, "baseline must be in --methods"
BASE_SEED = 12345
FOLDER_SEP = "__THR__"

SEQUENTIAL_METHODS = {"numpy", "solve", "sm", "woodbury", "cpu_numpy"}
BATCHED_METHODS = {          # name -> (use_cache, use_prune)
    "woodbury_batch": (False, False),
    "wb_cache": (True, False),
    "wb_prune": (False, True),
    "fast": (True, True),
}
for m in METHODS:
    if m not in SEQUENTIAL_METHODS and m not in BATCHED_METHODS:
        raise ValueError(f"Unknown method {m!r}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128 if args.dtype == "complex128" else torch.complex64

REAL_DATA_AVAILABLE = os.path.exists(args.y) and os.path.exists(args.d)
if not REAL_DATA_AVAILABLE or args.quick:
    if not REAL_DATA_AVAILABLE:
        print(f"[WARN] {args.y} / {args.d} not found; using synthetic PDN.")
    NUM_RUNS = min(NUM_RUNS, 3)
    TARGETS = TARGETS[:2]
    MAX_CAPS = 5
    N_PARTICLES = 8
    N_ITERATIONS = 4

WARM_START_FRACTION = 0.4
N_WARM_START = int(round(WARM_START_FRACTION * N_PARTICLES))
WARM_START_JITTER = 0.02

print(f"[GPU] device={DEVICE} dtype={CDTYPE}")
if DEVICE.type == "cuda":
    print(f"[GPU] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")


def cuda_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# DATA
# ============================================================

def _generate_synthetic_pdn(n_nodes, n_freq, n_cap_models, seed=0):
    rng = np.random.default_rng(seed)
    freqs = np.logspace(4, 8, n_freq)
    w = 2.0 * np.pi * freqs
    R_shunt = rng.uniform(5.0, 20.0, size=n_nodes)
    L_shunt = rng.uniform(1e-9, 8e-9, size=n_nodes)
    C_shunt = rng.uniform(1e-10, 5e-10, size=n_nodes)
    R_link = rng.uniform(0.02, 0.08, size=n_nodes)
    L_link = rng.uniform(0.5e-9, 3e-9, size=n_nodes)
    y = np.zeros((n_freq, n_nodes, n_nodes), dtype=np.complex128)
    Y_sh = (1 / R_shunt)[None] + 1 / (1j * w[:, None] * L_shunt[None]) + 1j * w[:, None] * C_shunt[None]
    Y_ln = 1 / (R_link[None] + 1j * w[:, None] * L_link[None])
    for k in range(n_nodes):
        j = (k + 1) % n_nodes
        y[:, k, k] += Y_sh[:, k] + Y_ln[:, k]
        y[:, j, j] += Y_ln[:, k]
        y[:, k, j] -= Y_ln[:, k]
        y[:, j, k] -= Y_ln[:, k]
    y += 1e-6 * np.eye(n_nodes)[None]
    d = np.zeros((n_cap_models, n_freq), dtype=np.complex128)
    for m in range(n_cap_models):
        c = rng.integers(0, 5)
        C = 10 ** rng.uniform(-9 + c * 0.4, -6 + c * 0.5)
        L = 10 ** rng.uniform(-10.3 + c * 0.15, -9.6 + c * 0.15)
        R = 10 ** rng.uniform(-2.3, -0.7)
        d[m] = 1 / (R + 1j * w * L + 1 / (1j * w * C))
    return y, d, freqs


print("Loading PDN data...")
_t0 = time.perf_counter()
if REAL_DATA_AVAILABLE:
    y_np = np.transpose(sio.loadmat(args.y)["y"], (2, 0, 1))
    d_np = sio.loadmat(args.d)["decaps"]
    freq_np = sio.loadmat(args.freq)["freq"].ravel() if args.freq else None
else:
    y_np, d_np, freq_np = _generate_synthetic_pdn(8, 40, 150)

assert y_np.shape[0] == d_np.shape[1], f"PDN has {y_np.shape[0]} freqs, library {d_np.shape[1]}"
if args.fmax is not None or args.fmin is not None:
    assert freq_np is not None, "--fmax/--fmin need --freq"
    keep = np.ones(len(freq_np), bool)
    if args.fmax is not None:
        keep &= freq_np <= args.fmax
    if args.fmin is not None:
        keep &= freq_np >= args.fmin
    y_np, d_np, freq_np = y_np[keep], d_np[:, keep], freq_np[keep]
    print(f"[band] restricted to {freq_np[0]:.4g}-{freq_np[-1]:.4g} Hz ({keep.sum()} pts)")

y_np = np.ascontiguousarray(y_np, dtype=np.complex128)
d_np = np.ascontiguousarray(d_np, dtype=np.complex128)
N_FREQS, N_NODES, _ = y_np.shape
N_CAP_MODELS = d_np.shape[0]
LOAD_TIME_S = time.perf_counter() - _t0
print(f"N_FREQS={N_FREQS} N_NODES={N_NODES} N_CAP_MODELS={N_CAP_MODELS} real_data={REAL_DATA_AVAILABLE}")

VALID_PORTS = np.array([p for p in range(N_NODES) if p != TARGET_PORT], dtype=int)


def frac_to_index(frac, K):
    return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)   # FIX 2


def decode_particle(vec, n_caps):
    return (frac_to_index(vec[:n_caps], N_CAP_MODELS),
            VALID_PORTS[frac_to_index(vec[n_caps:], len(VALID_PORTS))])     # FIX 3


def resolve_adjacent_ports(models, ports):                                   # FIX 12
    used = {TARGET_PORT}
    for i in range(len(ports)):
        if ports[i] not in used:
            used.add(ports[i])
            continue
        placed = False
        for off in range(1, N_NODES):
            for cand in (ports[i] + off, ports[i] - off):
                if 0 <= cand < N_NODES and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            used.add(ports[i])
    return models, ports


def decode_swarm(particles, n_caps):
    """Decode every particle -> (P,k) model and port arrays (same mapping as
    the per-particle path, so trajectories are identical)."""
    P = particles.shape[0]
    M = np.empty((P, n_caps), dtype=np.int64)
    Pt = np.empty((P, n_caps), dtype=np.int64)
    for i in range(P):
        m, p = decode_particle(particles[i], n_caps)
        m, p = resolve_adjacent_ports(m, p)
        M[i], Pt[i] = m, p
    return M, Pt


# ---- device tensors ----
_t0 = time.perf_counter()
if DEVICE.type == "cuda":
    torch.zeros(1, device=DEVICE)
    cuda_sync()
CUDA_INIT_TIME_S = time.perf_counter() - _t0

_t0 = time.perf_counter()
Y = torch.from_numpy(y_np).to(DEVICE, CDTYPE)                 # (F,n,n)
D = torch.from_numpy(d_np).to(DEVICE, CDTYPE)                 # (C,F)
E1 = torch.zeros(N_FREQS, N_NODES, 1, dtype=CDTYPE, device=DEVICE)
E1[:, TARGET_PORT, 0] = 1
cuda_sync()
TRANSFER_TIME_S = time.perf_counter() - _t0

_t0 = time.perf_counter()
Z0 = torch.linalg.inv(Y)                                       # (F,n,n) base impedance
_Dsafe = torch.where(D.abs() < 1e-300, torch.full_like(D, 1e-300), D)
DINV = 1.0 / _Dsafe                                            # (C,F) cap impedances
Z0_TT = Z0[:, TARGET_PORT, TARGET_PORT]                        # (F,)
cuda_sync()
BASE_INV_TIME_S = time.perf_counter() - _t0
INIT_TIME_S = LOAD_TIME_S + CUDA_INIT_TIME_S + TRANSFER_TIME_S + BASE_INV_TIME_S
print(f"[GPU] init load={LOAD_TIME_S:.3f}s ctx={CUDA_INIT_TIME_S:.3f}s h2d={TRANSFER_TIME_S:.3f}s "
      f"base_inv={BASE_INV_TIME_S:.3f}s")

# initial pruning subset: stride grid + band edges + local maxima of bare |Z11|
_bare = np.abs(np.linalg.inv(y_np)[:, TARGET_PORT, TARGET_PORT])
_lmax = [i for i in range(1, N_FREQS - 1) if _bare[i] >= _bare[i - 1] and _bare[i] >= _bare[i + 1]]
_lmax = sorted(_lmax, key=lambda i: -_bare[i])[:16]
BASE_SUBSET = sorted(set(range(0, N_FREQS, max(1, args.stride))) | {0, N_FREQS - 1} | set(_lmax))


# ============================================================
# FITNESS KERNELS
# ============================================================

def assemble_Yeq(config):
    A = Y.clone()
    for cap, port in config:
        A[:, port, port] += D[cap, :]
    return A


def k_numpy(config):
    return torch.linalg.inv(assemble_Yeq(config))[:, TARGET_PORT, TARGET_PORT]


def k_solve(config):
    return torch.linalg.solve(assemble_Yeq(config), E1)[:, TARGET_PORT, 0]


def k_sm(config):
    Z = Z0.clone()
    for cap, port in config:
        val = D[cap, :]
        denom = 1.0 + val * Z[:, port, port]
        if torch.any(denom.abs() < 1e-12):
            raise torch.linalg.LinAlgError("Sherman-Morrison denominator ~ 0")
        Z = Z - (Z[:, :, port].unsqueeze(2) * (val.unsqueeze(1) * Z[:, port, :]).unsqueeze(1)) \
            / denom.view(-1, 1, 1)
    return Z[:, TARGET_PORT, TARGET_PORT]


def wb_peak(models_t, ports_t, fidx=None):
    """
    Batched Woodbury fitness. models_t, ports_t: LongTensor (B,k) on DEVICE.
    fidx: optional LongTensor of frequency indices (subset). Returns
    (peak (B,), argmax_freq_global (B,)) as device tensors. Singular systems
    score 1e200, mirroring the LinAlgError handling of the other methods.
    """
    B, k = ports_t.shape
    Z0f = Z0 if fidx is None else Z0.index_select(0, fidx)            # (F',n,n)
    Dv = DINV if fidx is None else DINV.index_select(1, fidx)         # (C,F')
    Fp = Z0f.shape[0]
    peaks, args_ = [], []
    chunk = max(1, int(args.max_batch_elems // max(1, Fp * k * k)))
    for s in range(0, B, chunk):
        pt, mt = ports_t[s:s + chunk], models_t[s:s + chunk]
        M = Z0f[:, pt.unsqueeze(2), pt.unsqueeze(1)].permute(1, 0, 2, 3)   # (b,F',k,k)
        M = M + torch.diag_embed(Dv[mt].transpose(1, 2))                   # + diag(1/d)
        zc = Z0f[:, pt, TARGET_PORT].permute(1, 0, 2).unsqueeze(-1)        # (b,F',k,1)
        zr = Z0f[:, TARGET_PORT, pt].permute(1, 0, 2).unsqueeze(-2)        # (b,F',1,k)
        x, info = torch.linalg.solve_ex(M, zc)
        val = Z0f[:, TARGET_PORT, TARGET_PORT].unsqueeze(0) - (zr @ x)[..., 0, 0]
        pk, am = val.abs().max(dim=1)
        bad = (info != 0).any(dim=1) | ~torch.isfinite(pk)
        peaks.append(torch.where(bad, torch.full_like(pk, 1e200), pk))
        args_.append(am)
    peak, am = torch.cat(peaks), torch.cat(args_)
    if fidx is not None:
        am = fidx[am]
    return peak, am


def k_woodbury(config):
    m = torch.tensor([[c for c, _ in config]], device=DEVICE)
    p = torch.tensor([[q for _, q in config]], device=DEVICE)
    return wb_peak(m, p)[0][0]


def _cpu_numpy_peak(config):
    peak = 0.0
    for f in range(N_FREQS):
        A = y_np[f].copy()
        for cap, port in config:
            A[port, port] += d_np[cap, f]
        try:
            val = np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]
        except np.linalg.LinAlgError:
            return 1e200
        peak = max(peak, abs(val))
    return peak


SEQ_KERNELS = {"numpy": k_numpy, "solve": k_solve, "sm": k_sm}


def evaluate_config(config, method):
    """Sequential (one particle) evaluation -> float."""
    if method == "cpu_numpy":
        return _cpu_numpy_peak(config)
    try:
        if method == "woodbury":
            return float(k_woodbury(config).item())        # already a peak
        vals = SEQ_KERNELS[method](config)
        return float(vals.abs().max().item())
    except torch.linalg.LinAlgError:
        return 1e200


class SwarmEvaluator:
    """Batched evaluation of a whole swarm with optional cache + exact
    lower-bound pruning. State is reset per run so each timed run is
    independent."""

    def __init__(self, use_cache, use_prune):
        self.use_cache, self.use_prune = use_cache, use_prune
        self.reset()

    def reset(self):
        self.cache = {}
        self.subset = set(BASE_SUBSET)
        self._subset_t = None
        self.n_full = self.n_screen = self.n_hits = self.n_pruned = 0

    def _fidx(self):
        if self._subset_t is None:
            self._subset_t = torch.tensor(sorted(self.subset), device=DEVICE)
        return self._subset_t

    def evaluate(self, models, ports, pbest_val):
        P = models.shape[0]
        cost = np.empty(P)
        todo = []
        keys = [None] * P
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
            lb, _ = wb_peak(mt, pt, self._fidx())
            lb = lb.cpu().numpy()
            self.n_screen += len(idx)
            # exact need iff the bound could beat pbest (tiny margin for rounding)
            need_full = lb < pbest_val[idx] * (1 + 1e-9)
            cost[idx[~need_full]] = lb[~need_full]       # >= pbest: cannot update anything
            self.n_pruned += int((~need_full).sum())

        if need_full.any():
            sel = np.nonzero(need_full)[0]
            sel_t = torch.from_numpy(sel).to(DEVICE)
            pk, am = wb_peak(mt.index_select(0, sel_t), pt.index_select(0, sel_t))
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


EVALUATORS = {m: SwarmEvaluator(*BATCHED_METHODS[m]) for m in METHODS if m in BATCHED_METHODS}


# ============================================================
# VERIFICATION + WARM-UP
# ============================================================

def verify(n_trials=6):
    rng = np.random.default_rng(999)
    report = {}
    cfgs = []
    for _ in range(n_trials):
        n = int(rng.integers(1, min(6, len(VALID_PORTS)) + 1))
        m, p = decode_particle(rng.random(2 * n), n)
        m, p = resolve_adjacent_ports(m, p)
        cfgs.append(list(zip(m.tolist(), p.tolist())))
    refs = [_cpu_numpy_peak(c) for c in cfgs]
    for method in METHODS:
        if method == "cpu_numpy":
            continue
        errs = []
        for c, r in zip(cfgs, refs):
            if method in BATCHED_METHODS:
                ev = SwarmEvaluator(*BATCHED_METHODS[method])
                k = len(c)
                M = np.array([[x for x, _ in c]] * 2)
                Pp = np.array([[q for _, q in c]] * 2)
                got = ev.evaluate(M, Pp, np.full(2, np.inf))[0]
            else:
                got = evaluate_config(c, method)
            errs.append(abs(got - r) / max(abs(r), 1e-30))
        report[method] = float(max(errs))
        print(f"[verify] {method:<15} max_rel={report[method]:.3e}")
        if CDTYPE == torch.complex128:
            assert report[method] < 1e-6, f"{method} disagrees with CPU reference"
    with open(os.path.join(ROOT_OUT, "verification.json"), "w") as f:
        json.dump(report, f, indent=2)


verify()


def warmup():
    cfg = [(0, int(VALID_PORTS[i % len(VALID_PORTS)])) for i in range(min(2, len(VALID_PORTS)))]
    for _ in range(3):
        for m in METHODS:
            if m in BATCHED_METHODS:
                M = np.array([[c for c, _ in cfg]] * N_PARTICLES)
                Pp = np.array([[p for _, p in cfg]] * N_PARTICLES)
                SwarmEvaluator(False, True).evaluate(M, Pp, np.zeros(N_PARTICLES))
                SwarmEvaluator(False, False).evaluate(M, Pp, np.full(N_PARTICLES, np.inf))
            elif m != "cpu_numpy":
                evaluate_config(cfg, m)
    cuda_sync()


_t0 = time.perf_counter()
warmup()
print(f"[GPU] warm-up {time.perf_counter() - _t0:.3f}s")


# ============================================================
# PSO
# ============================================================

def setup_logger(path):
    lg = logging.getLogger(path)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter('%(asctime)s INFO: %(message)s'))
    lg.addHandler(h)
    return lg


def seed_particles(n_caps, prev_best):                                       # FIX 11
    n_warm = min(N_WARM_START, N_PARTICLES)
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(N_PARTICLES, 2 * n_caps)
    pn = n_caps - 1
    warm = np.hstack([np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
                      np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1)])
    if WARM_START_JITTER > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, bool)
        carried[pn] = False
        carried[-1] = False
        noise = np.random.normal(0.0, WARM_START_JITTER, warm.shape)
        noise[0, :] = 0.0
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    if N_PARTICLES - n_warm > 0:
        warm = np.vstack([warm, np.random.rand(N_PARTICLES - n_warm, 2 * n_caps)])
    return warm


def run_pso(method, threshold, out_folder, run_id):
    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))
    batched = method in BATCHED_METHODS
    ev = EVALUATORS.get(method)
    if ev:
        ev.reset()

    run_caps, run_minZ = [], []
    cuda_sync()
    start = time.perf_counter()
    time_to_target = caps_at_target = None
    n_evals = 0
    prev_best = None

    for n_caps in range(1, MAX_CAPS + 1):
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)                  # FIX 1
        particles = seed_particles(n_caps, prev_best)
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_val = np.full(N_PARTICLES, np.inf)
        gbest, gbest_particle = np.inf, None
        stop = False
        it = 0

        for it in range(N_ITERATIONS):
            if batched:
                M, Pp = decode_swarm(particles, n_caps)
                costs = ev.evaluate(M, Pp, pbest_val)
            for i in range(N_PARTICLES):
                if batched:
                    cost = costs[i]
                else:
                    m, p = decode_particle(particles[i], n_caps)
                    m, p = resolve_adjacent_ports(m, p)
                    cost = evaluate_config(list(zip(m.tolist(), p.tolist())), method)
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]
                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()
                if gbest <= threshold:          # immediate stop, same order as sequential
                    stop = True
                    break
            if stop:
                if time_to_target is None:
                    cuda_sync()
                    time_to_target = time.perf_counter() - start
                    caps_at_target = n_caps
                break
            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)
            velocities = w * velocities \
                + C1 * np.random.rand(*particles.shape) * (pbest - particles) \
                + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles)
            particles = np.clip(particles + velocities, 0, 1)

        if gbest_particle is None:                                           # FIX 13
            prev_best = None
            continue
        prev_best = gbest_particle
        bm, bp = decode_particle(gbest_particle, n_caps)
        bm, bp = resolve_adjacent_ports(bm, bp)
        logger.info("n_caps=%d | iter=%d | minZ=%.6f | placement=%s", n_caps, it + 1, gbest,
                    {int(p): int(c) for c, p in zip(bm, bp)})
        run_caps.append(n_caps)
        run_minZ.append(gbest)
        if gbest <= threshold:
            break

    cuda_sync()
    total = time.perf_counter() - start
    best = min(run_minZ) if run_minZ else float("inf")
    rec = {
        "threshold": threshold, "method": method, "run_id": run_id,
        "time_to_target_s": time_to_target, "total_wall_time_s": total,
        "init_time_s": INIT_TIME_S, "caps_at_target": caps_at_target,
        "best_minz_ohm": best, "success": time_to_target is not None,
        "n_evals": n_evals, "final_n_caps": run_caps[-1] if run_caps else None,
        "gpu_full_evals": ev.n_full if ev else n_evals,
        "gpu_screen_evals": ev.n_screen if ev else 0,
        "cache_hits": ev.n_hits if ev else 0,
        "pruned": ev.n_pruned if ev else 0,
        "subset_size": len(ev.subset) if ev else 0,
    }
    logger.info("RESULT | %s", json.dumps(rec))
    with open(os.path.join(ROOT_OUT, "results_long.jsonl"), "a") as jf:
        jf.write(json.dumps(rec) + "\n")
    return rec


# ============================================================
# MAIN
# ============================================================

open(os.path.join(ROOT_OUT, "results_long.jsonl"), "w").close()
records = []
for thr in TARGETS:
    for method in METHODS:
        print(f"\nMETHOD {method} TARGET {thr}", flush=True)
        for run_id in range(1, NUM_RUNS + 1):
            folder = os.path.join(ROOT_OUT, f"{method}{FOLDER_SEP}{thr}", f"run_{run_id}")
            os.makedirs(folder, exist_ok=True)
            r = run_pso(method, thr, folder, run_id)
            records.append(r)
            print(f"  run {run_id:2d}: {r['total_wall_time_s']:.3f}s caps={r['caps_at_target']} "
                  f"minZ={r['best_minz_ohm']:.5f} full={r['gpu_full_evals']} hits={r['cache_hits']} "
                  f"pruned={r['pruned']}", flush=True)

# ============================================================
# ANALYSIS
# ============================================================

df = pd.DataFrame(records)
df.to_csv(os.path.join(ROOT_OUT, "results_long.csv"), index=False)

base = df[df.method == BASELINE_METHOD].set_index(["threshold", "run_id"])
df["speedup"] = df.apply(lambda r: base.loc[(r.threshold, r.run_id), "total_wall_time_s"]
                         / r.total_wall_time_s, axis=1)
df["traj_match"] = df.apply(
    lambda r: (abs(base.loc[(r.threshold, r.run_id), "best_minz_ohm"] - r.best_minz_ohm)
               <= 1e-9 * max(1.0, abs(r.best_minz_ohm)))
    and (base.loc[(r.threshold, r.run_id), "final_n_caps"] == r.final_n_caps), axis=1)


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0


rows = []
for (thr, m), g in df.groupby(["threshold", "method"]):
    rows.append({
        "threshold": thr, "method": m,
        "time_s_mean": g.total_wall_time_s.mean(), "time_s_ci95": ci95(g.total_wall_time_s),
        "speedup_mean": g.speedup.mean(), "speedup_ci95": ci95(g.speedup),
        "caps_mean": g.caps_at_target.dropna().mean() if g.caps_at_target.notna().any() else np.nan,
        "best_minz_mean": g.best_minz_ohm.mean(), "success_rate": g.success.mean(),
        "n_evals_mean": g.n_evals.mean(), "gpu_full_evals_mean": g.gpu_full_evals.mean(),
        "cache_hits_mean": g.cache_hits.mean(), "pruned_mean": g.pruned.mean(),
        "trajectory_match_rate": g.traj_match.mean(), "n_runs": len(g),
    })
summary = pd.DataFrame(rows).sort_values(["threshold", "method"])
summary.to_csv(os.path.join(ROOT_OUT, "summary_table.csv"), index=False)
summary.pivot(index="method", columns="threshold", values="speedup_mean") \
    .to_csv(os.path.join(ROOT_OUT, "speedup_matrix.csv"))

with open(os.path.join(ROOT_OUT, "statistics.txt"), "w") as f:
    f.write(summary.to_string(index=False))
with open(os.path.join(ROOT_OUT, "timeouts.txt"), "w") as f:
    for (thr, m), g in df[~df.success].groupby(["threshold", "method"]):
        f.write(f"{m} T={thr}: runs {g.run_id.tolist()} best_minZ {np.round(g.best_minz_ohm.values, 5).tolist()}\n")

figs = os.path.join(ROOT_OUT, "figs")
os.makedirs(figs, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for m, g in summary.groupby("method"):
    ax[0].errorbar(g.threshold, g.time_s_mean, yerr=g.time_s_ci95, marker="o", capsize=3, label=m)
    ax[1].errorbar(g.threshold, g.speedup_mean, yerr=g.speedup_ci95, marker="o", capsize=3, label=m)
ax[0].set_yscale("log")
ax[0].set_ylabel("Wall time per run (s)")
ax[1].set_ylabel(f"Speedup vs {BASELINE_METHOD}")
ax[1].set_yscale("log")
for a in ax:
    a.set_xlabel("Target |Z11| (Ω)")
    a.grid(True, which="both", alpha=.3)
    a.legend()
fig.tight_layout()
fig.savefig(os.path.join(figs, "runtime_and_speedup.png"), dpi=200)
plt.close(fig)

bm = [m for m in METHODS if m in BATCHED_METHODS]
if bm:
    fig, a = plt.subplots(figsize=(9, 5))
    s = summary[summary.method.isin(bm)].groupby("method")[
        ["gpu_full_evals_mean", "pruned_mean", "cache_hits_mean"]].mean()
    s.plot.bar(stacked=True, ax=a)
    a.set_ylabel("evaluations per run (mean over thresholds)")
    a.set_title("Where the batched methods' evaluations went")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "eval_breakdown.png"), dpi=200)
    plt.close(fig)

print("\n" + summary[["threshold", "method", "time_s_mean", "speedup_mean",
                      "trajectory_match_rate", "success_rate"]].to_string(index=False))
print(f"\nOutputs under {ROOT_OUT}\nDone.")
