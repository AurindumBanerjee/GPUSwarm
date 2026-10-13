# nohup python GPUBench.py > GPUBench.log 2>&1 &
# ps -u $USER | grep GPUBench
# pkill -9 -f GPUBench.py
#
# ============================================================
# GPU PORT OF ScratchBench.py, UPDATED WITH THE SAME FIXES APPLIED TO
# THE CORRECTED CPU SCRIPT (ScratchBench.py)
# ============================================================
# Numbering below matches the CPU corrected script's FIX list so the two
# can be cross-referenced directly.
#
# FIX 1  Deterministic, paired seeding, done PER STAGE not just per run.
#        The previous GPU version seeded once per (threshold, method, run)
#        at the top level. That only pairs stage-1 particles across
#        methods -- the instant one method's early-stop timing diverges
#        from another's, the shared RNG stream has consumed a different
#        number of draws and every later stage drifts apart. Seeding is
#        now done inside run_pso() at the top of EVERY n_caps stage, keyed
#        only on (run_id, n_caps) -- independent of threshold and of how
#        much randomness earlier stages/methods consumed -- so every
#        method starts stage n_caps from an identical swarm regardless of
#        how the earlier stages played out.
#
# FIX 2  Off-by-one AND overflow-safety in discretisation. map_port_fracs
#        used `floor(frac*(len(VALID_PORTS)-1))`, and the model dimension
#        used `floor(frac*(N_CAP_MODELS-1))` -- both can never select the
#        last valid entry (frac in [0,1) tops out one index short),
#        silently shrinking the search space. frac_to_index() now uses
#        floor(frac*K) clipped to [0,K-1] (the clip guards the frac==1.0
#        edge case produced by np.clip(particles,0,1) after a velocity
#        update, which would otherwise index out of bounds).
#
# FIX 3  TARGET_PORT exclusion -- ALREADY PRESENT in this GPU script
#        (VALID_PORTS + resolve_adjacent_ports reserving TARGET_PORT).
#        Unchanged; frac_to_index/decode_particle below just centralise
#        the mapping so FIX 2 only has to be fixed in one place.
#
# FIX 4  Warm-start between capacitor-count stages, per the paper's
#        Section III-A -- ADDED HERE for the first time. The GPU version
#        previously fully re-randomised every stage (`particles =
#        np.random.rand(N_PARTICLES, DIM)`), discarding all prior search
#        progress every time n_caps incremented. seed_particles() now
#        seeds every particle's first 2N genes identically from the
#        previous stage's winning particle; only the two new genes for
#        the (N+1)-th capacitor are randomised per particle.
#
# FIX 5  SM/matrix-assembly overhead avoidance -- ALREADY PRESENT
#        (evaluate_config only calls assemble_Yeq() for methods that need
#        the full Y_eq tensor; inv_sm rebuilds from the precomputed base
#        inverse). Unchanged.
#
# FIX 6  No more bare `except Exception`. The previous evaluate_config
#        wrapped everything (including the `else: raise ValueError(...)`
#        for an unknown method) in one broad except, so a typo'd method
#        name would silently score as cost=1e200 instead of failing
#        loudly. Method validation now happens BEFORE the try block, and
#        only torch.linalg.LinAlgError (the expected near-singular-system
#        failure, exactly mirroring np.linalg.LinAlgError on the CPU
#        side) is caught during evaluation.
#
# FIX 7/8  Timeouts are no longer only visible by filtering the
#        DataFrame -- a timeouts.txt is written alongside statistics.txt,
#        grouped by (method, threshold), listing every run that hit
#        MAX_CAPS without reaching its target and the best |Z11| it
#        managed. (The GPU script already logs best_minz_ohm/success in
#        every JSON record, unlike the original CPU script's regex log
#        scraping -- this just makes non-convergence visible without a
#        manual pandas query.)
#
# FIX 9  Folder names use FOLDER_SEP ("__THR__") instead of a bare
#        underscore, matching the CPU corrected script. Not load-bearing
#        here (this script's analysis reads structured JSON records, it
#        never re-parses folder names), but it keeps folder layout
#        consistent across both scripts and avoids ambiguity now that
#        method names like "cpu_numpy" and "pure_python" contain
#        underscores themselves.
#
# ADDITION  pure_python baseline, ported from the CPU corrected script:
#        Gauss-Jordan elimination with partial pivoting, plain Python
#        lists + the built-in `complex` type, zero NumPy/SciPy/PyTorch on
#        its timed path. Included for API/roster parity with the CPU
#        study, but -- like cpu_numpy -- it is NOT in the default METHODS
#        list. Running a hand-written O(n^3) Python inner loop makes no
#        sense to call "GPU"; it exists here purely as an optional
#        apples-to-apples reference point, same as cpu_numpy already was.
#        Add "pure_python" to METHODS to enable it; expect it to be
#        drastically slower than every other method at full paper scale.
#
# Everything else (batched torch.linalg.inv/solve, Sherman-Morrison and
# Neumann-series kernels, CUDA synchronisation/timing discipline,
# host<->device transfer accounting, warm-up, GPU-vs-CPU verification,
# structured JSON logging, and all figures) is UNCHANGED from the
# previous GPU script.
# ============================================================

import os
import time
import json
import math
import logging
import shutil
import argparse

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import torch  # [GPU] PyTorch replaces NumPy on the timed fitness path

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "MinTime/GPUTest2"
os.makedirs(ROOT_OUT, exist_ok=True)

TARGET_PORT = 0
MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 40

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

NUM_RUNS = 20
TARGETS = [0.05, 0.045, 0.04, 0.03]

# "cpu_numpy" (original NumPy CPU path) and "pure_python" (zero-NumPy
# Gauss-Jordan) are both implemented below for optional comparison but
# NOT in the default roster -- add either to METHODS to time it.
METHODS = ["numpy", "solve", "sm", "iterative"]
BASELINE_METHOD = "numpy"
assert BASELINE_METHOD in METHODS

BASE_SEED = 12345
FOLDER_SEP = "__"   # FIX 9

# [GPU] Device + dtype. complex128 mirrors the CPU study's float64/complex128
# precision exactly, so numerical results are directly comparable. Set
# --dtype complex64 only if you knowingly accept reduced precision.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128

# ---- data locations ----
BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")
Y_MAT_PATH = os.path.join(DATA_DIR, "y2.mat")
D_MAT_PATH = os.path.join(DATA_DIR, "decaps.mat")

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
parser.add_argument("--quick", action="store_true",
                    help="tiny run schedule for smoke-testing")
args, _ = parser.parse_known_args()
CDTYPE = torch.complex128 if args.dtype == "complex128" else torch.complex64
RDTYPE = torch.float64 if CDTYPE == torch.complex128 else torch.float32

REAL_DATA_AVAILABLE = os.path.exists(Y_MAT_PATH) and os.path.exists(D_MAT_PATH)

if not REAL_DATA_AVAILABLE or args.quick:
    if not REAL_DATA_AVAILABLE:
        print(f"[WARN] Real PDN data not found under {DATA_DIR}; using synthetic PDN.")
    NUM_RUNS = 3
    TARGETS = [0.05, 0.04]
    MAX_CAPS = 5
    N_PARTICLES = 8
    N_ITERATIONS = 4

print(f"[GPU] device={DEVICE}  dtype={CDTYPE}")
if DEVICE.type == "cuda":
    print(f"[GPU] {torch.cuda.get_device_name(0)}  "
          f"torch={torch.__version__}  cuda={torch.version.cuda}")


def cuda_sync():
    """[GPU] CUDA kernels are asynchronous; every timed section must be
    bracketed by a synchronize or the measured time is meaningless."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# LOGGING / PER-RUN PLOTS  (unchanged from the CPU version)
# ============================================================

def setup_logger(path):
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter('%(asctime)s INFO: %(message)s'))
    logger.addHandler(handler)
    return logger


def plot_run_curve(run_caps, run_minz, threshold, method, run_id, out_folder):
    if not run_caps:
        return
    xs = np.array(run_caps, dtype=int)
    ys = np.array(run_minz, dtype=float)
    best_so_far = np.minimum.accumulate(ys)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker='o', linewidth=1.5, label='minZ per n_caps')
    plt.plot(xs, best_so_far, marker='s', linestyle='--', linewidth=1.5,
             label='best-so-far envelope')
    plt.axhline(y=threshold, linestyle='--', linewidth=1.0, label=f'target={threshold}')
    plt.xlabel("Decaps")
    plt.ylabel("Peak |Z11| (Ohm)")
    plt.title(f"{method} T={threshold} Run {run_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "run_plot.png"), dpi=200)
    plt.close()


def plot_convergence(history_dict, threshold, out_folder):
    if not history_dict:
        return
    plt.figure(figsize=(9, 5))
    for n_caps, hist in history_dict.items():
        if hist:
            plt.plot(hist, label=f"n={n_caps}")
    plt.axhline(y=threshold, linestyle='--', linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Peak |Z11| (Ohm)")
    plt.title("Global Convergence")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "global_convergence.png"), dpi=200)
    plt.close()


# ============================================================
# SYNTHETIC DATA FALLBACK
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
    Y_shunt = (1.0 / R_shunt)[None, :] + 1.0 / (1j * w[:, None] * L_shunt[None, :]) \
        + 1j * w[:, None] * C_shunt[None, :]
    Z_link = R_link[None, :] + 1j * w[:, None] * L_link[None, :]
    Y_link = 1.0 / Z_link

    for k in range(n_nodes):
        j = (k + 1) % n_nodes
        y[:, k, k] += Y_shunt[:, k] + Y_link[:, k]
        y[:, j, j] += Y_link[:, k]
        y[:, k, j] -= Y_link[:, k]
        y[:, j, k] -= Y_link[:, k]

    y += (1e-6 * np.eye(n_nodes, dtype=np.complex128))[None, :, :]

    d = np.zeros((n_cap_models, n_freq), dtype=np.complex128)
    for m in range(n_cap_models):
        case = rng.integers(0, 5)
        cap_val = 10.0 ** rng.uniform(-9 + case * 0.4, -6 + case * 0.5)
        esl = 10.0 ** rng.uniform(-10.3 + case * 0.15, -9.6 + case * 0.15)
        esr = 10.0 ** rng.uniform(-2.3, -0.7)
        z = esr + 1j * w * esl + 1.0 / (1j * w * cap_val)
        d[m, :] = 1.0 / z
    return y, d


# ============================================================
# LOAD DATA  +  HOST->DEVICE TRANSFER (timed separately)
# ============================================================

print("Loading PDN data...")
_t_load0 = time.perf_counter()

if REAL_DATA_AVAILABLE:
    y_np = sio.loadmat(Y_MAT_PATH)["y"]
    d_np = sio.loadmat(D_MAT_PATH)["decaps"]
    y_np = np.transpose(y_np, (2, 0, 1))
else:
    y_np, d_np = _generate_synthetic_pdn(n_nodes=8, n_freq=40, n_cap_models=150, seed=0)

y_np = np.ascontiguousarray(y_np, dtype=np.complex128)
d_np = np.ascontiguousarray(d_np, dtype=np.complex128)

N_FREQS, N_NODES, _ = y_np.shape
N_CAP_MODELS = d_np.shape[0]
LOAD_TIME_S = time.perf_counter() - _t_load0

print(f"N_FREQS={N_FREQS}  N_NODES={N_NODES}  N_CAP_MODELS={N_CAP_MODELS}  "
      f"real_data={REAL_DATA_AVAILABLE}")

# TARGET_PORT (port 0) is the measurement port used to read Z11 -- it is
# NOT a valid decap placement site. VALID_PORTS is every other node index;
# all particle port dimensions are mapped onto this reduced set so a
# capacitor can never land on the measurement port in the first place.
VALID_PORTS = np.array([p for p in range(N_NODES) if p != TARGET_PORT], dtype=int)
assert len(VALID_PORTS) >= 1, "need at least one non-measurement port for decap placement"


# ============================================================
# DISCRETISATION  (FIX 2)
# ============================================================

def frac_to_index(frac_array, K):
    """
    Map particle values in [0,1] onto discrete indices {0, ..., K-1}
    uniformly. Uses floor(frac*K), not floor(frac*(K-1)) -- the latter can
    never reach index K-1 for any frac in [0,1) and silently shrinks the
    search space by one entry. Clips as a safeguard for the frac==1.0 edge
    case (particles are clipped to the *closed* interval [0,1] after each
    PSO velocity update), which would otherwise index out of bounds.
    """
    idx = np.floor(np.asarray(frac_array) * K).astype(int)
    return np.clip(idx, 0, K - 1)


def decode_particle(vec, n_caps):
    """
    Decode a particle's [0,1]^(2*n_caps) vector into (models, ports).
    Layout is [model_1..model_n, port_1..port_n]. Centralises FIX 2 (the
    off-by-one) and FIX 3 (TARGET_PORT exclusion via VALID_PORTS) in one
    place instead of duplicating the mapping at every call site.
    """
    model_idx = frac_to_index(vec[:n_caps], N_CAP_MODELS)
    port_idx = frac_to_index(vec[n_caps:], len(VALID_PORTS))
    ports = VALID_PORTS[port_idx]
    return model_idx, ports


# [GPU] One-time CUDA context creation, excluded from the benchmark below.
_t_ctx0 = time.perf_counter()
if DEVICE.type == "cuda":
    torch.zeros(1, device=DEVICE)
    cuda_sync()
CUDA_INIT_TIME_S = time.perf_counter() - _t_ctx0

# [GPU] Single host->device transfer of the whole problem. y and d stay
# resident on the GPU for the entire optimisation; nothing is copied back
# except one scalar per fitness evaluation (the peak |Z11|).
_t_xfer0 = time.perf_counter()
Y = torch.from_numpy(y_np).to(device=DEVICE, dtype=CDTYPE)      # (F, n, n)
D = torch.from_numpy(d_np).to(device=DEVICE, dtype=CDTYPE)      # (C, F)
EYE = torch.eye(N_NODES, dtype=CDTYPE, device=DEVICE)           # (n, n)
# Right-hand side e_1, broadcast over all F frequencies -> (F, n, 1).
E1 = torch.zeros(N_FREQS, N_NODES, 1, dtype=CDTYPE, device=DEVICE)
E1[:, TARGET_PORT, 0] = 1
cuda_sync()
TRANSFER_TIME_S = time.perf_counter() - _t_xfer0

# ============================================================
# PRECOMPUTE BASE INVERSE  (used only by sm and iterative)
# ============================================================
# [GPU] Batched inverse of all F matrices in one call instead of a Python
# loop over frequencies. Timed and reported as initialisation cost.
_t_base0 = time.perf_counter()
Y_INV_BASE = torch.linalg.inv(Y)          # (F, n, n)
cuda_sync()
BASE_INV_TIME_S = time.perf_counter() - _t_base0

print(f"[GPU] init: load={LOAD_TIME_S:.3f}s  cuda_ctx={CUDA_INIT_TIME_S:.3f}s  "
      f"h2d={TRANSFER_TIME_S:.3f}s  base_inv={BASE_INV_TIME_S:.3f}s")

# CPU mirrors, used only by the optional cpu_numpy/pure_python methods and
# the correctness check. Never touched on the GPU timed path.
y_cpu = y_np
d_cpu = d_np
y_inv_base_cpu = np.stack([np.linalg.inv(y_np[f]) for f in range(N_FREQS)])

# ADDITION: plain-Python precompute for the optional pure_python baseline.
# Built once with NumPy's own C-speed .tolist() marshalling -- no NumPy
# call happens on inv_pure_python's timed path below, only list
# indexing/copying/arithmetic, exactly mirroring the CPU corrected script.
Y_LISTS = [y_cpu[f].tolist() for f in range(N_FREQS)]   # F x [n x [n x complex]]
D_LISTS = d_cpu.tolist()                                  # C x [F x complex]


# ============================================================
# PORT RESOLUTION  (unchanged, pure Python/NumPy, off the hot path)
# ============================================================

def resolve_adjacent_ports(models, ports):
    # Defense in depth: TARGET_PORT is reserved as "already used" so that
    # even if a caller passes a raw port in [0, N_NODES), the collision
    # search below can never place (or leave) a capacitor on it.
    used = {TARGET_PORT}
    for i in range(len(ports)):
        if ports[i] not in used:
            used.add(ports[i])
            continue
        for offset in range(1, N_NODES):
            for cand in [ports[i] + offset, ports[i] - offset]:
                if 0 <= cand < N_NODES and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    break
            if ports[i] in used:
                break
    return models, ports


# ============================================================
# MATRIX ASSEMBLY  (batched over all frequencies)
# ============================================================

def assemble_Yeq(config):
    """
    [GPU] Build Y_eq(f) for ALL F frequencies at once:
        Y_eq = Y_PDN + sum_k d_k(f) e_pk e_pk^T
    which is just an addition to diagonal entry (p_k, p_k) of every
    frequency slice. Returns a (F, n, n) complex tensor on the GPU.
    The CPU version did this inside a per-frequency Python loop; the math
    is identical, only the loop is gone.
    """
    A = Y.clone()
    for cap, port in config:
        A[:, port, port] += D[cap, :]     # (F,) vector add, no host round-trip
    return A


# ============================================================
# INVERSION METHODS  (all batched over frequency, all on GPU)
# ============================================================

def inv_numpy(A):
    """
    Conventional full matrix inversion -- the baseline being improved on.
    [GPU] torch.linalg.inv() on the whole (F, n, n) batch; Z11 is element
    (TARGET_PORT, TARGET_PORT) of each inverse. O(n^3) per frequency, as
    in the CPU version. The full inverse is deliberately formed here --
    this is the method under comparison and must not be optimised away.
    """
    return torch.linalg.inv(A)[:, TARGET_PORT, TARGET_PORT]


def inv_solve(A):
    """
    Proposed method: never form the inverse.
    [GPU] torch.linalg.solve(A, e_1) gives x = A^{-1} e_1, whose entry
    x[TARGET_PORT] equals [A^{-1}]_{TARGET_PORT, TARGET_PORT} -- exactly
    the quantity the PSO fitness needs. Batched LU factorisation plus one
    O(n^2) triangular solve per frequency, avoiding the extra O(n^3) work
    of assembling the full inverse.
    Assumption: A is nonsingular (guaranteed for a physical PDN admittance
    matrix); a LinAlgError propagates to the caller exactly as in the CPU
    version, where it is caught and scored as an invalid candidate.
    """
    x = torch.linalg.solve(A, E1)          # (F, n, 1)
    return x[:, TARGET_PORT, 0]


def inv_sm(config):
    """
    Sherman-Morrison rank-1 update, one capacitor at a time, starting from
    the precomputed base inverse. NO Woodbury / multi-rank batch update --
    the capacitors are applied sequentially, matching the CPU version and
    the reference method being compared against.
    [GPU] The rank-1 correction
        Z <- Z - (Z u)(v^T Z) / (1 + v^T Z u)
    is applied to all F frequency slices simultaneously. With u = e_p and
    v = d_k(f) e_p this reduces to a column times a row:
        Z u   = Z[:, :, p]
        v^T Z = d_k(f) * Z[:, p, :]
        denom = 1 + d_k(f) * Z[:, p, p]
    No matrix inversions at runtime; N rank-1 updates of O(n^2) per
    frequency, i.e. O(N n^2 F) per fitness call, as before.
    """
    Z = Y_INV_BASE.clone()                                  # (F, n, n)
    for cap, port in config:
        val = D[cap, :]                                     # (F,)
        Zu = Z[:, :, port]                                  # (F, n)
        vZ = val.unsqueeze(1) * Z[:, port, :]               # (F, n)
        denom = 1.0 + val * Z[:, port, port]                # (F,)
        if torch.any(denom.abs() < 1e-12):
            raise torch.linalg.LinAlgError("Sherman-Morrison denominator ~ 0")
        Z = Z - (Zu.unsqueeze(2) * vZ.unsqueeze(1)) / denom.view(-1, 1, 1)
    return Z[:, TARGET_PORT, TARGET_PORT]


def inv_iterative(A):
    """
    Second-order Neumann series refinement of the precomputed base inverse:
        Y_eq^{-1} ~= Z0 (I - E + E^2),   E = Y_eq Z0 - I
    with a full-inversion fallback wherever ||E||_1 >= 1 (series diverges).
    [GPU] E, the correction and the product are batched matmuls. The
    divergence test is evaluated per frequency (1-norm = max absolute
    column sum) and torch.linalg.inv() is applied ONLY to the diverging
    subset, preserving the CPU version's per-frequency fallback semantics
    rather than falling back wholesale.
    """
    B = Y_INV_BASE
    E = torch.bmm(A, B) - EYE                               # (F, n, n)
    norms = E.abs().sum(dim=1).max(dim=1).values            # ||E||_1 per freq
    diverged = norms >= 1.0

    corr = EYE - E + torch.bmm(E, E)
    Z = torch.bmm(B, corr)
    out = Z[:, TARGET_PORT, TARGET_PORT].clone()

    if bool(diverged.any()):
        idx = diverged.nonzero(as_tuple=True)[0]
        out[idx] = torch.linalg.inv(A[idx])[:, TARGET_PORT, TARGET_PORT]
    return out


# ---- original CPU NumPy path, kept verbatim for CPU-vs-GPU comparison ----

def _cpu_numpy_peak(config):
    peak = 0.0
    for f in range(N_FREQS):
        A = y_cpu[f].copy()
        for cap, port in config:
            A[port, port] += d_cpu[cap, f]
        try:
            val = np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]
        except np.linalg.LinAlgError:
            return 1e200
        peak = max(peak, abs(val))
    return peak


# ---- ADDITION: pure-Python / math-lib-only baseline (no NumPy, no torch) ----

def _gauss_jordan_inverse(A):
    """
    Full matrix inverse via Gauss-Jordan elimination with partial
    pivoting (largest-magnitude pivot in the remaining column). Plain
    Python lists and the built-in `complex` type only -- no NumPy, no
    SciPy, no PyTorch, anywhere in this function. O(n^3) arithmetic, same
    asymptotic cost as the LAPACK/cuSOLVER routines it's being compared
    against, just executed in the interpreter instead of compiled code.
    Ported from the CPU corrected script unchanged.
    """
    n = len(A)
    M = [row[:] + [1.0 + 0.0j if i == r else 0.0 + 0.0j for i in range(n)]
         for r, row in enumerate(A)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-300:
            raise ZeroDivisionError("Singular matrix in pure-Python inversion")
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        inv_pivot = 1.0 / pivot
        row_col = M[col]
        for k in range(2 * n):
            row_col[k] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            row_r = M[r]
            for k in range(2 * n):
                row_r[k] -= factor * row_col[k]

    return [row[n:] for row in M]


def inv_pure_python(config, f):
    """Same math as inv_numpy(), computed with zero NumPy/LAPACK/PyTorch
    involvement: plain Python lists, `complex` arithmetic, Gauss-Jordan."""
    Yeq = [row[:] for row in Y_LISTS[f]]
    for cap, port in config:
        Yeq[port][port] += D_LISTS[cap][f]
    inv = _gauss_jordan_inverse(Yeq)
    return inv[TARGET_PORT][TARGET_PORT]


def _pure_python_peak(config):
    peak = 0.0
    for f in range(N_FREQS):
        try:
            val = inv_pure_python(config, f)
        except ZeroDivisionError:
            return 1e200
        peak = max(peak, abs(val))
    return peak


# ============================================================
# FITNESS
# ============================================================

def evaluate_config(config, method):
    """
    Same objective as the CPU version: peak |Z11| over all frequencies.
    [GPU] The per-frequency Python loop (for the GPU methods) is replaced
    by one batched evaluation; the only device->host transfer is the
    final scalar.

    FIX 6: method is validated BEFORE the try block, so an unknown method
    name raises ValueError immediately and loudly instead of being
    silently absorbed by a catch-all and scored as cost=1e200. Only
    torch.linalg.LinAlgError (the expected near-singular-system failure
    mode) is caught for the GPU methods; _cpu_numpy_peak/_pure_python_peak
    handle their own (np.linalg.LinAlgError / ZeroDivisionError) failure
    modes internally, mirroring the CPU corrected script.
    """
    if method not in ("numpy", "solve", "sm", "iterative", "cpu_numpy", "pure_python"):
        raise ValueError(f"Unknown method: {method!r}")

    if method == "cpu_numpy":
        return _cpu_numpy_peak(config)
    if method == "pure_python":
        return _pure_python_peak(config)

    try:
        if method == "sm":
            vals = inv_sm(config)                 # uses precomputed base inverse
        else:
            A = assemble_Yeq(config)
            if method == "numpy":
                vals = inv_numpy(A)
            elif method == "solve":
                vals = inv_solve(A)
            elif method == "iterative":
                vals = inv_iterative(A)

        peak = vals.abs().max()                   # reduction stays on GPU
        return float(peak.item())                 # single scalar D2H copy
    except torch.linalg.LinAlgError:
        return 1e200


# ============================================================
# CORRECTNESS CHECK: GPU vs original CPU NumPy implementation
# ============================================================

def verify_gpu_against_cpu(n_trials=5, tol=1e-8):
    """
    Requirement 11: confirm the GPU kernels (and, if enabled, the
    pure-Python baseline) reproduce the CPU results. Reports max absolute
    and max relative error of peak |Z11| against the original NumPy CPU
    computation. Uses decode_particle()/resolve_adjacent_ports() -- the
    same discretisation and target-port-exclusion path the real PSO
    search uses -- so this also implicitly validates FIX 2/FIX 3 rather
    than testing raw matrix math on arbitrary indices.
    """
    rng = np.random.default_rng(999)
    methods_to_check = ["numpy", "solve", "sm", "iterative"]
    if "pure_python" in METHODS:
        methods_to_check.append("pure_python")

    report = {}
    for method in methods_to_check:
        max_abs, max_rel = 0.0, 0.0
        for _ in range(n_trials):
            n_caps = int(rng.integers(1, min(4, N_CAP_MODELS, len(VALID_PORTS)) + 1))
            frac_vec = rng.random(2 * n_caps)
            models, ports = decode_particle(frac_vec, n_caps)
            models, ports = resolve_adjacent_ports(list(models), list(ports))
            config = list(zip(models, ports))

            ref = _cpu_numpy_peak(config)
            got = evaluate_config(config, method)
            max_abs = max(max_abs, abs(got - ref))
            max_rel = max(max_rel, abs(got - ref) / max(abs(ref), 1e-30))
        report[method] = {"max_abs_err": max_abs, "max_rel_err": max_rel}
        print(f"[verify] {method:<12} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")

    # inv and solve are mathematically identical, so they must agree to
    # machine precision. sm and iterative are approximations by
    # construction and are reported, not asserted. pure_python should
    # also agree to machine precision (it's an exact inverse too).
    if CDTYPE == torch.complex128:
        for m in ("numpy", "solve"):
            assert report[m]["max_rel_err"] < 1e-6, \
                f"{m} diverged from CPU reference: {report[m]}"
    if "pure_python" in report:
        assert report["pure_python"]["max_rel_err"] < 1e-6, \
            f"pure_python diverged from CPU reference: {report['pure_python']}"

    with open(os.path.join(ROOT_OUT, "gpu_cpu_verification.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


VERIFICATION = verify_gpu_against_cpu()


# ============================================================
# WARM-UP  (requirement 10: exclude lazy kernel/JIT costs from timings)
# ============================================================

def warmup(n_iters=3):
    """[GPU] Run every GPU method a few times before benchmarking so
    cuBLAS/cuSOLVER handle creation and workspace allocation do not land
    inside a timed section. cpu_numpy/pure_python are skipped -- they have
    no GPU context to warm up and pure_python in particular is far too
    slow to spend on a throwaway warm-up call."""
    cfg = [(0, VALID_PORTS[p % len(VALID_PORTS)]) for p in range(min(2, len(VALID_PORTS)))]
    for _ in range(n_iters):
        for m in METHODS:
            if m in ("cpu_numpy", "pure_python"):
                continue
            evaluate_config(cfg, m)
    cuda_sync()


_t_warm0 = time.perf_counter()
warmup()
WARMUP_TIME_S = time.perf_counter() - _t_warm0
print(f"[GPU] warm-up done in {WARMUP_TIME_S:.3f}s")

INIT_TIME_S = (LOAD_TIME_S + CUDA_INIT_TIME_S + TRANSFER_TIME_S + BASE_INV_TIME_S)


# ============================================================
# STRUCTURED RESULT LOGGING
# ============================================================

RESULTS_JSONL_PATH = os.path.join(ROOT_OUT, "results_long.jsonl")


def append_result_record(record):
    with open(RESULTS_JSONL_PATH, "a") as jf:
        jf.write(json.dumps(record) + "\n")
        jf.flush()


# ============================================================
# WARM-START PARTICLE SEEDING  (FIX 4)
# ============================================================

def seed_particles(n_caps, prev_best):
    """
    Stage 1 (or no usable previous best): fully random swarm, as before.
    Stage N+1 with a previous stage's global best available: every
    particle's first 2N genes are seeded IDENTICALLY from prev_best (the
    winning particle from stage N); only the two new genes for the
    (N+1)-th capacitor are randomised per particle. Matches the paper's
    Section III-A warm-start and the CPU corrected script exactly. Particle
    bookkeeping stays on the CPU/NumPy (as documented in run_pso below),
    so this needs no GPU-specific handling.
    """
    if prev_best is None or n_caps == 1:
        return np.random.rand(N_PARTICLES, 2 * n_caps)

    prev_n = n_caps - 1
    prev_models = prev_best[:prev_n]
    prev_ports = prev_best[prev_n:]

    new_model_gene = np.random.rand(N_PARTICLES, 1)
    new_port_gene = np.random.rand(N_PARTICLES, 1)

    particles = np.hstack([
        np.tile(prev_models, (N_PARTICLES, 1)), new_model_gene,
        np.tile(prev_ports, (N_PARTICLES, 1)), new_port_gene,
    ])
    return particles


# ============================================================
# PSO  (logic unchanged apart from FIX 1 reseeding and FIX 4 warm-start;
# timing calls remain GPU-aware)
# ============================================================

def run_pso(method, threshold, out_folder, run_id):

    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))

    run_caps = []
    run_minZ = []
    histories = {}

    cuda_sync()                                   # [GPU] no pending work
    start_global = time.perf_counter()            # [GPU] perf_counter, post-sync
    time_to_target = None
    caps_at_target = None
    n_evals = 0

    logger.info(
        "RUN_START | method=%s | threshold=%.6f | run=%d | particles=%d | "
        "iterations=%d | device=%s | base_seed=%d",
        method, threshold, run_id, N_PARTICLES, N_ITERATIONS, DEVICE.type, BASE_SEED
    )

    prev_best = None   # winning particle vector from the previous n_caps stage

    for n_caps in range(1, MAX_CAPS + 1):

        # FIX 1: reseed per (run, n_caps) -- independent of threshold and
        # of how many RNG draws earlier stages/methods consumed -- so
        # every method sees an identical starting swarm at this stage for
        # this run index, regardless of how quickly earlier stages
        # converged for that method.
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)

        DIM = 2 * n_caps

        # Particle bookkeeping stays on the CPU in NumPy: it is O(P*DIM)
        # per iteration, negligible next to the fitness evaluations, and
        # keeping the same RNG calls guarantees paired methods explore the
        # identical search trajectory for a given (run, n_caps) seed.
        particles = seed_particles(n_caps, prev_best)   # FIX 4
        velocities = np.zeros_like(particles)

        pbest = particles.copy()
        pbest_val = np.full(N_PARTICLES, np.inf)

        gbest = np.inf
        gbest_particle = None
        history = []

        stop_flag = False
        it = 0

        for it in range(N_ITERATIONS):

            for i in range(N_PARTICLES):

                models, ports = decode_particle(particles[i], n_caps)   # FIX 2, FIX 3

                models, ports = resolve_adjacent_ports(models, ports)
                config = list(zip(models, ports))

                cost = evaluate_config(config, method)
                n_evals += 1

                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]

                if cost < gbest:
                    gbest = cost
                    gbest_particle = particles[i].copy()

                # IMMEDIATE STOP: the instant the running best meets the
                # target, stop evaluating the remaining particles in this
                # iteration. We do not wait for the iteration -- let alone
                # the full N_ITERATIONS budget -- to finish.
                if gbest <= threshold:
                    stop_flag = True
                    break

            history.append(gbest)

            if stop_flag:
                if time_to_target is None:
                    cuda_sync()                   # [GPU] before reading the clock
                    time_to_target = time.perf_counter() - start_global
                    caps_at_target = n_caps
                break

            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)

            velocities = w * velocities \
                + C1 * np.random.rand(*particles.shape) * (pbest - particles) \
                + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles)

            particles += velocities
            particles = np.clip(particles, 0, 1)

        histories[n_caps] = history

        if gbest_particle is None:
            logger.info("n_caps=%d | iter=%d | minZ=inf | placement={}", n_caps, it + 1)
            continue

        prev_best = gbest_particle   # FIX 4: carried into the next stage's seeding

        best_models, best_ports = decode_particle(gbest_particle, n_caps)   # FIX 2, FIX 3
        best_models, best_ports = resolve_adjacent_ports(best_models, best_ports)
        placement = {int(port): int(cap) for cap, port in zip(best_models, best_ports)}

        logger.info("n_caps=%d | iter=%d | minZ=%.6f | placement=%s",
                    n_caps, it + 1, gbest, placement)

        run_caps.append(n_caps)
        run_minZ.append(gbest)

        if gbest <= threshold:
            break

    cuda_sync()                                   # [GPU] flush before final timing
    total_wall_time = time.perf_counter() - start_global
    best_minz = min(run_minZ) if run_minZ else float("inf")
    success = time_to_target is not None

    logger.info(
        "RESULT | time_to_target=%.4f | caps=%s | best_minZ=%.6f | total_wall_time=%.4f | success=%s",
        time_to_target if time_to_target else -1,
        caps_at_target if caps_at_target else -1,
        best_minz, total_wall_time, success,
    )

    plot_run_curve(run_caps, run_minZ, threshold, method, run_id, out_folder)
    plot_convergence(histories, threshold, out_folder)

    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": threshold,
        "method": method,
        "run_id": run_id,
        "base_seed": BASE_SEED,
        "time_to_target_s": time_to_target,
        "total_wall_time_s": total_wall_time,     # optimisation time only
        "init_time_s": INIT_TIME_S,               # one-time setup, reported apart
        "total_incl_init_s": total_wall_time + INIT_TIME_S,
        "caps_at_target": caps_at_target,
        "best_minz_ohm": best_minz,
        "success": success,
        "n_fitness_evals": n_evals,
        "n_nodes": N_NODES,
        "n_freqs": N_FREQS,
        "n_cap_models": N_CAP_MODELS,
        "device": DEVICE.type,
        "dtype": str(CDTYPE),
        "folder": out_folder,
    }
    append_result_record(record)
    return record


# ============================================================
# MAIN
# ============================================================

all_records = []

for threshold in TARGETS:
    for method in METHODS:

        print(f"\nMETHOD {method} TARGET {threshold}")

        method_folder = os.path.join(ROOT_OUT, f"{method}{FOLDER_SEP}{threshold}")   # FIX 9
        os.makedirs(method_folder, exist_ok=True)

        for run in range(1, NUM_RUNS + 1):

            run_folder = os.path.join(method_folder, f"run_{run}")
            os.makedirs(run_folder, exist_ok=True)

            # FIX 1: seeding now happens PER STAGE inside run_pso() itself
            # (keyed on run_id & n_caps), not once here at the top level --
            # see run_pso() for why per-stage reseeding is necessary for a
            # genuine paired comparison across methods.

            try:
                all_records.append(run_pso(method, threshold, run_folder, run))
            except Exception as exc:
                logging.getLogger("main").error(
                    "RUN_FAILED | method=%s threshold=%s run=%s | %s",
                    method, threshold, run, exc)
                print(f"[ERROR] run failed: method={method} threshold={threshold} "
                      f"run={run}: {exc}")

print("\nAll runs completed.")


# ============================================================
# GLOBAL ANALYSIS
# ============================================================

def load_records_from_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


records = load_records_from_jsonl(RESULTS_JSONL_PATH)
df = pd.DataFrame(records)

if df.empty:
    print("[WARN] No results recorded -- skipping analysis/plots.")
else:
    results_long_csv = os.path.join(ROOT_OUT, "results_long.csv")
    df.to_csv(results_long_csv, index=False)

    pivot_time = df.pivot_table(index=["threshold", "run_id"], columns="method",
                                values="total_wall_time_s")
    speedup_rows = []
    if BASELINE_METHOD in pivot_time.columns:
        for method in METHODS:
            if method == BASELINE_METHOD or method not in pivot_time.columns:
                continue
            paired = pivot_time[BASELINE_METHOD] / pivot_time[method]
            tmp = paired.reset_index()
            tmp.columns = ["threshold", "run_id", "paired_speedup"]
            tmp["method"] = method
            speedup_rows.append(tmp)
    speedup_long = pd.concat(speedup_rows, ignore_index=True) if speedup_rows else pd.DataFrame()
    if not speedup_long.empty:
        speedup_long.to_csv(os.path.join(ROOT_OUT, "paired_speedup_long.csv"), index=False)

    def _ci95(s):
        n = s.count()
        return 0.0 if n < 2 else 1.96 * s.std(ddof=1) / math.sqrt(n)

    summary = df.groupby(["threshold", "method"]).agg(
        time_s_mean=("total_wall_time_s", "mean"),
        time_s_ci95=("total_wall_time_s", _ci95),
        time_to_target_mean=("time_to_target_s", "mean"),
        caps_mean=("caps_at_target", "mean"),
        best_minz_mean=("best_minz_ohm", "mean"),
        best_minz_ci95=("best_minz_ohm", _ci95),
        success_rate=("success", "mean"),
        n_evals_mean=("n_fitness_evals", "mean"),
        n_runs=("total_wall_time_s", "count"),
    ).reset_index()

    if not speedup_long.empty:
        speedup_summary = speedup_long.groupby(["threshold", "method"]).agg(
            speedup_mean=("paired_speedup", "mean"),
            speedup_ci95=("paired_speedup", _ci95),
        ).reset_index()
        summary = summary.merge(speedup_summary, on=["threshold", "method"], how="left")
        summary["speedup_mean"] = summary["speedup_mean"].fillna(1.0)
        summary["speedup_ci95"] = summary["speedup_ci95"].fillna(0.0)

    summary = summary.sort_values(["threshold", "method"])
    summary_csv = os.path.join(ROOT_OUT, "summary_table.csv")
    summary.to_csv(summary_csv, index=False)

    matrix_csv = None
    if not speedup_long.empty:
        matrix = speedup_long.pivot_table(index="method", columns="threshold",
                                          values="paired_speedup", aggfunc="mean")
        matrix_csv = os.path.join(ROOT_OUT, "speedup_matrix.csv")
        matrix.to_csv(matrix_csv)

    config_dict = dict(
        ROOT_OUT=ROOT_OUT, TARGET_PORT=TARGET_PORT, MAX_CAPS=MAX_CAPS,
        N_PARTICLES=N_PARTICLES, N_ITERATIONS=N_ITERATIONS,
        W_MAX=W_MAX, W_MIN=W_MIN, C1=C1, C2=C2,
        NUM_RUNS=NUM_RUNS, TARGETS=TARGETS, METHODS=METHODS,
        BASELINE_METHOD=BASELINE_METHOD, BASE_SEED=BASE_SEED,
        REAL_DATA_AVAILABLE=REAL_DATA_AVAILABLE,
        N_NODES=N_NODES, N_FREQS=N_FREQS, N_CAP_MODELS=N_CAP_MODELS,
        device=DEVICE.type, dtype=str(CDTYPE),
        gpu_name=(torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        load_time_s=LOAD_TIME_S, cuda_init_time_s=CUDA_INIT_TIME_S,
        h2d_transfer_time_s=TRANSFER_TIME_S, base_inverse_time_s=BASE_INV_TIME_S,
        warmup_time_s=WARMUP_TIME_S, init_time_total_s=INIT_TIME_S,
        gpu_cpu_verification=VERIFICATION,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(os.path.join(ROOT_OUT, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # ---- figures ----
    METHOD_STYLE = {
        "numpy":       dict(color="#1f77b4", marker="o", label="Full inverse (GPU)"),
        "solve":       dict(color="#2ca02c", marker="s", label="Solve (proposed, GPU)"),
        "sm":          dict(color="#ff7f0e", marker="^", label="Sherman-Morrison"),
        "iterative":   dict(color="#e83e8c", marker="D", label="Iterative"),
        "cpu_numpy":   dict(color="#7f3fbf", marker="P", label="NumPy (CPU)"),
        "pure_python": dict(color="#8c564b", marker="X", label="Pure-Python (CPU)"),
    }
    METHOD_ORDER = ["numpy", "solve", "sm", "iterative", "cpu_numpy", "pure_python"]

    def _label(m):
        return METHOD_STYLE.get(m, dict(label=m))["label"]

    def _finalize_ax(ax, legend_title=None):
        ax.tick_params(axis="both", which="major", direction="in",
                       top=True, right=True, length=6, width=1.2, labelsize=11)
        ax.tick_params(axis="both", which="minor", direction="in",
                       top=True, right=True, length=3, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(12)
        if ax.get_title():
            ax.title.set_fontweight("bold")
            ax.title.set_fontsize(13)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        leg = ax.get_legend()
        if leg is not None:
            if legend_title is not None:
                leg.set_title(legend_title)
            if leg.get_title() is not None:
                leg.get_title().set_fontweight("bold")
            for text in leg.get_texts():
                text.set_fontweight("bold")

    def _finalize_colorbar(cbar):
        cbar.ax.yaxis.label.set_fontweight("bold")
        cbar.ax.yaxis.label.set_fontsize(12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")

    figs_dir = os.path.join(ROOT_OUT, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    if matrix_csv is not None:
        mat = pd.read_csv(matrix_csv, index_col=0)
        rows = [m for m in METHOD_ORDER if m != BASELINE_METHOD and m in mat.index]
        mat = mat.reindex(rows)
        mat = mat[sorted(mat.columns, key=lambda c: -float(c))]
        thr = mat.columns.astype(float)

        fig, ax = plt.subplots(figsize=(7, 3.2 + 0.4 * len(mat)))
        im = ax.imshow(mat.values, cmap="Greens", aspect="auto")
        ax.set_xticks(range(len(thr)))
        ax.set_xticklabels([f"{t:.3f}" for t in thr])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([_label(m) for m in mat.index])
        ax.set_xlabel("Target Impedance Threshold (Ohm)")
        ax.set_ylabel("Inversion Method")
        ax.set_title(f"GPU Speedup Relative to {_label(BASELINE_METHOD)}\n(Higher is Better)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}x", ha="center", va="center",
                        fontweight="bold", fontsize=12,
                        color="white" if v > np.nanmax(mat.values) * 0.6 else "black")
        cbar = fig.colorbar(im, ax=ax, label=f"Speedup vs {_label(BASELINE_METHOD)}")
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(13)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        _finalize_colorbar(cbar)
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "speedup_heatmap.png"), dpi=300)
        plt.close(fig)

    s = summary.copy()
    if "speedup_mean" in s.columns:
        s["pct_reduction"] = 100.0 * (1.0 - 1.0 / s["speedup_mean"].replace(0, np.nan))
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, g in s.groupby("method"):
            if method == BASELINE_METHOD:
                continue
            g = g.sort_values("threshold", ascending=False)
            style = METHOD_STYLE.get(method, dict(color="gray", marker="x", label=method))
            ax.plot(g["threshold"], g["pct_reduction"], marker=style["marker"],
                    color=style["color"], label=style["label"], linewidth=2)
        ax.axhline(0, linestyle="--", color="tab:blue",
                   label=f"{_label(BASELINE_METHOD)} baseline (0%)")
        ax.invert_xaxis()
        ax.set_xlabel("Target Impedance Threshold (Ohm)")
        ax.set_ylabel(f"Wall-Clock Time Reduction vs {_label(BASELINE_METHOD)} (%)")
        ax.set_title(f"Runtime Reduction Relative to {_label(BASELINE_METHOD)}")
        ax.legend(title="Method")
        ax.grid(alpha=0.3)
        _finalize_ax(ax, legend_title="Method")
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "runtime_reduction_vs_threshold.png"), dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for method, g in summary.groupby("method"):
        g = g.sort_values("threshold", ascending=False)
        style = METHOD_STYLE.get(method, dict(color="gray", marker="x", label=method))
        ax.plot(g["threshold"], g["best_minz_mean"], marker=style["marker"],
                color=style["color"], label=style["label"], linewidth=2)
        ax.fill_between(g["threshold"],
                        g["best_minz_mean"] - g["best_minz_ci95"],
                        g["best_minz_mean"] + g["best_minz_ci95"],
                        color=style["color"], alpha=0.15)
    ax.invert_xaxis()
    ax.set_xlabel("Target Impedance Threshold (Ohm)")
    ax.set_ylabel("Average Best |Z11| (Ohm)")
    ax.set_title("Best Achieved Impedance vs. Target\n(Shaded Band = 95% CI, Lower is Better)")
    ax.legend(title="Method")
    ax.grid(alpha=0.3, linestyle=":")
    _finalize_ax(ax, legend_title="Method")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "best_minz_vs_threshold.png"), dpi=300)
    plt.close(fig)

    thresholds_sorted = sorted(summary["threshold"].unique(), reverse=True)
    methods_present = list(summary["method"].unique())
    ordered = [BASELINE_METHOD] + [m for m in METHOD_ORDER if m != BASELINE_METHOD]
    methods_for_bar = [m for m in ordered if m in methods_present]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(thresholds_sorted))
    width = 0.8 / max(len(methods_for_bar), 1)
    for i, method in enumerate(methods_for_bar):
        g = summary[summary["method"] == method].set_index("threshold").reindex(thresholds_sorted)
        style = METHOD_STYLE.get(method, dict(color="gray", label=method))
        label = style["label"] + (" [baseline]" if method == BASELINE_METHOD else "")
        ax.bar(x + i * width, g["time_s_mean"], width=width, yerr=g["time_s_ci95"],
               color=style["color"], label=label, capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(methods_for_bar) - 1) / 2)
    ax.set_xticklabels([f"{t:.3f}" for t in thresholds_sorted])
    ax.set_xlabel("Target Impedance Threshold (Ohm)")
    ax.set_ylabel("Mean Wall-Clock Time (s, log scale)")
    ax.set_title(f"Absolute Runtime by Method ({DEVICE.type.upper()})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    _finalize_ax(ax, legend_title="Method")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "absolute_runtime.png"), dpi=300)
    plt.close(fig)

    valid = df[df["success"] & df["time_to_target_s"].notna()]
    top10 = valid.sort_values("time_to_target_s").head(10)

    with open(os.path.join(ROOT_OUT, "top10_fastest.txt"), "w") as f:
        for i, row in enumerate(top10.itertuples(index=False), 1):
            f.write(f"{i}. {row._asdict()}\n")
            src = os.path.join(row.folder, "run_plot.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ROOT_OUT, f"top{i}.png"))

    with open(os.path.join(ROOT_OUT, "statistics.txt"), "w") as f:
        f.write(f"device={DEVICE.type}  dtype={CDTYPE}\n")
        f.write(f"init breakdown (s): load={LOAD_TIME_S:.4f} "
                f"cuda_ctx={CUDA_INIT_TIME_S:.4f} h2d={TRANSFER_TIME_S:.4f} "
                f"base_inv={BASE_INV_TIME_S:.4f} warmup={WARMUP_TIME_S:.4f}\n")
        f.write(f"gpu-vs-cpu verification: {json.dumps(VERIFICATION)}\n")
        for threshold in TARGETS:
            f.write(f"\n=== Threshold {threshold} ===\n")
            subset = valid[valid["threshold"] == threshold]
            if not subset.empty:
                best = subset.loc[subset["time_to_target_s"].idxmin()]
                f.write(f"BEST: {best.to_dict()}\n")

    # ---- FIX 7/8: timeouts made explicit, not just filterable ----
    timeouts = df[~df["success"]]
    with open(os.path.join(ROOT_OUT, "timeouts.txt"), "w") as f:
        if timeouts.empty:
            f.write("No timeouts -- every run reached its threshold within MAX_CAPS.\n")
        else:
            for (method, threshold), g in timeouts.groupby(["method", "threshold"]):
                f.write(f"method={method}  threshold={threshold}  "
                        f"timeouts={len(g)}/{NUM_RUNS}\n")
                for row in g.itertuples(index=False):
                    f.write(f"    {row.folder}  best_minz={row.best_minz_ohm:.6f}\n")

    print(f"\nStructured outputs written under: {ROOT_OUT}")
    print(f"  {results_long_csv}")
    print(f"  {summary_csv}")
    if matrix_csv:
        print(f"  {matrix_csv}")
    print(f"  {figs_dir}/*.png")

print("Done.")