"""
Shared PSO core for the SP_Sep2026 PDN boards.

Skeleton is GPUSwarm/CPUTest/SB.py -- its FIX 1-13 semantics are kept verbatim:
  FIX 1   reseed per (run, n_caps), independent of threshold and of how many
          RNG draws earlier stages consumed
  FIX 2   floor(frac*K) discretisation, clipped -- never floor(frac*(K-1))
  FIX 3   observation ports are never decap sites, enforced in BOTH the decode
          and the collision resolver
  FIX 4   warm start: the previous stage's global best is carried forward
  FIX 5   the dense matrix is assembled only for methods that consume it
  FIX 6   unknown methods raise; only genuine singularity is scored invalid
  FIX 10  time.perf_counter, not time.time
  FIX 11  warm block gets clipped Gaussian jitter + a random remainder, so the
          carried dimensions are not frozen
  FIX 12  the collision resolver tries every offset before giving up
  FIX 13  prev_best resets to None when a stage produced no usable particle

Ported over from DecapStudy:
  * Problem: ports come from the loaded matrix. n_obs observation ports sit
    first and never take a capacitor; pads are the rest; max_caps = N - n_obs.
    Supports several observation ports (y_sp_mphy_full has five).
  * multi-band objective with re-scoring of every winning placement on ALL
    bands, so a result found on one band is comparable on the others
  * threaded fitness evaluation (STUDY_THREADS), with BLAS pinned to 1
  * per-run wall-clock limit, flagged as timed_out rather than silently cut
  * one JSON record per run, carrying the full placement (model + pad)

Data: Data/{MPHY,VDDQ_DDR3}/y_sp_*.mat + Data/Decaps/combined_decaps.mat, all on
one 2588-point grid (110.89 Hz - 997.21 MHz). The board freq_sp.mat files are
byte-identical to Decaps/sp_freq.mat, so nothing is resampled here.
"""
import os
import sys
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

THREADS = int(os.environ.get("STUDY_THREADS", "1"))
if THREADS > 1:                                   # pin BLAS before numpy loads
    for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG  (SB.py values, unchanged)
# ============================================================
N_PARTICLES = 100      # was 50 in SB.py; raised to search harder for the floor
N_ITERATIONS = 40      # was 15
W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5
N_WARM_START = 20
WARM_START_JITTER = 0.02
BASE_SEED = 12345
NUM_RUNS = int(os.environ.get("NUM_RUNS", "10"))
TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", "1800"))
# Only numpy is active. The others are kept for the timing study and are NOT
# run here: they compute the same Z (verified to ~1e-16 relative), so they add
# 5x the compute for identical placements. "iterative" is an approximation and
# would not even give identical placements.
#     "solve"        same math, batched triangular solve
#     "sm"           rank-1 Sherman-Morrison, exact
#     "iterative"    2nd-order Neumann series, APPROXIMATION
#     "pure_python"  Gauss-Jordan, no numpy -- O(n^3) interpreted, hours per
#                    evaluation at these sizes. Never enable it on the large nets.
METHODS = os.environ.get("METHODS", "numpy").split(",")

# Objective bands, in Hz. None = the whole grid. Every winning placement is
# re-scored on all of them; RANK_BAND is the one reported as the headline.
BANDS = {"B50": 50e6, "B200": 200e6, "BFULL": None}
RANK_BAND = os.environ.get("RANK_BAND", "B50")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PDN_DATA", os.path.join(os.path.dirname(HERE), "Data"))
EXTRAP_START_HZ = 99.83e6    # decap models are measured below this, extrapolated above

# ============================================================
# DATA
# ============================================================
_lib = {}

def library():
    """decaps (n_models, F) complex admittance, plus its metadata."""
    if not _lib:
        m = sio.loadmat(os.path.join(DATA, "Decaps", "combined_decaps.mat"))
        _lib["d"] = m["decaps"]
        _lib["names"] = [str(x[0][0]) for x in m["names"]]
        _lib["slope"] = m["loglog_slope"].ravel()
        _lib["freq"] = sio.loadmat(os.path.join(DATA, "Decaps", "sp_freq.mat"))["freq"].ravel()
        _lib["lists"] = None
    return _lib


def band_mask(name):
    f = library()["freq"]
    top = BANDS[name]
    return np.ones(len(f), bool) if top is None else (f <= top)


class Problem:
    """One PDN to optimise.

    y is (N, N, F) on disk and transposed to (F, N, N) here. The first n_obs
    indices are observation ports and never take a capacitor; pads are every
    remaining index, so max_caps = N - n_obs -- all read from the matrix.
    """

    def __init__(self, name, path, n_obs=1, obs_labels=None):
        y = sio.loadmat(path)["y"]
        self.y = np.transpose(y, (2, 0, 1)).astype(np.complex128)
        self.name = name
        self.F, self.N, _ = self.y.shape
        self.n_obs = int(n_obs)
        assert 1 <= self.n_obs < self.N, f"{name}: n_obs={n_obs} with N={self.N}"
        self.obs = np.arange(self.n_obs)
        self.pads = np.arange(self.n_obs, self.N)
        self.max_caps = len(self.pads)
        self.obs_labels = obs_labels or [f"{name}_obs{k+1}" for k in range(self.n_obs)]
        lib = library()
        assert self.F == lib["d"].shape[1] == len(lib["freq"]), (
            f"{name}: F={self.F} but library has {lib['d'].shape[1]} and the grid "
            f"has {len(lib['freq'])} -- these must be one shared axis")
        self._yinv = None
        self._lists = None

    @property
    def y_inv_base(self):                         # only sm / iterative need it
        if self._yinv is None:
            self._yinv = np.linalg.inv(self.y)
        return self._yinv

    @property
    def lists(self):                              # only pure_python needs it
        if self._lists is None:
            self._lists = [self.y[f].tolist() for f in range(self.F)]
        return self._lists


# ============================================================
# DISCRETISATION AND PORT RULES  (FIX 2, FIX 3, FIX 12)
# ============================================================

def frac_to_index(frac, K):
    """floor(frac*K) clipped to [0, K-1]. Not floor(frac*(K-1)), which can
    never reach the last index and silently shrinks the search space."""
    return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)


def decode_particle(vec, n_caps, prob):
    """[0,1]^(2n) -> (models, pads). Layout [models..., ports...] as in SB.py.
    Ports are indices INTO prob.pads, so an observation port can never appear."""
    models = frac_to_index(vec[:n_caps], library()["d"].shape[0])
    ports = prob.pads[frac_to_index(vec[n_caps:], len(prob.pads))]
    return models, ports


def resolve_adjacent_ports(models, ports, prob):
    """Push colliding capacitors onto the nearest free pad.

    Every observation port starts in `used`, so none can ever be assigned
    (FIX 3). Every offset is tried before the collision is accepted as
    unavoidable (FIX 12) -- the original gave up after offset 1 and returned
    duplicates.
    """
    used = set(int(o) for o in prob.obs)
    padset = set(int(p) for p in prob.pads)
    ports = list(int(p) for p in ports)

    for i in range(len(ports)):
        if ports[i] not in used:
            used.add(ports[i])
            continue
        placed = False
        for offset in range(1, prob.N):
            for cand in (ports[i] + offset, ports[i] - offset):
                if cand in padset and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    placed = True
                    break
            if placed:
                break
        if not placed:                 # n_caps > pad count: genuinely unavoidable
            used.add(ports[i])
    return models, ports


# ============================================================
# INVERSION METHODS  (FIX 5, FIX 6)
# ============================================================

def _diag_obs(Z, prob):
    return np.abs(np.einsum("fkk->fk", Z[:, :prob.n_obs, :prob.n_obs]))


def inv_numpy(A, prob):
    return _diag_obs(np.linalg.inv(A), prob)


def inv_solve(A, prob):
    E = np.zeros((prob.F, prob.N, prob.n_obs))
    for k in range(prob.n_obs):
        E[:, k, k] = 1.0
    X = np.linalg.solve(A, E.astype(np.complex128))
    return np.abs(np.stack([X[:, k, k] for k in range(prob.n_obs)], axis=1))


def inv_sm(prob, config):
    """Rank-1 Sherman-Morrison, one capacitor at a time. Exact, not a series."""
    d = library()["d"]
    Z = prob.y_inv_base.copy()
    for cap, port in config:
        v = d[cap]                                   # (F,)
        col = Z[:, :, port]                          # Z u
        row = Z[:, port, :]                          # v^T Z   (times v)
        denom = 1.0 + v * Z[:, port, port]
        if np.any(np.abs(denom) < 1e-12):
            raise np.linalg.LinAlgError("Sherman-Morrison denominator ~ 0")
        Z = Z - (v / denom)[:, None, None] * col[:, :, None] * row[:, None, :]
    return _diag_obs(Z, prob)


def inv_iterative(A, prob):
    """Second-order Neumann correction, with a full inverse fallback wherever
    the series is outside its validity radius. An APPROXIMATION: its cost can
    differ from numpy by double digits, so it is a timing reference only."""
    B = prob.y_inv_base
    I = np.eye(prob.N)
    E = A @ B - I
    nrm = np.abs(E).sum(axis=1).max(axis=1)          # 1-norm per frequency
    Z = B @ (I - E + E @ E)
    bad = nrm >= 1.0
    if bad.any():
        Z[bad] = np.linalg.inv(A[bad])
    return _diag_obs(Z, prob)


def _gauss_jordan_inverse(A):
    """Full inverse by Gauss-Jordan with partial pivoting. Plain Python lists
    and the built-in complex type -- no numpy anywhere on this path."""
    n = len(A)
    M = [row[:] + [1.0 + 0.0j if i == r else 0.0 + 0.0j for i in range(n)]
         for r, row in enumerate(A)]
    for col in range(n):
        piv_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv_row][col]) < 1e-300:
            raise ZeroDivisionError("singular matrix")
        if piv_row != col:
            M[col], M[piv_row] = M[piv_row], M[col]
        inv_p = 1.0 / M[col][col]
        rc = M[col]
        for k in range(2 * n):
            rc[k] *= inv_p
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            if fac == 0:
                continue
            rr = M[r]
            for k in range(2 * n):
                rr[k] -= fac * rc[k]
    return [row[n:] for row in M]


def inv_pure_python(prob, config, f):
    d_lists = library().setdefault("lists_d", library()["d"].tolist())
    Yeq = [row[:] for row in prob.lists[f]]
    for cap, port in config:
        Yeq[port][port] += d_lists[cap][f]
    inv = _gauss_jordan_inverse(Yeq)
    return [abs(inv[k][k]) for k in range(prob.n_obs)]


# ============================================================
# FITNESS
# ============================================================

def peaks_all_bands(prob, config, method):
    """Peak |Z_kk| per observation port, per band, for one placement.
    Returns {band: (n_obs,) array} in ohm."""
    d = library()["d"]
    if method not in ("numpy", "solve", "sm", "iterative", "pure_python"):
        raise ValueError(f"Unknown method: {method!r}")
    try:
        if method == "sm":
            Zd = inv_sm(prob, config)
        elif method == "pure_python":
            Zd = np.array([inv_pure_python(prob, config, f) for f in range(prob.F)])
        else:
            A = prob.y.copy()                                       # FIX 5
            for cap, port in config:
                A[:, port, port] += d[cap]
            Zd = (inv_numpy(A, prob) if method == "numpy" else
                  inv_solve(A, prob) if method == "solve" else
                  inv_iterative(A, prob))
    except (np.linalg.LinAlgError, ZeroDivisionError):              # FIX 6
        return {b: np.full(prob.n_obs, 1e200) for b in BANDS}
    return {b: Zd[band_mask(b)].max(axis=0) for b in BANDS}


def cost_of(peaks, band):
    """Scalar objective: worst observation port's peak |Z11| in ohm."""
    return float(np.max(peaks[band]))


def bare_peaks(prob, method="numpy"):
    return peaks_all_bands(prob, [], method)


# ============================================================
# WARM-START SEEDING  (FIX 4, FIX 11)
# ============================================================

def seed_particles(n_caps, prev_best):
    n_warm = min(N_WARM_START, N_PARTICLES)
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(N_PARTICLES, 2 * n_caps)

    pn = n_caps - 1
    warm = np.hstack([
        np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
        np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1),
    ])
    if WARM_START_JITTER > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, bool)
        carried[pn] = False            # the new model gene is already uniform
        carried[-1] = False            # the new port gene likewise
        noise = np.random.normal(0.0, WARM_START_JITTER, warm.shape)
        noise[0, :] = 0.0              # particle 0 is elite, untouched
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    n_rand = N_PARTICLES - n_warm
    if n_rand > 0:
        warm = np.vstack([warm, np.random.rand(n_rand, 2 * n_caps)])
    return warm


# ============================================================
# LOGGING AND PLOTS
# ============================================================

def setup_logger(path):
    lg = logging.getLogger(str(path))
    lg.setLevel(logging.INFO)
    if lg.hasHandlers():
        lg.handlers.clear()
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s INFO: %(message)s"))
    lg.addHandler(h)
    return lg


def plot_run_curve(caps, minz, prob, method, run_id, out, threshold=None):
    if not caps:
        return
    xs, ys = np.array(caps, int), np.array(minz, float)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o", lw=1.5, label="peak |Z11| per n_caps")
    plt.plot(xs, np.minimum.accumulate(ys), marker="s", ls="--", lw=1.5,
             label="best-so-far envelope")
    if threshold:
        plt.axhline(threshold, ls="--", lw=1.0, label=f"target={threshold:g}")
    plt.yscale("log")
    plt.xlabel("Decaps"); plt.ylabel(r"Peak $|Z_{11}|$ ($\Omega$)")
    plt.title(f"{prob.name} - {method} - run {run_id} ({RANK_BAND})")
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "run_plot.png"), dpi=200); plt.close()


def plot_convergence(hist, prob, out, threshold=None):
    if not hist:
        return
    plt.figure(figsize=(9, 5))
    for n, h in hist.items():
        if h:
            plt.plot(h, label=f"n={n}")
    if threshold:
        plt.axhline(threshold, ls="--", lw=1.0)
    plt.yscale("log")
    plt.xlabel("Iteration"); plt.ylabel(r"Peak $|Z_{11}|$ ($\Omega$)")
    plt.title(f"{prob.name} - global convergence")
    plt.grid(True, which="both", alpha=.3); plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "global_convergence.png"), dpi=200); plt.close()


def plot_impedance(prob, config, out, method="numpy"):
    """Bare vs optimised |Z11| at the first observation port, log-log."""
    d, f = library()["d"], library()["freq"]
    A = prob.y.copy()
    for cap, port in config:
        A[:, port, port] += d[cap]
    zb = np.abs(np.linalg.inv(prob.y)[:, 0, 0])
    zo = np.abs(np.linalg.inv(A)[:, 0, 0])
    plt.figure(figsize=(9, 5))
    plt.loglog(f, zb, lw=1.6, label="bare")
    plt.loglog(f, zo, lw=1.6, label=f"optimised ({len(config)} decaps)")
    plt.axvspan(EXTRAP_START_HZ, f[-1], color="0.5", alpha=.13, lw=0)
    for b, top in BANDS.items():
        if top:
            plt.axvline(top, color="crimson", ls=":", lw=1.0)
    plt.xlabel("Frequency (Hz)"); plt.ylabel(r"$|Z_{11}|$ ($\Omega$)")
    plt.title(f"{prob.name} - bare vs optimised (shaded: extrapolated decap data)")
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "impedance.png"), dpi=200); plt.close()


_jsonl_lock = threading.Lock()

def append_record(path, rec):
    with _jsonl_lock, open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def placement_records(config):
    names, slope = library()["names"], library()["slope"]
    return [{"model": int(c), "pad": int(p), "name": names[c],
             "loglog_slope": float(slope[c])} for c, p in config]


# ============================================================
# PSO
# ============================================================

def run_pso(prob, method, run_id, out_folder, results_path,
            threshold=None, max_caps=None, pool=None, out_root=None):
    """One PSO run: capacitor counts 1..max_caps, warm-started stage to stage.

    With threshold=None the full budget is searched and the whole
    quality-versus-capacitor-count curve is recorded. With a threshold in ohm
    the run stops the instant the running best meets it, and time_to_target is
    recorded -- that is the mode to use when timing inversion methods.
    """
    os.makedirs(out_folder, exist_ok=True)
    log = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))
    cap_ceiling = min(max_caps or prob.max_caps, prob.max_caps)
    # Effective warm count. If it equals the swarm, there is no random
    # remainder and the carried dimensions cannot move (the failure FIX 11
    # exists to prevent), so refuse rather than run a degenerate swarm.
    n_warm_eff = min(N_WARM_START, N_PARTICLES)
    if N_PARTICLES > 1 and n_warm_eff >= N_PARTICLES:
        raise ValueError(
            f"N_WARM_START={N_WARM_START} >= N_PARTICLES={N_PARTICLES}: the whole "
            f"swarm would be warm with no random remainder. Lower N_WARM_START.")

    log.info("RUN_START | prob=%s | N=%d n_obs=%d pads=%d | method=%s | run=%d | "
             "particles=%d iters=%d seed=%d warm=%d/%d jitter=%.4f | band=%s | "
             "threshold=%s | max_caps=%d | threads=%d",
             prob.name, prob.N, prob.n_obs, len(prob.pads), method, run_id,
             N_PARTICLES, N_ITERATIONS, BASE_SEED, n_warm_eff, N_PARTICLES,
             WARM_START_JITTER, RANK_BAND, threshold, cap_ceiling, THREADS)

    bare = cost_of(bare_peaks(prob, method), RANK_BAND)
    caps_curve, minz_curve, histories = [], [], {}
    best = {"cost": np.inf, "config": [], "n_caps": 0, "peaks": None}
    t0 = time.perf_counter()                                       # FIX 10
    time_to_target = caps_at_target = None
    n_evals = 0
    timed_out = False
    prev_best = None

    for n_caps in range(1, cap_ceiling + 1):
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)        # FIX 1
        particles = seed_particles(n_caps, prev_best)              # FIX 4 / FIX 11
        assert particles.shape == (N_PARTICLES, 2 * n_caps)
        velocities = np.zeros_like(particles)
        pbest, pbest_val = particles.copy(), np.full(N_PARTICLES, np.inf)
        gbest, gbest_particle, history = np.inf, None, []
        stop_flag = False
        it = 0

        for it in range(N_ITERATIONS):
            configs = []
            for i in range(N_PARTICLES):
                m, p = decode_particle(particles[i], n_caps, prob)   # FIX 2, FIX 3
                m, p = resolve_adjacent_ports(m, p, prob)            # FIX 12
                configs.append([(int(a), int(b)) for a, b in zip(m, p)])

            # Threaded mode evaluates the whole iteration at once; results are
            # consumed in particle order, so the trajectory and the immediate-
            # stop point are identical to serial.
            allp = list(pool.map(lambda c: peaks_all_bands(prob, c, method), configs)) \
                if pool else None

            for i in range(N_PARTICLES):
                pk = allp[i] if pool else peaks_all_bands(prob, configs[i], method)
                cost = cost_of(pk, RANK_BAND)
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i], pbest[i] = cost, particles[i]
                if cost < gbest:
                    gbest, gbest_particle = cost, particles[i].copy()
                if cost < best["cost"]:
                    best = {"cost": cost, "config": configs[i],
                            "n_caps": n_caps, "peaks": pk}
                if threshold is not None and gbest <= threshold:    # immediate stop
                    stop_flag = True
                    break

            history.append(gbest)
            if stop_flag:
                if time_to_target is None:
                    time_to_target = time.perf_counter() - t0
                    caps_at_target = n_caps
                break
            if time.perf_counter() - t0 > TIME_LIMIT_S:
                timed_out = True
                break

            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)
            velocities = (w * velocities
                          + C1 * np.random.rand(*particles.shape) * (pbest - particles)
                          + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles))
            particles = np.clip(particles + velocities, 0.0, 1.0)

        histories[n_caps] = history
        if gbest_particle is None:
            prev_best = None                                        # FIX 13
            log.info("n_caps=%d | iter=%d | minZ=inf | placement={}", n_caps, it + 1)
            continue
        prev_best = gbest_particle                                  # FIX 4
        m, p = decode_particle(gbest_particle, n_caps, prob)
        m, p = resolve_adjacent_ports(m, p, prob)
        caps_curve.append(n_caps)
        minz_curve.append(gbest)
        log.info("n_caps=%d | iter=%d | minZ=%.6g | placement=%s",
                 n_caps, it + 1, gbest, {int(b): int(a) for a, b in zip(m, p)})
        if stop_flag or timed_out:
            break

    wall = time.perf_counter() - t0
    # Re-score the winning placement on EVERY band, so a result found on one
    # band is directly comparable on the others.
    pk = best["peaks"] if best["peaks"] is not None else bare_peaks(prob, method)
    slopes = [library()["slope"][c] for c, _ in best["config"]]
    rec = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problem": prob.name, "N": prob.N, "n_obs": prob.n_obs,
        "n_pads": len(prob.pads), "max_caps": cap_ceiling,
        "method": method, "run_id": run_id, "seed_base": BASE_SEED,
        "rank_band": RANK_BAND, "threshold_ohm": threshold,
        "bare_peak_ohm": {b: float(np.max(bare_peaks(prob, method)[b])) for b in BANDS},
        "best_peak_ohm": {b: float(np.max(pk[b])) for b in BANDS},
        "best_peak_per_obs_ohm": {b: [float(x) for x in pk[b]] for b in BANDS},
        "best_n_caps": best["n_caps"],
        "placement": placement_records(best["config"]),
        "frac_caps_slope_outside_0p8_1p2":
            float(np.mean([(s < 0.8) or (s > 1.2) for s in slopes])) if slopes else 0.0,
        "curve_caps": caps_curve, "curve_peak_ohm": minz_curve,
        "time_to_target_s": time_to_target, "caps_at_target": caps_at_target,
        "success": time_to_target is not None, "timed_out": timed_out,
        "total_wall_time_s": wall, "n_fitness_evals": n_evals,
        "threads": THREADS, "n_warm_effective": n_warm_eff,
        "folder": os.path.relpath(out_folder, out_root) if out_root else out_folder,
    }
    log.info("RESULT | best=%.6g ohm @ %d caps | wall=%.1fs | evals=%d | timed_out=%s",
             best["cost"], best["n_caps"], wall, n_evals, timed_out)
    append_record(results_path, rec)

    plot_run_curve(caps_curve, minz_curve, prob, method, run_id, out_folder, threshold)
    plot_convergence(histories, prob, out_folder, threshold)
    if best["config"]:
        plot_impedance(prob, best["config"], out_folder)
    return rec


# ============================================================
# SELF-TESTS
# ============================================================

def selftest(prob, methods):
    """Port rules, then method agreement against numpy on identical placements."""
    rng = np.random.default_rng(7)
    for n_caps in (1, max(1, len(prob.pads) // 2), len(prob.pads)):
        v = rng.random(2 * n_caps)
        m, p = decode_particle(v, n_caps, prob)
        m, p = resolve_adjacent_ports(m, p, prob)
        assert all(q in set(prob.pads.tolist()) for q in p), \
            f"{prob.name}: decap landed outside the pads"
        assert not (set(p) & set(int(o) for o in prob.obs)), \
            f"{prob.name}: decap landed on an observation port"
        if n_caps <= len(prob.pads):
            assert len(set(p)) == n_caps, f"{prob.name}: duplicate pads at n_caps={n_caps}"
    ref = None
    cfg = [(int(rng.integers(0, library()["d"].shape[0])), int(q))
           for q in prob.pads[:min(4, len(prob.pads))]]
    for meth in methods:
        c = cost_of(peaks_all_bands(prob, cfg, meth), RANK_BAND)
        if ref is None:
            ref = c
            print(f"  [selftest] {prob.name}: numpy reference {ref:.6g} ohm")
        else:
            rel = abs(c - ref) / ref
            tag = "APPROXIMATION" if meth == "iterative" else ("ok" if rel < 1e-9 else "MISMATCH")
            print(f"  [selftest] {prob.name}: {meth:12s} rel diff {rel:.3g}  {tag}")
            if meth != "iterative":
                assert rel < 1e-9, f"{meth} disagrees with numpy by {rel:.3g}"
    print(f"  [selftest] {prob.name}: N={prob.N} n_obs={prob.n_obs} "
          f"pads={len(prob.pads)} max_caps={prob.max_caps}  OK")


# ============================================================
# DRIVER
# ============================================================

def main(board, problems, out_root, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    validate = "--validate" in argv
    os.makedirs(out_root, exist_ok=True)
    results = os.path.join(out_root, "results_long.jsonl")

    lib = library()
    print(f"{board}: library {lib['d'].shape[0]} models x {lib['d'].shape[1]} freqs, "
          f"{lib['freq'][0]:.4g}-{lib['freq'][-1]:.4g} Hz")
    print(f"  extrapolated above {EXTRAP_START_HZ/1e6:.2f} MHz = "
          f"{100*(lib['freq'] > EXTRAP_START_HZ).mean():.1f}% of grid points")
    print(f"  methods={METHODS}  rank_band={RANK_BAND}  runs={NUM_RUNS}  threads={THREADS}")

    probs = [Problem(**kw) for kw in problems]
    for p in probs:
        selftest(p, METHODS)
        bp = bare_peaks(p, "numpy")
        print(f"  {p.name:14s} N={p.N:3d} pads={len(p.pads):3d}  bare peak "
              + "  ".join(f"{b}={np.max(bp[b]):.4g}" for b in BANDS))
    if validate:
        print("--validate: data, port rules and methods check out; no PSO run.")
        return

    pool = ThreadPoolExecutor(THREADS) if THREADS > 1 else None
    try:
        for p in probs:
            for meth in METHODS:
                for run in range(1, NUM_RUNS + 1):
                    folder = os.path.join(out_root, p.name, meth, f"run_{run}")
                    print(f"[{time.strftime('%H:%M:%S')}] {p.name} {meth} run {run}/{NUM_RUNS}",
                          flush=True)
                    try:
                        r = run_pso(p, meth, run, folder, results, pool=pool,
                                    out_root=out_root)
                        print(f"    best {r['best_peak_ohm'][RANK_BAND]:.6g} ohm "
                              f"@ {r['best_n_caps']} caps, {r['total_wall_time_s']:.1f}s",
                              flush=True)
                    except Exception as exc:
                        logging.getLogger("main").error(
                            "RUN_FAILED | %s %s run %s | %s", p.name, meth, run, exc)
                        print(f"    [ERROR] {exc}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    print(f"\nDone. Records appended to {results}")
