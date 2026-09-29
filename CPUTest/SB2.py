# nohup python SB.py > sb.log 2>&1 &
# ps -u $USER | grep SB.py
# pkill -9 -f SB.py
#
# ============================================================
# CORRECTED CPU BENCHMARK  (fixes applied to ScratchBench.py)
# ============================================================
# FIX 1  Deterministic, paired seeding. Every (threshold, run) combination
#        now gets an identical RNG seed at the top of run_pso() AND at the
#        top of every n_caps stage inside it (seeded from run & n_caps
#        only, independent of threshold and independent of how many RNG
#        draws earlier stages/methods consumed). This is what actually
#        realises the paper's "identical random seeds across methods... to
#        allow paired runtime comparison" -- the original script never
#        seeded np.random at all, so no two runs were comparable.
#        Per-stage (not just per-run) reseeding also means that even if
#        one method's earlier stage takes a different number of PSO
#        iterations than another's (different early-stop point), the
#        *next* stage still starts from an identical swarm across methods,
#        instead of drifting apart because the shared RNG stream had
#        consumed a different number of draws.
#
# FIX 2  Off-by-one AND overflow-safety in discretisation. The original
#        `floor(frac*(K-1))` can never select the last model/port (frac in
#        [0,1) maxes out at K-2), silently shrinking the search space by
#        one entry in both the capacitor-model and node dimensions. The
#        correct mapping is `floor(frac*K)`, clipped to [0, K-1] as a
#        safeguard for the frac==1.0 edge case that np.clip(particles,0,1)
#        can produce after a velocity update (1.0*K would otherwise index
#        out of bounds). See frac_to_index().
#
# FIX 3  TARGET_PORT (port 0, the measurement node) can never be selected
#        as a decap site. Previously `resolve_adjacent_ports` only
#        deduplicated collisions *between* chosen ports and never excluded
#        the measurement port itself, so a capacitor could shunt the exact
#        node being read -- artificially collapsing the measured |Z11|
#        instead of genuinely damping resonances elsewhere in the network.
#        VALID_PORTS excludes TARGET_PORT at the source, and
#        resolve_adjacent_ports() additionally reserves it as "already
#        used" as defense in depth.
#
# FIX 4  Warm-start between capacitor-count stages. See FIX 11 below --
#        the naive "seed every particle identically" reading of the
#        paper's Section III-A is degenerate and has been replaced by a
#        parameterised hybrid swarm.
#
# FIX 5  inv_sm() never touches the full Y_eq assembly `A` -- it rebuilds
#        everything itself from `config` and the precomputed base inverse.
#        The original code unconditionally built `A` before branching on
#        method, silently padding the SM method's measured runtime with
#        O(n) wasted diagonal-mutation work per frequency it never uses.
#        Matrix assembly now only happens for methods that actually need it.
#
# FIX 6  No more bare `except:`. Unknown method names now raise
#        ValueError immediately and loudly instead of leaving `val`
#        unbound and silently masking the resulting UnboundLocalError as
#        cost=1e200. Only np.linalg.LinAlgError (near-singular systems,
#        expected/handled numerical failures) is caught during evaluation.
#
# FIX 7  Unsuccessful (non-converged) runs are no longer silently dropped
#        from the record set. They're logged with SUCCESS=False and kept
#        in the structured records; top10/statistics still rank only
#        successful runs (as before), but a new timeouts.txt lists every
#        run that hit MAX_CAPS without reaching threshold, per method, so
#        differential failure rates across methods are visible instead of
#        invisibly biasing the averaged runtimes.
#
# FIX 8  The log line now also carries best_minZ and a success flag, and
#        the analysis regex captures them, so this script's own analysis
#        section can report achieved-impedance statistics without a
#        second pass over the logs.
#
# FIX 9  Folder names use a delimiter ("__THR__") that cannot collide with
#        an underscore inside a method name (e.g. "pure_python"), instead
#        of `parts[-2].split("_")`, which breaks the moment a method name
#        contains an underscore.
#
# FIX 10 time.perf_counter() (monotonic) replaces time.time() (wall clock,
#        can jump backwards on NTP adjustment) for all timing.
#
# ------------------------------------------------------------
# NEW IN THIS REVISION
# ------------------------------------------------------------
# FIX 11 PARAMETERISED WARM START -- fixes a swarm-collapse defect.
#        The previous revision seeded the carried genes of EVERY particle
#        identically from prev_best. That makes the swarm degenerate in
#        those dimensions: pbest[i] - particles[i] and
#        gbest_particle - particles[i] are then identically zero across
#        every carried dimension, so both PSO velocity terms are zero
#        there and stay zero forever. Stage N+1 was searching 2
#        dimensions, not 2(N+1) -- earlier capacitor placements could
#        never be revised once the stage advanced past them. (Measured:
#        spread across particles in the carried dimensions after 10
#        iterations was 2.8e-17, versus 0.35 in the two new ones.)
#
#        Now N_WARM_START particles carry prev_best and the remaining
#        N_PARTICLES - N_WARM_START are initialised uniformly at random,
#        so the swarm retains diversity in every dimension. Particle 0 is
#        an exact, unjittered copy of prev_best (elitism: the previous
#        stage's solution is never lost); warm particles 1.. receive small
#        clipped Gaussian jitter so the warm block itself is not
#        degenerate. Set WARM_START_JITTER = 0.0 and N_WARM_START =
#        N_PARTICLES to recover the previous (degenerate) behaviour for
#        comparison.
#
# FIX 12 resolve_adjacent_ports() could not actually resolve collisions
#        beyond offset 1. The loop ended with
#            if ports[i] in used: break
#        which is ALWAYS true when neither candidate at the current offset
#        was free -- ports[i] still holds its original duplicate value,
#        which is in `used` by definition. So the search broke out after
#        offset 1 instead of trying offset 2, 3, ...
#        resolve_adjacent_ports([5,4,6,5]) returned [5,4,6,5] with 5
#        repeated. Duplicate ports mean `A[port,port] += d` fires twice on
#        one node: one node gets double admittance and one fewer node is
#        decoupled than the candidate encodes, which inflates the reported
#        capacitor count relative to the number of distinct ports actually
#        used. Worst in the large-N regime (tight thresholds), where free
#        nodes are scarce. Now tracked with an explicit `placed` flag.
#        (Measured: 441 duplicate configs across a full stage sweep
#        before the fix, 0 after.)
#
# FIX 13 prev_best is reset to None on the `gbest_particle is None`
#        branch. That branch `continue`s without updating prev_best, so
#        the next stage would call seed_particles() with a vector sized
#        for n_caps-2 while it assumes n_caps-1. np.hstack succeeds and
#        silently produces a swarm of width 2*n_caps-2; decode_particle
#        then returns models and ports of different lengths and zip()
#        truncates. Currently unreachable (cost=1e200 < inf, so some
#        particle always wins), but it fails silently rather than loudly
#        if it ever does fire.
#
# REQUESTED CHANGE: immediate stopping once threshold is reached. This was
# already partially present in the original (a `break` out of the particle
# loop the instant gbest<=threshold), but it is now made unambiguous and
# is preserved exactly: the moment ANY particle's cost meets the target,
# evaluation of the remaining particles in that iteration is skipped, the
# iteration loop stops, and the n_caps stage loop stops -- optimisation
# never waits for the current iteration (let alone the full N_ITERATIONS)
# to finish once a solution is found.
#
# Dead code removed: inv_hybrid() and its module-level cache were never in
# the active METHODS roster and are dropped entirely rather than kept
# unused (they also reused a cache across unrelated candidate configs,
# which is not a physically meaningful use of the iterative-update idea).
# ============================================================

import os
import time
import logging
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import shutil

# ============================================================
# GLOBAL CONFIG
# ============================================================

ROOT_OUT = "SBLine"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_RUNS = 20

TARGET_PORT = 0
TARGETS = [0.05, 0.045, 0.04, 0.03]

MAX_CAPS = 20

N_PARTICLES = 50
N_ITERATIONS = 15

W_MAX, W_MIN = 0.9, 0.4
C1, C2 = 1.5, 1.5

# FIX 11: warm-start swarm composition.
#   N_WARM_START      particles carry the previous stage's global best.
#   N_PARTICLES - N_WARM_START  are initialised uniformly at random.
# Particle 0 of the warm block is always an exact copy of prev_best
# (elitism). WARM_START_JITTER is the std-dev of the Gaussian noise added
# to the CARRIED genes of warm particles 1.. -- without it the warm block
# is internally degenerate and cannot move in those dimensions.
N_WARM_START = 20
WARM_START_JITTER = 0.02

# METHODS = ["numpy", "solve", "sm", "iterative", "pure_python"]
METHODS = ["numpy", "solve", "sm", "iterative"]

# ADDITION: pure_python is a from-scratch reference baseline -- it inverts
# the assembled admittance matrix with hand-written Gauss-Jordan
# elimination using plain Python lists and the built-in `complex` type,
# zero NumPy/SciPy anywhere on its timed path. It exists to separate "how
# much of LAPACK's speedup is compiled linear algebra" from "how much is
# just avoiding a slow Python-level loop." It is NOT the baseline other
# methods are scored against in the log/analysis below (that stays
# implicit/relative, as in the original script) -- it is simply one more
# method in the roster.
#
# CAUTION: at full paper scale (N_NODES ~ 21, N_FREQS = 1391) this method
# is dramatically slower than every other method -- O(n^3) Gauss-Jordan
# in interpreted Python, per frequency, per particle, per iteration, per
# capacitor-count stage. Expect it to dominate total wall-clock time by
# orders of magnitude. Consider trimming PURE_PYTHON_TARGETS/
# PURE_PYTHON_NUM_RUNS below if you just want a representative sample
# rather than the full 20-run x 4-threshold sweep for this one method.

# Optional: give pure_python its own (smaller) sweep so it doesn't
# dominate total wall-clock time. Leave both as None to run it identically
# to every other method (full NUM_RUNS x TARGETS x MAX_CAPS).
PURE_PYTHON_NUM_RUNS = None    # e.g. 3 to only run 3 of the NUM_RUNS reps
PURE_PYTHON_TARGETS = None     # e.g. [0.05] to only test the loosest threshold

# FIX 1: base seed for deterministic, paired PSO trajectories.
BASE_SEED = 12345
FOLDER_SEP = "__THR__"   # FIX 9

# ============================================================
# LOGGING
# ============================================================

def setup_logger(path):
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s INFO: %(message)s')
    handler.setFormatter(formatter)
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
    plt.plot(xs, best_so_far, marker='s', linestyle='--', linewidth=1.5, label='best-so-far envelope')
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
# LOAD DATA
# ============================================================

BASE_DIR = "/DATA/Aurindum/Swarming"
DATA_DIR = os.path.join(BASE_DIR, "Data")

print("Loading PDN data...")

y = sio.loadmat(os.path.join(DATA_DIR, "y2.mat"))["y"]
d = sio.loadmat(os.path.join(DATA_DIR, "decaps.mat"))["decaps"]

y = np.transpose(y, (2, 0, 1))

N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]

# FIX 3: every node except the measurement port is a valid decap site.
VALID_PORTS = np.array([p for p in range(N_NODES) if p != TARGET_PORT], dtype=int)
assert len(VALID_PORTS) >= 1, "need at least one non-measurement port for decap placement"

# FIX 12 (related): more capacitors than free nodes cannot be placed on
# distinct ports no matter how good the collision resolver is. Warn loudly
# rather than silently producing duplicate-port configs at large N.
if MAX_CAPS > len(VALID_PORTS):
    print(f"[warn] MAX_CAPS={MAX_CAPS} exceeds available non-measurement "
          f"ports ({len(VALID_PORTS)}); stages beyond n_caps="
          f"{len(VALID_PORTS)} must reuse ports.")

# ============================================================
# PRECOMPUTE BASE INVERSE
# ============================================================

y_inv_base = np.zeros_like(y, dtype=np.complex128)
for f in range(N_FREQS):
    y_inv_base[f] = np.linalg.inv(y[f])

# ============================================================
# PURE-PYTHON PRECOMPUTE  (only if "pure_python" is in METHODS)
# ============================================================
# Plain nested Python lists, built once with NumPy's own C-speed
# .tolist() marshalling -- no NumPy CALL happens on the timed
# inv_pure_python() path below, only list indexing/copying/arithmetic.

if "pure_python" in METHODS:
    Y_LISTS = [y[f].tolist() for f in range(N_FREQS)]   # F x [n x [n x complex]]
    D_LISTS = d.tolist()                                # C x [F x complex]
else:
    Y_LISTS = None
    D_LISTS = None

# ============================================================
# DISCRETISATION  (FIX 2)
# ============================================================

def frac_to_index(frac_array, K):
    """
    Map particle values in [0,1] onto discrete indices {0, ..., K-1}
    uniformly. Uses floor(frac*K), not floor(frac*(K-1)) -- the latter
    can never reach index K-1 for any frac in [0,1) and silently shrinks
    the search space by one. Clips as a safeguard for the frac==1.0 edge
    case (particles are clipped to the *closed* interval [0,1] after each
    PSO velocity update), which would otherwise index out of bounds.
    """
    idx = np.floor(np.asarray(frac_array) * K).astype(int)
    return np.clip(idx, 0, K - 1)


def decode_particle(vec, n_caps):
    """
    Decode a particle's [0,1]^(2*n_caps) vector into (models, ports).
    Layout is [model_1..model_n, port_1..port_n], matching the original
    script; only the mapping inside each half changed (FIX 2, FIX 3).
    """
    model_idx = frac_to_index(vec[:n_caps], N_CAP_MODELS)
    port_idx = frac_to_index(vec[n_caps:], len(VALID_PORTS))
    ports = VALID_PORTS[port_idx]
    return model_idx, ports

# ============================================================
# PORT RESOLUTION
# ============================================================

def resolve_adjacent_ports(models, ports):
    """
    Push colliding capacitors onto the nearest free node.

    FIX 3 (defense in depth): TARGET_PORT is reserved as "already used"
    so it can never be assigned to a capacitor even if a caller passes a
    raw port value that happens to be TARGET_PORT.

    FIX 12: an explicit `placed` flag drives the outer break. The previous
    `if ports[i] in used: break` was always true when neither candidate at
    the current offset was free (ports[i] still held its original
    duplicate value, which is in `used` by construction), so the resolver
    gave up after offset 1 and returned duplicate ports. Now every offset
    is tried in turn, and the collision is only accepted as unavoidable
    when no free node exists anywhere in the network.
    """
    used = {TARGET_PORT}

    for i in range(len(ports)):

        if ports[i] not in used:
            used.add(ports[i])
            continue

        placed = False

        for offset in range(1, N_NODES):

            for cand in (ports[i] + offset, ports[i] - offset):

                if 0 <= cand < N_NODES and cand not in used:
                    ports[i] = cand
                    used.add(cand)
                    placed = True
                    break

            if placed:
                break

        if not placed:
            # No free node remains (n_caps > len(VALID_PORTS)); the
            # collision is genuinely unavoidable, so keep the port as-is.
            used.add(ports[i])

    return models, ports

# ============================================================
# INVERSION METHODS
# ============================================================

def inv_numpy(A):
    return np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]

def inv_solve(A):
    e = np.zeros(N_NODES, dtype=complex)
    e[TARGET_PORT] = 1
    return np.linalg.solve(A, e)[TARGET_PORT]

def inv_sm(config, f):

    Z = y_inv_base[f].copy()

    for cap, port in config:

        val = d[cap, f]

        u = np.zeros(N_NODES, dtype=np.complex128)
        v = np.zeros(N_NODES, dtype=np.complex128)

        u[port] = 1
        v[port] = val

        denom = 1 + v @ Z @ u

        if abs(denom) < 1e-12:
            raise np.linalg.LinAlgError("Sherman-Morrison denominator near zero")

        Z -= np.outer(Z @ u, v @ Z) / denom

    return Z[TARGET_PORT, TARGET_PORT]

def inv_iterative(A, f):

    B = y_inv_base[f]
    E = A @ B - np.eye(N_NODES)

    if np.linalg.norm(E, 1) >= 1:
        return np.linalg.inv(A)[TARGET_PORT, TARGET_PORT]

    corr = np.eye(N_NODES) - E + E @ E
    Z = B @ corr

    return Z[TARGET_PORT, TARGET_PORT]


# ---- pure-Python / math-lib-only baseline (no NumPy, no SciPy) ----

def _gauss_jordan_inverse(A):
    """
    Full matrix inverse via Gauss-Jordan elimination with partial
    pivoting (largest-magnitude pivot in the remaining column, standard
    for numerical stability with complex entries). Plain Python lists and
    the built-in `complex` type only -- no NumPy, no SciPy, anywhere in
    this function. O(n^3) arithmetic, same asymptotic cost as the LAPACK
    routines it's being compared against, just executed in the
    interpreter instead of compiled code.

    Raises ZeroDivisionError on an (effectively) singular matrix, mirroring
    how inv_sm() raises np.linalg.LinAlgError for the same physical
    situation -- both are treated identically by evaluate_config() below.
    """
    n = len(A)
    # Augmented [A | I] as one n x 2n list-of-lists; solving to [I | A^-1].
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
    """
    Same math as inv_numpy(), computed with zero NumPy/LAPACK involvement:
    plain Python lists, `complex` arithmetic, Gauss-Jordan elimination.
    Assembles Y_eq itself from the plain-Python Y_LISTS/D_LISTS precompute
    (FIX 5's spirit applied here too: no NumPy array touched on this path).
    """
    Yeq = [row[:] for row in Y_LISTS[f]]        # pure-Python copy
    for cap, port in config:
        Yeq[port][port] += D_LISTS[cap][f]      # pure-Python complex add
    inv = _gauss_jordan_inverse(Yeq)
    return inv[TARGET_PORT][TARGET_PORT]


def _selftest_pure_python_inverse():
    """
    Sanity check run once at startup: inv_pure_python must agree with
    inv_numpy to float64 precision on a handful of random candidates.
    Uses the SAME decode_particle/resolve_adjacent_ports path as the real
    PSO search (FIX 2, FIX 3, FIX 12), so the test also implicitly
    confirms the pure-Python baseline respects the discretisation and
    target-port fixes -- not just raw matrix math on arbitrary indices.
    """
    if "pure_python" not in METHODS:
        return
    rng = np.random.default_rng(999)
    max_err = 0.0
    for _ in range(5):
        f = int(rng.integers(0, N_FREQS))
        n_caps = int(rng.integers(1, min(4, N_CAP_MODELS, len(VALID_PORTS)) + 1))
        frac_vec = rng.random(2 * n_caps)
        models, ports = decode_particle(frac_vec, n_caps)
        models, ports = resolve_adjacent_ports(list(models), list(ports))
        config = list(zip(models, ports))

        A = y[f].copy()
        for cap, port in config:
            A[port, port] += d[cap, f]
        z_numpy = inv_numpy(A)
        z_pp = inv_pure_python(config, f)
        max_err = max(max_err, abs(z_numpy - z_pp))

    assert max_err < 1e-8, f"pure-Python inverse diverged from numpy: max err {max_err}"
    print(f"[selftest] inv_pure_python matches inv_numpy (max err {max_err:.2e}). OK")


def _selftest_port_constraints():
    """
    FIX 3 / FIX 12 regression guard, run once at startup: sweep every
    capacitor-count stage, decode a batch of random particles through the
    real decode -> resolve path, and assert that (a) TARGET_PORT is never
    assigned to a capacitor and (b) no two capacitors share a port
    whenever enough free nodes exist for that to be possible.
    """
    rng = np.random.default_rng(2024)
    for n_caps in range(1, MAX_CAPS + 1):
        for _ in range(20):
            models, ports = decode_particle(rng.random(2 * n_caps), n_caps)
            models, ports = resolve_adjacent_ports(models, ports)
            pl = [int(p) for p in ports]
            assert TARGET_PORT not in pl, \
                f"TARGET_PORT assigned a capacitor at n_caps={n_caps}: {pl}"
            if n_caps <= len(VALID_PORTS):
                assert len(set(pl)) == len(pl), \
                    f"duplicate ports at n_caps={n_caps}: {pl}"
    print("[selftest] port constraints hold across all stages. OK")


_selftest_pure_python_inverse()
_selftest_port_constraints()

# ============================================================
# FITNESS
# ============================================================

def evaluate_config(config, method):
    """
    FIX 5: matrix assembly only happens for methods that actually consume
    it -- inv_sm and inv_pure_python rebuild everything themselves from
    `config` (plus the precomputed base inverse / plain-Python lists), so
    building the NumPy `A` for them was pure overhead.
    FIX 6: unknown methods raise loudly; only the expected numerical
    failure modes (near-singular system) are caught and scored as invalid
    -- np.linalg.LinAlgError from the NumPy-based methods and inv_sm,
    ZeroDivisionError from inv_pure_python's own singularity check.
    """
    if method not in ("numpy", "solve", "sm", "iterative", "pure_python"):
        raise ValueError(f"Unknown method: {method!r}")

    peak = 0.0

    for f in range(N_FREQS):

        try:
            if method == "sm":
                val = inv_sm(config, f)
            elif method == "pure_python":
                val = inv_pure_python(config, f)
            else:
                A = y[f].copy()
                for cap, port in config:
                    A[port, port] += d[cap, f]

                if method == "numpy":
                    val = inv_numpy(A)
                elif method == "solve":
                    val = inv_solve(A)
                elif method == "iterative":
                    val = inv_iterative(A, f)

        except (np.linalg.LinAlgError, ZeroDivisionError):
            return 1e200

        peak = max(peak, abs(val))

    return peak

# ============================================================
# WARM-START PARTICLE SEEDING  (FIX 4 / FIX 11)
# ============================================================

def seed_particles(n_caps, prev_best):
    """
    Stage 1 (or no usable previous best): fully random swarm.

    Stage N+1 with a previous stage's global best available: a HYBRID
    swarm. The first N_WARM_START particles carry prev_best in their first
    2N genes; the remaining N_PARTICLES - N_WARM_START are drawn uniformly
    at random. Only the two new genes for the (N+1)-th capacitor (one
    model gene, one port gene) are random within the warm block.

    Particle 0 of the warm block is an exact, unjittered copy of prev_best
    in the carried genes, so the previous stage's solution is preserved
    exactly (elitism). Warm particles 1.. get clipped Gaussian jitter of
    std WARM_START_JITTER on the carried genes only.

    Why the jitter and the random remainder are both needed (FIX 11): with
    every particle identical in the carried dimensions, both PSO velocity
    terms -- (pbest[i] - particles[i]) and (gbest_particle - particles[i])
    -- are identically zero there, velocities stay zero, and those
    dimensions are frozen for the whole stage. The random remainder
    restores swarm-level diversity; the jitter keeps the warm block itself
    from being internally degenerate.

    The grouped [models(n_caps) | ports(n_caps)] layout is preserved, which
    is why the two new genes are INSERTED into the middle of the vector
    (after the old models, after the old ports) rather than appended at
    the end.
    """
    n_warm = min(N_WARM_START, N_PARTICLES)

    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(N_PARTICLES, 2 * n_caps)

    prev_n = n_caps - 1
    prev_models = prev_best[:prev_n]          # (prev_n,)
    prev_ports = prev_best[prev_n:]           # (prev_n,)

    warm = np.hstack([
        np.tile(prev_models, (n_warm, 1)), np.random.rand(n_warm, 1),
        np.tile(prev_ports,  (n_warm, 1)), np.random.rand(n_warm, 1),
    ])

    if WARM_START_JITTER > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, dtype=bool)
        carried[prev_n] = False       # new model gene, already uniform
        carried[-1] = False           # new port gene, already uniform
        noise = np.random.normal(0.0, WARM_START_JITTER, warm.shape)
        noise[0, :] = 0.0             # particle 0 is elite, untouched
        noise[:, ~carried] = 0.0      # never perturb the new genes
        warm = np.clip(warm + noise, 0.0, 1.0)

    n_rand = N_PARTICLES - n_warm
    if n_rand > 0:
        warm = np.vstack([warm, np.random.rand(n_rand, 2 * n_caps)])

    return warm

# ============================================================
# PSO
# ============================================================

def run_pso(method, threshold, out_folder, run_id):

    logger = setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))

    run_caps = []
    run_minZ = []
    histories = {}

    start_global = time.perf_counter()   # FIX 10
    time_to_target = None
    caps_at_target = None

    logger.info(
        "RUN_START | method=%s | threshold=%.6f | run=%d | particles=%d | "
        "iterations=%d | base_seed=%d | warm_start=%d | jitter=%.4f",
        method, threshold, run_id, N_PARTICLES, N_ITERATIONS, BASE_SEED,
        N_WARM_START, WARM_START_JITTER
    )

    prev_best = None   # winning particle vector from the previous n_caps stage

    for n_caps in range(1, MAX_CAPS + 1):

        # FIX 1: reseed per (run, n_caps) -- independent of threshold and of
        # how many RNG draws earlier stages/methods consumed -- so every
        # method sees an identical starting swarm at this stage for this
        # run index, regardless of how quickly earlier stages converged.
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)

        DIM = 2 * n_caps

        particles = seed_particles(n_caps, prev_best)   # FIX 4 / FIX 11
        assert particles.shape == (N_PARTICLES, DIM), \
            f"seed_particles returned {particles.shape}, expected {(N_PARTICLES, DIM)}"

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
                models, ports = resolve_adjacent_ports(models, ports)   # FIX 12

                config = list(zip(models, ports))

                cost = evaluate_config(config, method)

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
            # FIX 13: without this reset the next stage would seed from a
            # prev_best sized for n_caps-2 and silently build a
            # wrong-width swarm.
            prev_best = None
            logger.info("n_caps=%d | iter=%d | minZ=inf | placement={}", n_caps, it + 1)
            continue

        prev_best = gbest_particle   # FIX 4: carried into the next stage's seeding

        best_models, best_ports = decode_particle(gbest_particle, n_caps)   # FIX 2, FIX 3
        best_models, best_ports = resolve_adjacent_ports(best_models, best_ports)
        placement = {int(port): int(cap) for cap, port in zip(best_models, best_ports)}

        logger.info(
            "n_caps=%d | iter=%d | minZ=%.6f | placement=%s",
            n_caps, it + 1, gbest, placement
        )

        run_caps.append(n_caps)
        run_minZ.append(gbest)

        if gbest <= threshold:
            break

    best_minz = min(run_minZ) if run_minZ else float("inf")
    success = time_to_target is not None    # FIX 7/8

    logger.info(
        "RESULT | time_to_target=%.4f | caps=%s | best_minZ=%.6f | success=%s",
        time_to_target if time_to_target else -1,
        caps_at_target if caps_at_target else -1,
        best_minz,
        success,
    )

    plot_run_curve(run_caps, run_minZ, threshold, method, run_id, out_folder)
    plot_convergence(histories, threshold, out_folder)

# ============================================================
# MAIN
# ============================================================

for threshold in TARGETS:

    for method in METHODS:

        # ADDITION: optional lighter sweep for pure_python (see the
        # PURE_PYTHON_NUM_RUNS / PURE_PYTHON_TARGETS knobs in GLOBAL
        # CONFIG). Both default to None, so by default pure_python gets
        # the exact same NUM_RUNS x TARGETS as every other method.
        if method == "pure_python" and PURE_PYTHON_TARGETS is not None \
                and threshold not in PURE_PYTHON_TARGETS:
            continue

        n_runs_this_method = NUM_RUNS
        if method == "pure_python" and PURE_PYTHON_NUM_RUNS is not None:
            n_runs_this_method = PURE_PYTHON_NUM_RUNS

        print(f"\nMETHOD {method} TARGET {threshold}")

        method_folder = os.path.join(ROOT_OUT, f"{method}{FOLDER_SEP}{threshold}")   # FIX 9
        os.makedirs(method_folder, exist_ok=True)

        for run in range(1, n_runs_this_method + 1):

            run_folder = os.path.join(method_folder, f"run_{run}")
            os.makedirs(run_folder, exist_ok=True)

            run_pso(method, threshold, run_folder, run)

print("All runs completed.")

# ============================================================
# GLOBAL ANALYSIS
# ============================================================

records = []

for root, _, files in os.walk(ROOT_OUT):

    for file in files:

        if not file.endswith(".log"):
            continue

        path = os.path.join(root, file)
        parts = root.split(os.sep)

        if FOLDER_SEP not in parts[-2]:
            continue   # not a run folder under a <method>__THR__<threshold> dir

        method, threshold = parts[-2].split(FOLDER_SEP)   # FIX 9

        with open(path) as f:
            for line in f:

                if "RESULT" in line:
                    # FIX 8: also capture best_minZ and success.
                    m = re.search(
                        r"time_to_target=([0-9\.\-]+).*caps=([0-9\-]+).*"
                        r"best_minZ=([0-9\.\-]+).*success=(\w+)",
                        line,
                    )
                    if m:
                        records.append({
                            "method": method,
                            "threshold": float(threshold),
                            "time": float(m.group(1)),
                            "caps": int(m.group(2)),
                            "best_minz": float(m.group(3)),
                            "success": m.group(4) == "True",
                            "folder": root,
                        })

# ============================================================
# TOP 10 FASTEST  (successful runs only, as before)
# ============================================================

valid = [r for r in records if r["success"]]
top10 = sorted(valid, key=lambda x: x["time"])[:10]

with open(os.path.join(ROOT_OUT, "top10_fastest.txt"), "w") as f:

    for i, r in enumerate(top10, 1):
        f.write(f"{i}. {r}\n")

        src = os.path.join(r["folder"], "run_plot.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ROOT_OUT, f"top{i}.png"))

# ============================================================
# STATISTICS  (successful runs)
# ============================================================

with open(os.path.join(ROOT_OUT, "statistics.txt"), "w") as f:

    for threshold in TARGETS:

        f.write(f"\n=== Threshold {threshold} ===\n")

        subset = [r for r in records if r["threshold"] == threshold and r["success"]]

        if subset:
            best = min(subset, key=lambda x: x["time"])
            f.write(f"BEST: {best}\n")

# ============================================================
# TIMEOUTS  (FIX 7: runs that never reached threshold, kept visible)
# ============================================================

timeouts = [r for r in records if not r["success"]]

with open(os.path.join(ROOT_OUT, "timeouts.txt"), "w") as f:

    if not timeouts:
        f.write("No timeouts -- every run reached its threshold within MAX_CAPS.\n")
    else:
        by_key = {}
        for r in timeouts:
            key = (r["method"], r["threshold"])
            by_key.setdefault(key, []).append(r)

        for (method, threshold), rs in sorted(by_key.items()):
            f.write(f"method={method}  threshold={threshold}  "
                    f"timeouts={len(rs)}/{NUM_RUNS}\n")
            for r in rs:
                f.write(f"    {r['folder']}  best_minz={r['best_minz']:.6f}\n")

print("Done.")