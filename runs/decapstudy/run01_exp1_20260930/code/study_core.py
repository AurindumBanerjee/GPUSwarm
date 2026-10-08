"""
Core of the decap-placement study (CPU): data, fitness, PSO.

PSO logic follows GPUSwarm/CPUTest/SB.py (FIX 1-13): per-(run, n_caps)
reseeding, floor(frac*K) discretisation, hybrid warm start (40 % of the swarm),
duplicate-free port resolution, immediate stop on hitting the target.

Generalisations over SB.py
  * several observation ports (cost = max_k peak|Z_kk| / Ztarget_k, target 1.0)
  * observation ports are never decap sites (index 0 everywhere; indices 0-4 on
    y_sp_mphy_full) -- pads are whatever remains, N comes from the matrix
  * band-limited objective, re-scoring on all bands
  * a bare-network check (0 caps) before any search
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.io as sio

import study_config as C

# ============================================================ data

_CACHE = {}


def _mat(*parts):
    return os.path.join(C.DATA_ROOT, *parts)


def freqs():
    if "f" not in _CACHE:
        _CACHE["f"] = sio.loadmat(_mat("MPHY", "freq_sp.mat"))["freq"].ravel()
    return _CACHE["f"]


def decaps():
    if "d" not in _CACHE:
        _CACHE["d"] = np.ascontiguousarray(
            sio.loadmat(_mat("Decaps", "decaps_sp.mat"))["decaps"])
    return _CACHE["d"]


def _load_y(*parts):
    key = "/".join(parts)
    if key not in _CACHE:
        y = sio.loadmat(_mat(*parts))["y"]                    # (N, N, F)
        _CACHE[key] = np.ascontiguousarray(np.transpose(y, (2, 0, 1)))
    return _CACHE[key]


def n_freq(band):
    """Bands are prefixes of the ascending grid, so an objective band is y[:nf]."""
    hi = C.BANDS[band]
    f = freqs()
    if hi is None or hi == C.WEIGHTED:
        return len(f)
    return int(np.searchsorted(f, hi, side="right"))


def movement_weights(prob):
    """B4 weights, shape (F, n_obs), normalised to max 1 per observation port.

    w_k(f) = max over every (single decap model, single pad) of the relative
    REDUCTION of |Z_kk(f)| that one capacitor achieves, via the exact rank-1
    update  Z'_kk = Z_kk - d Z_kp Z_pk / (1 + d Z_pp).  Bins no single capacitor
    can lower get ~0 weight. Cached in memory and under OUT_ROOT/cache.
    """
    if getattr(prob, "_W", None) is not None:
        return prob._W
    path = os.path.join(C.OUT_ROOT, "cache", f"w_{prob.name}.npy")
    if os.path.exists(path):
        prob._W = np.load(path)
        return prob._W
    d = decaps()
    Z = np.linalg.inv(prob.y)
    W = np.zeros((Z.shape[0], len(prob.obs)))
    for k, o in enumerate(prob.obs):
        zkk = Z[:, o, o]
        base = np.abs(zkk)
        best = np.zeros(len(base))
        for p in prob.pads:
            ab = Z[:, o, p] * Z[:, p, o]
            znew = zkk[None, :] - d * ab[None, :] / (1.0 + d * Z[:, p, p][None, :])
            best = np.maximum(best, (base[None, :] - np.abs(znew)).max(axis=0) / base)
        W[:, k] = best / best.max()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, W)
    prob._W = W
    return W


def band_spec(prob, band):
    """(nf, W): W is None for physical bands, the (F, n_obs) weights for B4."""
    W = movement_weights(prob) if C.BANDS[band] == C.WEIGHTED else None
    return n_freq(band), W


class Problem:
    """A PDN + observation ports + decap sites + per-observation-port targets."""

    def __init__(self, name, y, obs, targets, rails=None):
        self.name = name
        self.y = y
        self.N = y.shape[1]
        self.obs = np.asarray(obs, dtype=int)
        self.pads = np.array([p for p in range(self.N) if p not in set(self.obs.tolist())],
                             dtype=int)
        self.targets = np.asarray(targets, dtype=float)
        self.rails = rails or [name]
        assert len(self.obs) == len(self.targets) == len(self.rails)
        assert len(self.pads) >= 1
        self._yinv = None
        self._lists = None
        self._E = None

    @property
    def y_inv_base(self):
        if self._yinv is None:
            self._yinv = np.linalg.inv(self.y)
        return self._yinv

    @property
    def y_lists(self):
        if self._lists is None:
            self._lists = [self.y[f].tolist() for f in range(self.y.shape[0])]
        return self._lists

    @property
    def E(self):
        if self._E is None:
            E = np.zeros((self.N, len(self.obs)), dtype=complex)
            E[self.obs, np.arange(len(self.obs))] = 1.0
            self._E = E
        return self._E


def ztarget(v, di):
    return v * C.RIPPLE / di


def _rail(name):
    for r in C.MPHY_RAILS:
        if r[0] == name:
            return r
    raise KeyError(name)


def build_problem(key):
    """key: mphy_<rail> | mphy_full | mphy_improvable | ddr21 | ddr_full | <rail>"""
    if key in _CACHE.get("_problems", {}):
        return _CACHE["_problems"][key]
    P = None
    rail_names = [r[0] for r in C.MPHY_RAILS]
    k = key[5:] if key.startswith("mphy_") and key[5:] in rail_names else key
    if k in rail_names:
        r = _rail(k)
        y = _load_y("MPHY", r[1])
        P = Problem(f"mphy_{k}", y, [0], [ztarget(r[5], r[6])], rails=[k])
    elif key == "mphy_full":
        y = _load_y("MPHY", "y_sp_mphy_full.mat")
        P = Problem("mphy_full", y, list(range(5)),
                    [ztarget(r[5], r[6]) for r in C.MPHY_RAILS],
                    rails=[r[0] for r in C.MPHY_RAILS])
    elif key == "mphy_improvable":
        P = _subnetwork_improvable()
    elif key == "ddr21":
        y = _load_y("VDDQ_DDR3", "y_sp_ddr21.mat")
        P = Problem("ddr21", y, [0], [ztarget(C.DDR_VOLTAGE, C.DDR_DI)], rails=["ddr"])
    elif key == "ddr_full":
        y = _load_y("VDDQ_DDR3", "y_sp_ddr_full.mat")
        P = Problem("ddr_full", y, [0], [ztarget(C.DDR_VOLTAGE, C.DDR_DI)], rails=["ddr"])
    else:
        raise KeyError(key)
    _CACHE.setdefault("_problems", {})[key] = P
    return P


def _subnetwork_improvable():
    """mer2 + pll1v0 + mphyvdd carved out of y_sp_mphy_full.

    Open-circuit reduction (never slice Y directly): Z = inv(Y_full), keep the
    rows/cols of the three observation ports and their pad blocks, Y = inv(Z_sub).
    Local order: [obs_mer2, obs_pll1v0, obs_mphyvdd, pads...]. Its pad budget
    is every pad of the three rails (38) -- mer1's and pll1v8's 30 pads are
    simply not in the network, so their share of the budget is redistributed.
    """
    full = _load_y("MPHY", "y_sp_mphy_full.mat")
    keep_rails = [_rail(n) for n in C.IMPROVABLE_RAILS]
    obs_idx = [r[2] for r in keep_rails]
    pad_idx = [r[3] + j for r in keep_rails for j in range(r[4])]
    idx = np.array(obs_idx + pad_idx)
    Zs = np.linalg.inv(full)[:, idx][:, :, idx]
    y = np.ascontiguousarray(np.linalg.inv(Zs))
    return Problem("mphy_improvable", y, list(range(len(obs_idx))),
                   [ztarget(r[5], r[6]) for r in keep_rails],
                   rails=[r[0] for r in keep_rails])


def scope_problems(scope):
    if scope == "A1":
        return [build_problem(f"mphy_{r[0]}") for r in C.MPHY_RAILS]
    return [build_problem({"A2": "mphy_full", "A3": "mphy_improvable",
                           "A4": "ddr21", "A5": "ddr_full"}[scope])]


# ============================================================ fitness

def _pure_gauss_jordan_inverse(A):
    """Gauss-Jordan with partial pivoting, plain lists + complex (SB.py)."""
    n = len(A)
    M = [row[:] + [1.0 + 0.0j if i == r else 0.0j for i in range(n)]
         for r, row in enumerate(A)]
    for col in range(n):
        pr = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pr][col]) < 1e-300:
            raise ZeroDivisionError("singular")
        if pr != col:
            M[col], M[pr] = M[pr], M[col]
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


def diag_z(prob, nf, config, method="numpy", mode="batched"):
    """Z_kk(f) at every observation port, shape (nf, n_obs).

    mode "batched": numpy over the whole frequency stack (Grid 1; method is
      always numpy there). mode "loop": one Python iteration per frequency for
      every method, so per-evaluation cost is comparable across methods
      (Grid 2, mirrors SB.py).
    """
    d = decaps()
    obs = prob.obs
    no = len(obs)

    if mode == "batched":
        if method != "numpy":
            raise ValueError("batched mode is numpy only")
        A = prob.y[:nf].copy()
        for cap, port in config:
            A[:, port, port] += d[cap, :nf]
        return np.linalg.inv(A)[:, obs, obs]

    out = np.empty((nf, no), dtype=complex)

    if method == "pure_python":
        Yl = prob.y_lists
        dcols = [(d[cap, :nf].tolist(), int(port)) for cap, port in config]
        ol = obs.tolist()
        for f in range(nf):
            Yeq = [row[:] for row in Yl[f]]
            for col, port in dcols:
                Yeq[port][port] += col[f]
            inv = _pure_gauss_jordan_inverse(Yeq)
            out[f] = [inv[o][o] for o in ol]
        return out

    yinv = prob.y_inv_base if method in ("sm", "iterative") else None
    I = np.eye(prob.N)
    ar = np.arange(no)
    for f in range(nf):
        if method == "sm":
            Z = yinv[f].copy()
            for cap, port in config:
                val = d[cap, f]
                denom = 1.0 + val * Z[port, port]
                if abs(denom) < 1e-12:
                    raise np.linalg.LinAlgError("Sherman-Morrison denominator ~ 0")
                Z -= np.outer(Z[:, port], val * Z[port, :]) / denom
            out[f] = Z[obs, obs]
            continue
        A = prob.y[f].copy()
        for cap, port in config:
            A[port, port] += d[cap, f]
        if method == "numpy":
            out[f] = np.linalg.inv(A)[obs, obs]
        elif method == "solve":
            out[f] = np.linalg.solve(A, prob.E)[obs, ar]
        elif method == "iterative":
            B = yinv[f]
            Er = A @ B - I
            if np.linalg.norm(Er, 1) >= 1:
                out[f] = np.linalg.inv(A)[obs, obs]
            else:
                out[f] = (B @ (I - Er + Er @ Er))[obs, obs]
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def evaluate(prob, bs, config, method="numpy", mode="batched"):
    """max_k  peak_f w_k(f)|Z_kk| / Ztarget_k   (<= 1.0 means every target is met).

    bs = (nf, W) from band_spec(); W is None (w=1) except on the weighted band.
    """
    nf, W = bs
    try:
        z = diag_z(prob, nf, config, method, mode)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return 1e200
    az = np.abs(z)
    if W is not None:
        az = az * W[:nf]
    return float((az.max(axis=0) / prob.targets).max())


def score_bands(prob, config):
    """Re-score a placement on ALL bands (independent of how it was found).
    B4's 'peaks_ohm' are the weighted peaks max_f w(f)|Z_kk(f)|."""
    z = np.abs(diag_z(prob, len(freqs()), config, "numpy", "batched"))
    out = {}
    for b in C.BANDS:
        nf, W = band_spec(prob, b)
        zz = z[:nf] * W[:nf] if W is not None else z[:nf]
        peaks = zz.max(axis=0)
        out[b] = {"peaks_ohm": peaks.tolist(),
                  "ratio": float((peaks / prob.targets).max())}
    return out


# ============================================================ PSO pieces

def frac_to_index(frac, K):
    return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)


def decode_particle(vec, n_caps, prob):
    models = frac_to_index(vec[:n_caps], decaps().shape[0])
    ports = prob.pads[frac_to_index(vec[n_caps:], len(prob.pads))]
    return models, ports


def resolve_adjacent_ports(models, ports, prob):
    """Push colliding capacitors to the nearest free PAD (never an observation port)."""
    padset = set(prob.pads.tolist())
    used = set(prob.obs.tolist())
    for i in range(len(ports)):
        if ports[i] not in used:
            used.add(int(ports[i]))
            continue
        placed = False
        for offset in range(1, prob.N):
            for cand in (ports[i] + offset, ports[i] - offset):
                if cand in padset and cand not in used:
                    ports[i] = cand
                    used.add(int(cand))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            used.add(int(ports[i]))
    return models, ports


def seed_particles(n_caps, prev_best, n_particles):
    n_warm = min(int(round(C.WARM_FRACTION * n_particles)), n_particles)
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(n_particles, 2 * n_caps)
    pn = n_caps - 1
    warm = np.hstack([
        np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
        np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1),
    ])
    if C.WARM_JITTER > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, dtype=bool)
        carried[pn] = False
        carried[-1] = False
        noise = np.random.normal(0.0, C.WARM_JITTER, warm.shape)
        noise[0, :] = 0.0
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    if n_particles > n_warm:
        warm = np.vstack([warm, np.random.rand(n_particles - n_warm, 2 * n_caps)])
    return warm


# ============================================================ PSO run

def run_pso(prob, band, run_id, cfg):
    """One seeded run: grow the capacitor count until every target is met.

    cfg keys: n_particles, n_iters, method (default numpy), mode (default
    batched), max_caps (default: number of pads), time_limit_s, patience.
    Returns a JSON-able dict (no rescoring of *other* bands is skipped: the
    winning placement is always scored on B1/B2/B3).
    """
    method = cfg.get("method", "numpy")
    mode = cfg.get("mode", "batched")
    P, IT = cfg["n_particles"], cfg["n_iters"]
    # MAX_CAPS = all pads (cap of 20 only on the 21-port DDR3 PDN); cfg["max_caps"]
    # can only lower it (smoke tests).
    max_caps = min(cfg.get("max_caps") or len(prob.pads),
                   C.MAX_CAPS_CAP.get(prob.name, len(prob.pads)), len(prob.pads))
    limit = cfg.get("time_limit_s")
    patience = cfg.get("patience", C.PATIENCE)
    threads = cfg.get("threads", C.THREADS) if mode == "batched" else 1
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    nf = band_spec(prob, band)          # (nf, weights-or-None); name kept for brevity

    t0 = time.perf_counter()
    n_evals = 1
    iters_total = 0
    timed_out = False
    time_to_target = None
    caps_needed = None

    bare = evaluate(prob, nf, [], method, mode)
    best = {"cost": bare, "config": [], "n_caps": 0, "iter": 0}
    bare_pass = bare <= 1.0
    curve_caps, curve_ratio, histories = [], [], []
    success = bare_pass
    if bare_pass:
        time_to_target, caps_needed = 0.0, 0

    prev_best = None
    stalled, ref_cost = 0, bare
    stop = bare_pass

    for n_caps in range(1, max_caps + 1):
        if stop:
            break
        np.random.seed(C.BASE_SEED + run_id * 10000 + n_caps)
        particles = seed_particles(n_caps, prev_best, P)
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_val = np.full(P, np.inf)
        gbest, gbest_particle = np.inf, None
        hist = []

        for it in range(IT):
            configs = []
            for i in range(P):
                models, ports = decode_particle(particles[i], n_caps, prob)
                models, ports = resolve_adjacent_ports(models, ports, prob)
                configs.append([(int(m), int(p)) for m, p in zip(models, ports)])
            # Threaded mode evaluates the whole iteration at once; results are then
            # consumed in particle order, so trajectories and the immediate-stop point
            # are identical to serial (n_evals counts up to the stopping particle).
            costs = list(pool.map(lambda c: evaluate(prob, nf, c, method, mode), configs)) \
                if pool else None
            for i in range(P):
                config = configs[i]
                cost = costs[i] if pool else evaluate(prob, nf, config, method, mode)
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]
                if cost < gbest:
                    gbest, gbest_particle = cost, particles[i].copy()
                if cost < best["cost"]:
                    best = {"cost": cost, "config": config, "n_caps": n_caps,
                            "iter": iters_total + 1}
                if gbest <= 1.0:                       # immediate stop
                    break
            hist.append(gbest)
            iters_total += 1
            if gbest <= 1.0:
                success, stop = True, True
                time_to_target = time.perf_counter() - t0
                caps_needed = n_caps
                break
            if limit is not None and time.perf_counter() - t0 > limit:
                timed_out, stop = True, True
                break
            w = C.W_MAX - (C.W_MAX - C.W_MIN) * (it / IT)
            velocities = (w * velocities
                          + C.C1 * np.random.rand(*particles.shape) * (pbest - particles)
                          + C.C2 * np.random.rand(*particles.shape) * (gbest_particle - particles))
            particles = np.clip(particles + velocities, 0.0, 1.0)

        histories.append(hist)
        if gbest_particle is None:
            prev_best = None                             # FIX 13
            continue
        prev_best = gbest_particle
        curve_caps.append(n_caps)
        curve_ratio.append(gbest)

        if patience and not stop:
            if best["cost"] < ref_cost * (1 - C.PATIENCE_TOL):
                ref_cost, stalled = best["cost"], 0
            else:
                stalled += 1
                if stalled >= patience:
                    stop = True

    wall = time.perf_counter() - t0
    if pool:
        pool.shutdown()
    config = best["config"]
    return {
        "success": bool(success),
        "bare_pass": bool(bare_pass),
        "timed_out": bool(timed_out),
        "caps_needed": caps_needed,
        "caps_used": best["n_caps"],
        "best_ratio_own_band": best["cost"],
        "conv_iter": best["iter"] if not success else iters_total,
        "total_iters": iters_total,
        "n_evals": n_evals,
        "wall_s": wall,
        "time_to_target_s": time_to_target,
        "max_caps": max_caps,
        "curve_caps": curve_caps,
        "curve_ratio": curve_ratio,
        "histories": histories,
        "placement": [[p, m] for m, p in config],       # [port, model]
        "rescored": score_bands(prob, config),
    }
