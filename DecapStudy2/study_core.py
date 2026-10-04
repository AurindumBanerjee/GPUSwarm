"""
Core of the decap-placement study, EXPERIMENT 2 (CPU): data, fitness, PSO.

PSO logic follows GPUSwarm/CPUTest/SB.py (FIX 1-13): per-(run, n_caps)
reseeding, floor(frac*K) discretisation, hybrid warm start (40 % of the swarm),
duplicate-free port resolution, immediate stop on hitting the target.

Exp 2 versus Exp 1
  * targets come from targets.json (several threshold LEVELS per rail), never
    from assumed voltages/currents; load_targets() refuses nulls
  * objective: A2 (mphy_full) minimises max_k peak|Z_kk| / Ztarget_k (tightest
    level); every other problem minimises the RAW peak |Z_kk| in ohm (max over
    its observation ports) -- targets only set the stop condition
  * a run stops when the TIGHTEST level is met; every looser level is
    recorded the first time any evaluated placement meets it
  * the placement (model + pad per capacitor) is returned for every record
  * observation ports come first, never take a capacitor; the capacitor limit
    is N - n_obs, derived from the loaded matrix

Generalisations over SB.py: several observation ports, band-limited objective,
re-scoring on all bands, a bare-network check (0 caps) before any search.
"""
import os
import json
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


class TargetsNotSet(Exception):
    """targets.json is missing, malformed, or still holds nulls."""


# Problems normally REQUIRE filled targets. validate.py switches this off so the
# port rules can be checked while targets.json still holds its placeholders;
# run_pso() refuses to run on a problem built that way.
_REQUIRE_TARGETS = True


def set_require_targets(flag):
    global _REQUIRE_TARGETS
    _REQUIRE_TARGETS = bool(flag)
    _CACHE.pop("_problems", None)


def load_targets(path=None):
    """(levels, table). levels = [(label, multiplier)] ascending, i.e. TIGHTEST
    first; table[key][label] = target in ohm. Raises TargetsNotSet unless every
    required target (5 rails + ddr21 + ddr_full, every level) is a positive number."""
    path = path or C.TARGETS_FILE
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as e:
        raise TargetsNotSet(f"cannot read {path}: {e}")
    mult = cfg.get("multipliers")
    if (not isinstance(mult, list) or not mult
            or not all(isinstance(m, (int, float)) and not isinstance(m, bool) and m > 0
                       for m in mult)):
        raise TargetsNotSet(f"{path}: 'multipliers' must be a non-empty list of positive numbers")
    levels = [(f"{m:g}", float(m)) for m in sorted(set(mult))]
    keys = [r[0] for r in C.MPHY_RAILS] + list(C.DDR_TARGET_KEYS)
    table, bad = {}, []
    for k in keys:
        row = (cfg.get("targets_ohm") or {}).get(k)
        if not isinstance(row, dict):
            bad.append(f"{k}: missing")
            continue
        table[k] = {}
        for lab, _ in levels:
            v = row.get(lab)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                bad.append(f"{k} x{lab} = {v!r}")
            else:
                table[k][lab] = float(v)
    if bad:
        raise TargetsNotSet(f"{path} is not filled in ({len(bad)} bad/null target(s)): "
                            + "; ".join(bad[:12]) + (" ..." if len(bad) > 12 else "")
                            + ". Run derive_targets.py after Exp 1 has finished.")
    return levels, table


def get_targets():
    if "targets" not in _CACHE:
        _CACHE["targets"] = load_targets()
    return _CACHE["targets"]


class Problem:
    """A PDN + observation ports + decap sites + per-observation-port targets.

    Port rules: the n_obs observation ports are indices 0..n_obs-1 and never take a
    capacitor; the pads are every remaining index, so max_caps = N - n_obs. Both come
    from the loaded matrix (n_obs from config.N_OBS_BY_FILE), never hardcoded here.
    """

    def __init__(self, name, y, n_obs, rails, target_keys, global_index=None):
        self.name = name
        self.y = y
        self.N = y.shape[1]
        self.n_obs = int(n_obs)
        assert 1 <= self.n_obs < self.N, f"{name}: n_obs={self.n_obs} with N={self.N}"
        self.obs = np.arange(self.n_obs, dtype=int)
        self.pads = np.arange(self.n_obs, self.N, dtype=int)
        self.max_caps = len(self.pads)                    # = N - n_obs
        self.rails = list(rails)
        self.target_keys = list(target_keys)
        assert len(self.rails) == len(self.target_keys) == self.n_obs
        # local index -> index in the parent PDN (y_sp_mphy_full / the DDR matrix)
        self.global_index = (np.arange(self.N) if global_index is None
                             else np.asarray(global_index, dtype=int))
        self.objective = "ratio" if name in C.RATIO_OBJECTIVE_PROBLEMS else "raw"
        if _REQUIRE_TARGETS:
            levels, table = get_targets()                 # raises TargetsNotSet on any null
            # [(label, multiplier, per-obs target vector in ohm)], tightest first
            self.levels = [(lab, m, np.array([table[k][lab] for k in self.target_keys]))
                           for lab, m in levels]
            self.targets = self.levels[0][2]              # tightest: stop condition / ratio scale
        else:                                             # --validate only: no PSO is possible
            self.levels, self.targets = [], None
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


def _rail(name):
    for r in C.MPHY_RAILS:
        if r[0] == name:
            return r
    raise KeyError(name)


def _n_obs(fname):
    return C.N_OBS_BY_FILE.get(fname, C.DEFAULT_N_OBS)


def build_problem(key):
    """key: mphy_<rail> | <rail> | mphy_full | mphy_improvable | ddr21 | ddr_full"""
    if key in _CACHE.get("_problems", {}):
        return _CACHE["_problems"][key]
    P = None
    rail_names = [r[0] for r in C.MPHY_RAILS]
    k = key[5:] if key.startswith("mphy_") and key[5:] in rail_names else key
    if k in rail_names:
        r = _rail(k)
        y = _load_y("MPHY", r[1])
        # rail file: obs first, then its pads; pad j (1..n-1) is pad r[3]+j-1 of mphy_full
        gi = [r[2]] + [r[3] + j for j in range(y.shape[1] - 1)]
        P = Problem(f"mphy_{k}", y, _n_obs(r[1]), [k], [k], global_index=gi)
    elif key == "mphy_full":
        fn = "y_sp_mphy_full.mat"
        y = _load_y("MPHY", fn)
        names = [r[0] for r in C.MPHY_RAILS]
        P = Problem("mphy_full", y, _n_obs(fn), names, names)
    elif key == "mphy_improvable":
        P = _subnetwork_improvable()
    elif key in ("ddr21", "ddr_full"):
        fn = "y_sp_ddr21.mat" if key == "ddr21" else "y_sp_ddr_full.mat"
        y = _load_y("VDDQ_DDR3", fn)
        P = Problem(key, y, _n_obs(fn), [key], [key])
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
    names = [r[0] for r in keep_rails]
    return Problem("mphy_improvable", y, len(obs_idx), names, names, global_index=idx)


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


def peaks_of(prob, bs, config, method="numpy", mode="batched"):
    """Per-observation-port peak  max_f w_k(f)|Z_kk(f)|  in ohm, shape (n_obs,), or
    None for a (near-)singular network. bs = (nf, W) from band_spec(); W is None
    (w = 1) except on the weighted band."""
    nf, W = bs
    try:
        z = diag_z(prob, nf, config, method, mode)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return None
    az = np.abs(z)
    if W is not None:
        az = az * W[:nf]
    return az.max(axis=0)


def cost_of(prob, peaks):
    """Objective. 'ratio' (A2): max_k peak_k / Ztarget_k at the tightest level.
    'raw' (all other problems): max_k peak_k in ohm, i.e. the raw peak |Z11| for
    single-observation problems. Targets only enter the 'ratio' objective."""
    if peaks is None:
        return 1e200
    if prob.objective == "ratio":
        return float((peaks / prob.targets).max())
    return float(peaks.max())


def levels_met(prob, peaks):
    """Labels of every threshold level whose per-rail targets are all met."""
    if peaks is None:
        return []
    return [lab for lab, _, vec in prob.levels if bool(np.all(peaks <= vec))]


def evaluate(prob, bs, config, method="numpy", mode="batched"):
    """(cost, peaks) of one placement."""
    peaks = peaks_of(prob, bs, config, method, mode)
    return cost_of(prob, peaks), peaks


def score_bands(prob, config):
    """Re-score a placement on ALL bands (independent of how it was found).
    B4's 'peaks_ohm' are the weighted peaks max_f w(f)|Z_kk(f)|. 'ratio' is
    max_k peak_k / Ztarget_k(tightest level); 'met_levels' lists the levels met."""
    z = np.abs(diag_z(prob, len(freqs()), config, "numpy", "batched"))
    out = {}
    for b in C.BANDS:
        nf, W = band_spec(prob, b)
        zz = z[:nf] * W[:nf] if W is not None else z[:nf]
        peaks = zz.max(axis=0)
        out[b] = {"peaks_ohm": peaks.tolist(),
                  "ratio": float((peaks / prob.targets).max()),
                  "met_levels": levels_met(prob, peaks)}
    return out


def placement_records(prob, config):
    """Every capacitor of a placement as explicit JSON. config = [(model, pad)]:
    model = 0-based row of decaps_sp.mat; pad = 0-based index in THIS problem's
    matrix; pad_global = index in the parent PDN (y_sp_mphy_full / the DDR matrix)."""
    return [{"cap": i + 1, "model": int(m), "pad": int(p),
             "pad_global": int(prob.global_index[p])} for i, (m, p) in enumerate(config)]


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
    """One seeded run: grow the capacitor count until the TIGHTEST target level is met.

    cfg keys: n_particles, n_iters, method (default numpy), mode (default
    batched), max_caps (default and ceiling: prob.max_caps = N - n_obs),
    time_limit_s, patience, threads.

    Every evaluated placement is checked against every threshold level; the first
    one that meets a level is recorded for that level (caps, iteration, evaluations,
    time, placement), so one run reports against all levels. The winning placement
    is always re-scored on every band.
    """
    if not prob.levels:
        raise TargetsNotSet("problem was built without targets; refusing to run PSO")
    method = cfg.get("method", "numpy")
    mode = cfg.get("mode", "batched")
    P, IT = cfg["n_particles"], cfg["n_iters"]
    max_caps = min(cfg.get("max_caps") or prob.max_caps, prob.max_caps)
    limit = cfg.get("time_limit_s")
    patience = cfg.get("patience", C.PATIENCE)
    threads = cfg.get("threads", C.THREADS) if mode == "batched" else 1
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    bs = band_spec(prob, band)          # (nf, weights-or-None)
    tight = prob.levels[0][0]           # label of the tightest level = stop condition

    t0 = time.perf_counter()
    n_evals = 0
    iters_total = 0
    timed_out = False
    hits = {}                           # level label -> info of the first placement meeting it

    def note(peaks, config, n_caps):
        """Record every level this placement is the first to meet; True if tightest met."""
        for lab in levels_met(prob, peaks):
            if lab not in hits:
                hits[lab] = {"caps_needed": n_caps, "iter": iters_total + 1, "n_evals": n_evals,
                             "time_s": time.perf_counter() - t0,
                             "placement": placement_records(prob, config)}
        return tight in hits

    cost0, peaks0 = evaluate(prob, bs, [], method, mode)
    n_evals += 1
    best = {"cost": cost0, "config": [], "n_caps": 0, "iter": 0}
    bare_pass = note(peaks0, [], 0)
    if bare_pass:
        for h in hits.values():
            h["iter"] = 0
    curve_caps, curve_cost, histories = [], [], []
    prev_best = None
    stalled, ref_cost = 0, cost0
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
            res = list(pool.map(lambda c: evaluate(prob, bs, c, method, mode), configs)) \
                if pool else None
            for i in range(P):
                config = configs[i]
                cost, peaks = res[i] if pool else evaluate(prob, bs, config, method, mode)
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i] = cost
                    pbest[i] = particles[i]
                if cost < gbest:
                    gbest, gbest_particle = cost, particles[i].copy()
                if cost < best["cost"]:
                    best = {"cost": cost, "config": config, "n_caps": n_caps,
                            "iter": iters_total + 1}
                if note(peaks, config, n_caps):          # immediate stop at the tightest level
                    best = {"cost": cost, "config": config, "n_caps": n_caps,
                            "iter": iters_total + 1}
                    stop = True
                    break
            hist.append(gbest)
            iters_total += 1
            if stop:
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
        curve_cost.append(gbest)

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
    success = tight in hits
    levels_out = {}
    for lab, mult, vec in prob.levels:
        h = hits.get(lab)
        levels_out[lab] = {
            "multiplier": mult, "target_ohm": vec.tolist(), "hit": h is not None,
            "caps_needed": h["caps_needed"] if h else None,
            "iter": h["iter"] if h else None,
            "n_evals": h["n_evals"] if h else None,
            "time_s": h["time_s"] if h else None,
            "placement": h["placement"] if h else None,
        }
    return {
        "objective": prob.objective,
        "cost_unit": "ratio" if prob.objective == "ratio" else "ohm",
        "success": bool(success),
        "bare_pass": bool(bare_pass),
        "timed_out": bool(timed_out),
        "caps_needed": hits[tight]["caps_needed"] if success else None,
        "caps_used": best["n_caps"],
        "best_cost_own_band": best["cost"],
        "conv_iter": hits[tight]["iter"] if success else best["iter"],
        "total_iters": iters_total,
        "n_evals": n_evals,
        "wall_s": wall,
        "time_to_target_s": hits[tight]["time_s"] if success else None,
        "max_caps": max_caps,
        "levels": levels_out,
        "curve_caps": curve_caps,
        "curve_cost": curve_cost,
        "histories": histories,
        "problem": prob.name,
        "placement": placement_records(prob, config),    # model + pad for every capacitor
        "rescored": score_bands(prob, config),
    }
