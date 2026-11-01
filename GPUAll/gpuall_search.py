"""
gpuall_search.py -- the search-quality arms of GPUAll (they change the search, every score stays exact on the tabulated library).

    pso_polish   AllCPU's PSO step for step (same seeding / warm start / velocity update / immediate stop / patience / time limit), evaluated
                 with the GPU cache, plus a gradient POLISH of the stage best at the end of every unfinished stage whose best is within
                 POLISH_RATIO of the target (or every stage when there is no target):
                   capacitors -> continuous series R-L-C (fit of each library part, phi = 0 reproduces the library curve exactly),
                   Adam through the exact multi-port Woodbury cost (smooth max over the ranking band and the observation ports),
                   snap to the nearest real parts in normalised log(R, L, 1/C) space, exact coordinate sweep over SNAP_CAND neighbours.
                   Accepted only if the exact cost beats the PSO stage best; it then seeds the next stage's warm start.
    ga           genetic algorithm with the PSO's staging, seeding, warm start (core.seed_particles), budget (particles x iterations per
                 capacitor count), immediate stop and patience: tournament (3), uniform crossover (0.9), Gaussian mutation (rate 1/dim,
                 sigma GA_SIGMA), 2 elites.
    ga_polish    ga + the same polish.
Records have AllCPU's fields (run_pso) plus polish statistics, so compare_gpuall.py treats every arm alike.
"""
import os
import time

import numpy as np
import torch

from gpuall_eval import DEV, CD, sync

POLISH_RATIO = float(os.environ.get("POLISH_RATIO", "1.3"))
POLISH_STEPS = int(os.environ.get("POLISH_STEPS", "120"))
POLISH_LR = float(os.environ.get("POLISH_LR", "0.05"))
PHI_BOUND = float(os.environ.get("PHI_BOUND", "2.0"))
SNAP_CAND = int(os.environ.get("SNAP_CAND", "8"))
SWEEPS = int(os.environ.get("SWEEPS", "2"))
GA_SIGMA = float(os.environ.get("GA_SIGMA", "0.1"))
SEARCH_ARMS = ("pso_polish", "ga", "ga_polish")


# ------------------------------------------------------------------ cached exact batch evaluation
class CachedBatch:
    def __init__(self, ev):
        self.ev = ev
        self.cache = {}
        self.n_full = self.n_hits = 0

    def __call__(self, configs):
        out, todo, keys = [None] * len(configs), [], []
        for i, cf in enumerate(configs):
            kk = tuple(sorted((int(p), int(c)) for c, p in cf))
            keys.append(kk)
            if kk in self.cache:
                out[i] = self.cache[kk]
                self.n_hits += 1
            else:
                todo.append(i)
        if todo:
            res = self.ev.peaks_batch([configs[i] for i in todo])
            self.n_full += len(todo)
            for i, r in zip(todo, res):
                out[i] = r
                self.cache[keys[i]] = r
        return out


# ------------------------------------------------------------------ R-L-C fit + polish
def fit_rlc(D, freq):
    """weighted LS series R-L-C fit of Z = 1/D for every part, on the device. Returns R, L, Ci (=1/C) and the median relative |Z| error."""
    om = torch.as_tensor(2 * np.pi * np.asarray(freq, float), device=DEV)
    Z = 1.0 / D
    w = 1.0 / Z.abs() ** 2
    R = (w * Z.real).sum(1) / w.sum(1)
    a, b = om[None, :], -1.0 / om[None, :]
    saa, sab, sbb = (w * a * a).sum(1), (w * a * b).sum(1), (w * b * b).sum(1)
    sya, syb = (w * Z.imag * a).sum(1), (w * Z.imag * b).sum(1)
    det = saa * sbb - sab ** 2
    L = ((sya * sbb - syb * sab) / det).clamp(min=1e-13)
    Ci = ((saa * syb - sab * sya) / det).clamp(min=1e-3)
    R = R.clamp(min=1e-5)
    Zf = R[:, None] + 1j * om[None, :] * L[:, None] + Ci[:, None] / (1j * om[None, :])
    err = ((Zf.abs() - Z.abs()).abs() / Z.abs()).median(dim=1).values
    return R, L, Ci, err


class Polisher:
    def __init__(self, ev):
        self.ev, self.core = ev, ev.core
        f = ev.freq
        self.om_all = torch.as_tensor(2 * np.pi * f, device=DEV)
        R, L, Ci, err = fit_rlc(ev.D, f)
        self.R, self.L, self.Ci, self.fit_err = R, L, Ci, err
        lp = torch.stack([R.log(), L.log(), Ci.log()], 1)
        self.sd = lp.std(0)
        self.lib = lp / self.sd
        self.log = []

    def _relaxed(self, phi, m, A0, Rr, Lr, zoo, Zc0, om, w):
        e = torch.exp(phi)
        dZ = ((self.R[m] * (e[:, 0] - 1)).unsqueeze(1) + 1j * om.unsqueeze(0) * (self.L[m] * (e[:, 1] - 1)).unsqueeze(1)
              + (self.Ci[m] * (e[:, 2] - 1)).unsqueeze(1) / (1j * om.unsqueeze(0)))          # (k,Fb)
        M = A0 + torch.diag_embed((Zc0 + dZ).t())
        X = torch.linalg.solve(M, Rr)
        val = zoo - (Lr * X.transpose(-1, -2)).sum(-1)
        a = val.abs()
        return a if w is None else a * w

    def polish(self, models, ports):
        """models, ports: lists. Returns (exact cost, models, ports) of the best of {input, nearest snap, local sweep}."""
        t0 = time.perf_counter()
        ev, core = self.ev, self.core
        no = ev.n_obs
        fb = torch.from_numpy(np.nonzero(core.band_mask(core.RANK_BAND))[0]).to(DEV)
        m = torch.tensor(models, device=DEV)
        p = torch.tensor(ports, device=DEV)
        Z0b = ev.Z0.index_select(0, fb)
        A0 = Z0b[:, p.unsqueeze(1), p.unsqueeze(0)]
        Rr = Z0b[:, p, :no]
        Lr = Z0b[:, :no, p]
        zoo = ev.zoo.index_select(0, fb)
        Zc0 = ev.DINV[m][:, fb]
        om = self.om_all.index_select(0, fb)
        w = ev.weights()
        k = len(models)
        c_in = float(ev.cost_batch([models], [ports])[0])
        phi = torch.zeros(k, 3, device=DEV, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=POLISH_LR)
        best, phi_best = c_in, phi.detach().clone()
        per = max(1, POLISH_STEPS // 3)
        for beta in (30.0, 100.0, 300.0):
            for _ in range(per):
                opt.zero_grad()
                a = self._relaxed(phi, m, A0, Rr, Lr, zoo, Zc0, om, w)
                hard = float(a.detach().max())
                if hard < best:
                    best, phi_best = hard, phi.detach().clone()
                loss = torch.logsumexp(beta * a.flatten() / c_in, dim=0) / beta
                loss.backward()
                opt.step()
                with torch.no_grad():
                    phi.clamp_(-PHI_BOUND, PHI_BOUND)
        tgt = torch.stack([self.R[m].log() + phi_best[:, 0], self.L[m].log() + phi_best[:, 1],
                           self.Ci[m].log() + phi_best[:, 2]], 1) / self.sd
        near = torch.cdist(tgt, self.lib).topk(min(SNAP_CAND, ev.C), dim=1, largest=False).indices.cpu().numpy()
        snap = near[:, 0].tolist()
        c_snap = float(ev.cost_batch([snap], [ports])[0])
        cur, c_cur = (list(snap), c_snap) if c_snap < c_in else (list(models), c_in)
        n_ev = 2
        for _ in range(SWEEPS):
            improved = False
            for j in range(k):
                cands = sorted(set(near[j].tolist()) | {int(models[j]), int(snap[j])})
                cfg = np.tile(np.array(cur), (len(cands), 1))
                cfg[:, j] = cands
                cs = ev.cost_batch(cfg, np.tile(np.array(ports), (len(cands), 1)))
                n_ev += len(cands)
                b = int(cs.argmin())
                if cs[b] < c_cur * (1 - 1e-12):
                    c_cur, cur, improved = float(cs[b]), cfg[b].tolist(), True
            if not improved:
                break
        sync()
        self.log.append({"k": k, "in": c_in, "relaxed": best, "snap": c_snap, "out": c_cur, "evals": n_ev,
                         "time_s": time.perf_counter() - t0})
        return c_cur, cur, list(ports), n_ev


# ------------------------------------------------------------------ staged search (PSO replica / GA)
def run_search(cell, ev, arm, run_id, out_folder, results_path, threshold=None, level=None, polisher=None, out_root=None):
    core, prob = cell.core, ev.prob
    os.makedirs(out_folder, exist_ok=True)
    log = core.setup_logger(os.path.join(out_folder, f"run_{run_id}.log"))
    NP, NI = core.N_PARTICLES, core.N_ITERATIONS
    cap_ceiling = prob.max_caps
    n_warm_eff = core.n_warm_count()
    use_polish = arm.endswith("polish")
    is_ga = arm.startswith("ga")
    log.info("RUN_START | prob=%s | N=%d n_obs=%d pads=%d | method=%s | run=%d | particles=%d iters=%d seed=%d warm=%d/%d jitter=%.4f | "
             "band=%s | threshold=%s | max_caps=%d | device=%s", prob.name, prob.N, prob.n_obs, len(prob.pads), arm, run_id, NP, NI,
             core.BASE_SEED, n_warm_eff, NP, core.WARM_START_JITTER, core.RANK_BAND, threshold, cap_ceiling, DEV)
    evalb = CachedBatch(ev)
    if polisher is not None:
        polisher.log.clear()
    bare = core.cost_of(core.bare_peaks(prob, "woodbury_batch"), core.RANK_BAND)
    caps_curve, minz_curve, histories = [], [], {}
    best = {"cost": np.inf, "config": [], "n_caps": 0, "peaks": None}
    sync()
    t0 = time.perf_counter()
    time_to_target = caps_at_target = None
    n_evals, n_polish_evals = 0, 0
    timed_out = stagnated = False
    ref_best, no_gain = np.inf, 0
    prev_best = None

    def decode_all(X, n):
        cfgs = []
        for i in range(X.shape[0]):
            mm, pp = core.decode_particle(X[i], n, prob)
            mm, pp = core.resolve_adjacent_ports(mm, pp, prob)
            cfgs.append([(int(a), int(b)) for a, b in zip(mm, pp)])
        return cfgs

    for n_caps in range(1, cap_ceiling + 1):
        np.random.seed(core.BASE_SEED + run_id * 10000 + n_caps)
        X = core.seed_particles(n_caps, prev_best)
        state = {"gbest": np.inf, "gpart": None, "stop": False}
        history = []

        def score(Xs, n=n_caps):
            """evaluate in particle order with immediate stop; returns costs (inf after a stop)"""
            nonlocal n_evals, best
            cfgs = decode_all(Xs, n)
            pks = evalb(cfgs)
            cs = np.full(len(cfgs), np.inf)
            for i, pk in enumerate(pks):
                c = core.cost_of(pk, core.RANK_BAND)
                cs[i] = c
                n_evals += 1
                if c < state["gbest"]:
                    state["gbest"], state["gpart"] = c, Xs[i].copy()
                if c < best["cost"]:
                    best = {"cost": c, "config": cfgs[i], "n_caps": n, "peaks": pk}
                if threshold is not None and state["gbest"] <= threshold:
                    state["stop"] = True
                    break
            return cs

        if not is_ga:                                                  # ---- PSO, step for step as allcpu_core.run_pso
            V = np.zeros_like(X)
            pbest, pbest_val = X.copy(), np.full(NP, np.inf)
            for it in range(NI):
                cs = score(X)
                upd = cs < pbest_val
                pbest_val[upd], pbest[upd] = cs[upd], X[upd]
                history.append(state["gbest"])
                if state["stop"] or time.perf_counter() - t0 > core.TIME_LIMIT_S:
                    timed_out = not state["stop"]
                    break
                w = core.W_MAX - (core.W_MAX - core.W_MIN) * (it / NI)
                V = (w * V + core.C1 * np.random.rand(*X.shape) * (pbest - X)
                     + core.C2 * np.random.rand(*X.shape) * (state["gpart"] - X))
                X = np.clip(X + V, 0.0, 1.0)
        else:                                                          # ---- GA
            val = score(X)
            history.append(state["gbest"])
            dim = 2 * n_caps
            for it in range(1, NI):
                if state["stop"] or time.perf_counter() - t0 > core.TIME_LIMIT_S:
                    timed_out = not state["stop"]
                    break
                order = np.argsort(val)
                elite = X[order[:2]].copy()

                def tour():
                    c = np.random.randint(0, NP, 3)
                    return X[c[np.argmin(val[c])]]
                kids = []
                while len(kids) < NP - 2:
                    a, b = tour(), tour()
                    if np.random.rand() < 0.9:
                        a = np.where(np.random.rand(dim) < 0.5, a, b)
                    mu = np.random.rand(dim) < 1.0 / dim
                    kids.append(np.clip(a + mu * np.random.normal(0, GA_SIGMA, dim), 0, 1))
                kids = np.array(kids)
                kv = score(kids)
                X = np.vstack([elite, kids])
                val = np.concatenate([val[order[:2]], kv])
                history.append(state["gbest"])
        if state["stop"] and time_to_target is None:
            sync()
            time_to_target, caps_at_target = time.perf_counter() - t0, n_caps

        gb, gp = state["gbest"], state["gpart"]
        # ---- polish at the end of an unfinished stage
        if use_polish and gp is not None and not state["stop"] and not timed_out and \
                (threshold is None or gb <= POLISH_RATIO * threshold):
            mm, pp = core.decode_particle(gp, n_caps, prob)
            mm, pp = core.resolve_adjacent_ports(mm, pp, prob)
            c_new, m_new, p_new, ne = polisher.polish([int(x) for x in mm], [int(x) for x in pp])
            n_polish_evals += ne
            if c_new < gb * (1 - 1e-12):
                gb = c_new
                gp = np.concatenate([(np.array(m_new) + 0.5) / ev.C,
                                     (np.searchsorted(prob.pads, np.array(p_new)) + 0.5) / len(prob.pads)])
                cfg = [(int(a), int(b)) for a, b in zip(m_new, p_new)]
                if c_new < best["cost"]:
                    best = {"cost": c_new, "config": cfg, "n_caps": n_caps, "peaks": ev.peaks_batch([cfg])[0]}
                history.append(gb)
                if threshold is not None and gb <= threshold:
                    state["stop"] = True
                    sync()
                    time_to_target, caps_at_target = time.perf_counter() - t0, n_caps

        histories[n_caps] = history
        if gp is None:
            prev_best = None
            continue
        prev_best = gp
        caps_curve.append(n_caps)
        minz_curve.append(gb)
        log.info("n_caps=%d | minZ=%.6g", n_caps, gb)
        if gb < ref_best * (1.0 - core.PATIENCE_TOL):
            ref_best, no_gain = gb, 0
        else:
            no_gain += 1
        if core.PATIENCE is not None and not state["stop"] and no_gain >= core.PATIENCE:
            stagnated = True
            break
        if state["stop"] or timed_out:
            break

    sync()
    wall = time.perf_counter() - t0
    pk = best["peaks"] if best["peaks"] is not None else core.bare_peaks(prob, "woodbury_batch")
    bare_all = core.bare_peaks(prob, "woodbury_batch")
    rec = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problem": prob.name, "N": prob.N, "n_obs": prob.n_obs, "n_pads": len(prob.pads), "max_caps": cap_ceiling,
        "method": arm, "run_id": run_id, "seed_base": core.BASE_SEED, "rank_band": core.RANK_BAND,
        "threshold_ohm": threshold, "threshold_level": level,
        "bare_peak_ohm": {b: float(np.max(bare_all[b])) for b in core.BANDS},
        "best_peak_ohm": {b: float(np.max(pk[b])) for b in core.BANDS},
        "best_peak_per_obs_ohm": {b: [float(x) for x in pk[b]] for b in core.BANDS},
        "best_n_caps": best["n_caps"], "placement": core.placement_records(best["config"]),
        "curve_caps": caps_curve, "curve_peak_ohm": minz_curve,
        "time_to_target_s": time_to_target, "caps_at_target": caps_at_target, "success": time_to_target is not None,
        "timed_out": timed_out, "stagnated": stagnated, "best_cost": float(best["cost"]),
        "objective": "ratio" if core.OBJ_WEIGHTS is not None else "ohm",
        "obj_weights": None if core.OBJ_WEIGHTS is None else [float(x) for x in core.OBJ_WEIGHTS],
        "total_wall_time_s": wall, "n_fitness_evals": n_evals, "n_polish_evals": n_polish_evals,
        "n_warm_effective": n_warm_eff, "device": str(DEV), "gpu_full_evals": evalb.n_full, "cache_hits": evalb.n_hits,
        "n_polish": len(polisher.log) if (use_polish and polisher) else 0,
        "polish_time_s": float(sum(d["time_s"] for d in polisher.log)) if (use_polish and polisher) else 0.0,
        "folder": os.path.relpath(out_folder, out_root) if out_root else out_folder,
    }
    log.info("RESULT | best=%.6g @ %d caps | wall=%.1fs | evals=%d (+%d polish)", best["cost"], best["n_caps"], wall, n_evals, n_polish_evals)
    core.append_record(results_path, rec)
    core.plot_run_curve(caps_curve, minz_curve, prob, arm, run_id, out_folder, threshold)
    core.plot_convergence(histories, prob, out_folder, threshold)
    if best["config"]:
        core.plot_impedance(prob, best["config"], out_folder)
    for h in list(log.handlers):                      # close the run log (one leaked handle per run otherwise)
        h.close()
        log.removeHandler(h)
    return rec
