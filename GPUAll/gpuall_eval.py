"""
gpuall_eval.py -- GPU fitness for the AllCPU PDNs (pdn1..pdn4), plugged into AllCPU's own PSO.

AllCPU (../AllCPU/common/allcpu_core.py) evaluates one placement with peaks_all_bands(prob, config, method) -> {band: (n_obs,) peak |Z_kk|}
and, when a `pool` is passed to run_pso, evaluates a whole iteration with pool.map(...) and consumes the results in particle order.
This module
  * replaces core.peaks_all_bands with a dispatcher: the CPU methods (numpy, solve, sm, iterative) still go to the original code, the GPU
    methods go to GpuEval;
  * provides GpuPool, whose map() evaluates the whole swarm in ONE batched GPU call.
Because the PSO itself is AllCPU's (same seeding, decoding, warm start, patience, stop rule, records), every exact GPU arm follows exactly
the trajectory of AllCPU's CPU numpy runs: GPU and CPU records can be compared run by run.

Exact GPU arms
    gpu_numpy       full inverse of the assembled Y at every frequency, one placement at a time (the CPU 'numpy' method on the GPU)
    woodbury_batch  multi-port Woodbury, whole swarm per call:
                        Z_OO = Z0[O,O] - Z0[O,P] (diag(1/d_P) + Z0[P,P])^-1 Z0[P,O]     (O = observation ports, P = pads, k x k solve)
                    all observation ports (MPHY: 5) and all bands from one solve
    fast            woodbury_batch + per-run cache of placements + exact lower-bound pruning: a particle is solved on all frequencies of the
                    ranking band only if its peak on a frequency subset (a lower bound) is below its personal best; otherwise it cannot change
                    pbest / gbest / the run best, so its exact value is never needed. pbest is read from AllCPU's run_pso frame.
"""
import os
import sys
import time

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CD = torch.complex128
GPU_METHODS = ("gpu_numpy", "woodbury_batch", "fast")
MAX_ELEMS = float(os.environ.get("GPUALL_MAX_ELEMS", "4e7"))      # b * F * k * k complex elements per batched solve
STRIDE = int(os.environ.get("GPUALL_STRIDE", "16"))


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


class GpuEval:
    """Device copy of one AllCPU Problem + the active library."""

    def __init__(self, core, prob):
        self.core, self.prob = core, prob
        lib = core.library()
        self.n_obs, self.F, self.N = prob.n_obs, prob.F, prob.N
        self.Y = torch.from_numpy(np.ascontiguousarray(prob.y)).to(DEV, CD)
        self.Z0 = torch.linalg.inv(self.Y)
        D = torch.from_numpy(np.ascontiguousarray(lib["d"], dtype=np.complex128)).to(DEV, CD)
        self.D = D
        self.DINV = 1.0 / torch.where(D.abs() < 1e-300, torch.full_like(D, 1e-300), D)
        self.C = D.shape[0]
        no = self.n_obs
        self.zoo = torch.diagonal(self.Z0[:, :no, :no], dim1=1, dim2=2)          # (F, n_obs)
        self.freq = np.asarray(lib["freq"], float)
        self._masks = {}
        sync()

    # ---------------- masks / weights
    def mask(self, band):
        if band not in self._masks:
            self._masks[band] = torch.from_numpy(self.core.band_mask(band)).to(DEV)
        return self._masks[band]

    def weights(self):
        w = self.core.OBJ_WEIGHTS
        return None if w is None else torch.as_tensor(np.asarray(w, float), device=DEV)

    # ---------------- kernels
    def obs_abs(self, m_t, p_t, fidx=None):
        """|Z_kk(f)| for B placements (B,k): (B, F', n_obs) float64 and a bad flag (B,). Exact Woodbury, chunked."""
        no = self.n_obs
        Z0f = self.Z0 if fidx is None else self.Z0.index_select(0, fidx)
        Dv = self.DINV if fidx is None else self.DINV.index_select(1, fidx)
        zoo = self.zoo if fidx is None else self.zoo.index_select(0, fidx)
        B, k = p_t.shape
        Fp = Z0f.shape[0]
        if k == 0:
            a = zoo.abs().unsqueeze(0).expand(B, -1, -1)
            return a, torch.zeros(B, dtype=torch.bool, device=DEV)
        ch = max(1, int(MAX_ELEMS // max(1, Fp * k * max(k, no))))
        outs, bads = [], []
        for s in range(0, B, ch):
            p, m = p_t[s:s + ch], m_t[s:s + ch]
            M = Z0f[:, p.unsqueeze(2), p.unsqueeze(1)].permute(1, 0, 2, 3) + torch.diag_embed(Dv[m].transpose(1, 2))
            R = Z0f[:, p, :no].permute(1, 0, 2, 3)                                 # (b,F',k,no)
            L = Z0f[:, :no, p].permute(2, 0, 1, 3)                                 # (b,F',no,k)
            X, info = torch.linalg.solve_ex(M, R)
            val = zoo.unsqueeze(0) - (L * X.transpose(-1, -2)).sum(-1)             # (b,F',no)
            a = val.abs()
            outs.append(a)
            bads.append((info != 0).any(dim=1) | ~torch.isfinite(a).all(dim=2).all(dim=1))
        return torch.cat(outs), torch.cat(bads)

    def obs_abs_numpy(self, models, ports):
        """gpu_numpy: assemble Y and invert it (full N x N) at every frequency for ONE placement -> (F, n_obs)."""
        A = self.Y.clone()
        for c, p in zip(models, ports):
            A[:, p, p] += self.D[c]
        Z, info = torch.linalg.inv_ex(A)
        a = torch.diagonal(Z[:, :self.n_obs, :self.n_obs], dim1=1, dim2=2).abs()
        return a, bool((info != 0).any()) or not bool(torch.isfinite(a).all())

    # ---------------- AllCPU-shaped results
    def band_peaks(self, a, bad):
        """a (B,F,no) -> list of {band: np.array(no)} like core.peaks_all_bands."""
        out = [dict() for _ in range(a.shape[0])]
        for b in self.core.BANDS:
            pk = a[:, self.mask(b), :].amax(dim=1).cpu().numpy()
            for i in range(a.shape[0]):
                out[i][b] = pk[i]
        badn = bad.cpu().numpy()
        for i in np.nonzero(badn)[0]:
            out[i] = {b: np.full(self.n_obs, 1e200) for b in self.core.BANDS}
        return out

    def peaks_batch(self, configs):
        if not configs:
            return []
        k = len(configs[0])
        if any(len(c) != k for c in configs):                       # mixed sizes (not produced by the PSO): one call per size
            return [self.peaks_batch([c])[0] for c in configs]
        m = torch.tensor([[c for c, _ in cf] for cf in configs], device=DEV, dtype=torch.long).reshape(len(configs), k)
        p = torch.tensor([[q for _, q in cf] for cf in configs], device=DEV, dtype=torch.long).reshape(len(configs), k)
        a, bad = self.obs_abs(m, p)
        return self.band_peaks(a, bad)

    def peaks_one(self, config, method):
        if method == "gpu_numpy":
            a, bad = self.obs_abs_numpy([c for c, _ in config], [p for _, p in config])
            return self.band_peaks(a.unsqueeze(0), torch.tensor([bad], device=DEV))[0]
        return self.peaks_batch([list(config)])[0]

    def cost_curve(self, a):
        """(B,F',no) -> weighted max over observation ports (B,F')"""
        w = self.weights()
        return (a if w is None else a * w).amax(dim=-1)

    def cost_batch(self, models, ports):
        """exact objective of placements (B,k) on the current RANK_BAND (np array)."""
        m = torch.as_tensor(np.asarray(models), device=DEV, dtype=torch.long)
        p = torch.as_tensor(np.asarray(ports), device=DEV, dtype=torch.long)
        a, bad = self.obs_abs(m, p)
        c = self.cost_curve(a)[:, self.mask(self.core.RANK_BAND)].amax(dim=1)
        c = torch.where(bad, torch.full_like(c, 1e200), c)
        return c.cpu().numpy()


# ------------------------------------------------------------------ dispatcher installed into allcpu_core
_EVALS = {}


def install(core):
    if getattr(core, "_gpuall_installed", False):
        return
    orig = core.peaks_all_bands

    def peaks_all_bands(prob, config, method):
        if method in GPU_METHODS:
            return _EVALS[id(prob)].peaks_one(config, method)
        return orig(prob, config, method)

    core._orig_peaks_all_bands = orig
    core.peaks_all_bands = peaks_all_bands
    core._gpuall_installed = True


def register(core, prob):
    install(core)
    ev = GpuEval(core, prob)
    _EVALS[id(prob)] = ev
    return ev


class GpuPool:
    """Passed as `pool` to allcpu_core.run_pso: map() evaluates the whole swarm in one call. arm = woodbury_batch | fast."""

    def __init__(self, ev, arm):
        self.ev, self.arm = ev, arm
        self.cache = {}
        self.n_full = self.n_screen = self.n_pruned = self.n_hits = 0
        self._sub = None
        self._sub_band = None

    def _subset(self):
        core, ev = self.ev.core, self.ev
        band = core.RANK_BAND
        if self._sub_band != band:
            idx = np.nonzero(core.band_mask(band))[0]
            bare = ev.cost_curve(ev.zoo.abs().unsqueeze(0))[0].cpu().numpy()[idx]
            lm = [i for i in range(1, len(idx) - 1) if bare[i] >= bare[i - 1] and bare[i] >= bare[i + 1]]
            lm = sorted(lm, key=lambda i: -bare[i])[:16]
            sel = set(range(0, len(idx), STRIDE)) | {0, len(idx) - 1} | set(lm)
            self._sub = set(int(idx[i]) for i in sel)
            self._sub_t, self._sub_band = None, band
        if self._sub_t is None:
            self._sub_t = torch.tensor(sorted(self._sub), device=DEV)
        return self._sub_t

    def map(self, fn, configs):
        configs = list(configs)
        ev, core = self.ev, self.ev.core
        if self.arm == "woodbury_batch":
            self.n_full += len(configs)
            return ev.peaks_batch(configs)
        # ---- fast: cache + exact pruning against the caller's personal bests
        pbest = None
        fr = sys._getframe(1)
        while fr is not None and pbest is None:
            pbest = fr.f_locals.get("pbest_val") if fr.f_code.co_name == "run_pso" else None
            fr = fr.f_back
        out = [None] * len(configs)
        keys = [tuple(sorted((int(p), int(c)) for c, p in cf)) for cf in configs]
        todo = []
        for i, kk in enumerate(keys):
            if kk in self.cache:
                out[i] = self.cache[kk]
                self.n_hits += 1
            else:
                todo.append(i)
        if not todo:
            return out
        k = len(configs[todo[0]])
        m = torch.tensor([[c for c, _ in configs[i]] for i in todo], device=DEV, dtype=torch.long).reshape(len(todo), k)
        p = torch.tensor([[q for _, q in configs[i]] for i in todo], device=DEV, dtype=torch.long).reshape(len(todo), k)
        need = np.ones(len(todo), bool)
        if pbest is not None and k > 0 and np.isfinite(pbest).any():
            fidx = self._subset()
            a, bad = ev.obs_abs(m, p, fidx)
            lb = ev.cost_curve(a).amax(dim=1)
            lb = torch.where(bad, torch.full_like(lb, 1e200), lb).cpu().numpy()
            self.n_screen += len(todo)
            pb = np.asarray([pbest[i] for i in todo])
            need = lb < pb * (1 + 1e-9)
            w = None if core.OBJ_WEIGHTS is None else np.asarray(core.OBJ_WEIGHTS, float)
            for j in np.nonzero(~need)[0]:                    # lower bound >= pbest: cannot update anything, exact value never used
                pk = np.full(ev.n_obs, lb[j]) if w is None else lb[j] / w      # so that cost_of(pk) == lb
                out[todo[j]] = {b: pk for b in core.BANDS}
            self.n_pruned += int((~need).sum())
        sel = np.nonzero(need)[0]
        if len(sel):
            st = torch.from_numpy(sel).to(DEV)
            a, bad = ev.obs_abs(m.index_select(0, st), p.index_select(0, st))
            res = ev.band_peaks(a, bad)
            self.n_full += len(sel)
            mrank = ev.mask(core.RANK_BAND)
            am = ev.cost_curve(a)[:, mrank].argmax(dim=1).cpu().numpy()
            ridx = np.nonzero(core.band_mask(core.RANK_BAND))[0]
            if self._sub is not None:
                new = set(int(ridx[x]) for x in am) - self._sub
                if new:
                    self._sub |= new
                    self._sub_t = None
            for j, r in zip(sel, res):
                out[todo[j]] = r
                self.cache[keys[todo[j]]] = r
        return out

    def stats(self):
        return {"gpu_full_evals": self.n_full, "gpu_screen_evals": self.n_screen, "pruned": self.n_pruned, "cache_hits": self.n_hits}


def verify(core, prob, ev, n=6, seed=11):
    """max relative deviation of the GPU arms vs AllCPU's CPU numpy on random placements, every band (cost units)."""
    rng = np.random.default_rng(seed)
    worst = {m: 0.0 for m in GPU_METHODS}
    for _ in range(n):
        k = int(rng.integers(1, min(8, len(prob.pads)) + 1))
        mdl, prt = core.decode_particle(rng.random(2 * k), k, prob)
        mdl, prt = core.resolve_adjacent_ports(mdl, prt, prob)
        cfg = [(int(a), int(b)) for a, b in zip(mdl, prt)]
        ref = core._orig_peaks_all_bands(prob, cfg, "numpy")
        for meth in GPU_METHODS:
            got = ev.peaks_one(cfg, "gpu_numpy" if meth == "gpu_numpy" else "woodbury_batch")
            for b in core.BANDS:
                worst[meth] = max(worst[meth], float(np.max(np.abs(got[b] - ref[b]) / np.maximum(ref[b], 1e-30))))
    return worst
