"""
Experiment 5 -- gradient polish of the PSO stage best.

Each library capacitor is also described by a series R-L-C fit (weighted least squares on 1/D over frequency). At the end of a PSO
stage (when its best is within POLISH_RATIO of the target and the target is not yet met) the stage best is relaxed to continuous
parameters and refined by gradient descent through the exact Woodbury cost (torch autograd, complex128):

    Z_j(f; phi) = Z_lib[m_j](f) + (R*-R_fit) + j w (L*-L_fit) + (Ci*-Ci_fit)/(j w),   (R*,L*,Ci*) = fit[m_j] * exp(phi_j)
    (exactly the library model at phi = 0, so the relaxed start equals the PSO best),
    objective: log-sum-exp smooth max of |Z11(f)| with a temperature ramp, Adam on phi, the exact (hard) max is tracked and the best
    phi is kept.
Snap: every relaxed capacitor goes to the nearest library model in normalised log(R, L, 1/C) space ('nearest'); an optional
exact coordinate sweep over the SNAP_CAND nearest candidates per pad ('local') follows. Everything is scored EXACTLY with the
library models. The returned stage best is the best of {PSO best, nearest, local}: never worse than the PSO best. A polished
placement becomes the warm start of the next stage; if it meets the target the run stops with fewer capacitors.

Arms: fast (PSO only, reference), polish (PSO + polish at the end of qualifying stages). Same paired seeds.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync

EXPERIMENT = "gradient_polish"
POLISH_RATIO = float(os.environ.get("POLISH_RATIO", "1.3"))
POLISH_STEPS = int(os.environ.get("POLISH_STEPS", "90" if S["QUICK"] else "180"))
POLISH_LR = float(os.environ.get("POLISH_LR", "0.05"))
PHI_BOUND = float(os.environ.get("PHI_BOUND", "2.0"))
SNAP_CAND = int(os.environ.get("SNAP_CAND", "8"))
SWEEPS = int(os.environ.get("SWEEPS", "2"))


def fit_rlc(d, omega):
    """Weighted LS series R-L-C fit of Z = 1/D: (R, L, Ci=1/C) per model, plus median/max relative |Z| error."""
    Z = 1.0 / d
    w = 1.0 / np.abs(Z) ** 2
    R = (w * Z.real).sum(1) / w.sum(1)
    # imag: Im Z ~ w L - Ci / w   (omega)
    a, b = omega[None, :], -1.0 / omega[None, :]
    saa, sab, sbb = (w * a * a).sum(1), (w * a * b).sum(1), (w * b * b).sum(1)
    sya, syb = (w * Z.imag * a).sum(1), (w * Z.imag * b).sum(1)
    det = saa * sbb - sab ** 2
    L = (sya * sbb - syb * sab) / det
    Ci = (saa * syb - sab * sya) / det
    R, L, Ci = np.clip(R, 1e-4, None), np.clip(L, 1e-12, None), np.clip(Ci, 1e-3, None)
    Zfit = R[:, None] + 1j * omega[None, :] * L[:, None] + Ci[:, None] / (1j * omega[None, :])
    err = np.abs(np.abs(Zfit) - np.abs(Z)) / np.abs(Z)
    return R, L, Ci, np.median(err, axis=1), err.max(axis=1)


class Polisher:
    def __init__(self, prob):
        self.prob = prob
        assert prob.freq is not None, "FREQ_PATH needed for the R-L-C fit"
        om = 2 * np.pi * prob.freq
        self.om = torch.from_numpy(om).to(DEVICE)
        R, L, Ci, med, mx = fit_rlc(prob.d_np, om)
        self.fit_med, self.fit_max = med, mx
        self.R, self.L, self.Ci = (torch.from_numpy(x).to(DEVICE) for x in (R, L, Ci))
        lp = np.log(np.stack([R, L, Ci], axis=1))
        self.lp_sd = lp.std(axis=0)
        self.lib = torch.from_numpy(lp / self.lp_sd).to(DEVICE)           # (C,3) normalised log parameters
        self.log = []

    def _relaxed_peak(self, phi, m_t, A0, zc, zr, z00, hard=False):
        R0, L0, C0 = self.R[m_t], self.L[m_t], self.Ci[m_t]
        e = torch.exp(phi)
        dZ = ((R0 * (e[:, 0] - 1)).unsqueeze(1) + 1j * self.om.unsqueeze(0) * (L0 * (e[:, 1] - 1)).unsqueeze(1)
              + (C0 * (e[:, 2] - 1)).unsqueeze(1) / (1j * self.om.unsqueeze(0)))                     # (k,F)
        Zc = self.prob.DINV[m_t] + dZ
        M = A0 + torch.diag_embed(Zc.t())
        x = torch.linalg.solve(M, zc)
        return (z00 - (zr @ x)[..., 0, 0]).abs()

    def _exact(self, models, ports):
        mt = torch.as_tensor(np.asarray(models), device=DEVICE).reshape(-1, len(models[0]) if np.ndim(models) > 1 else len(models))
        pt = torch.as_tensor(np.asarray(ports), device=DEVICE).reshape(mt.shape[0], -1)
        if pt.shape[0] != mt.shape[0]:
            pt = pt.expand(mt.shape[0], -1)
        return self.prob.wb_peak(mt, pt)[0].cpu().numpy()

    def polish(self, models, ports):
        """models, ports: lists (k). Returns dict with the exact costs of PSO best / nearest snap / local sweep, relaxed minimum."""
        prob = self.prob
        t0 = time.perf_counter()
        k = len(models)
        m_t = torch.tensor(models, device=DEVICE)
        p_t = torch.tensor(ports, device=DEVICE)
        _, zc, zr, z00 = prob.gather(m_t.unsqueeze(0), p_t.unsqueeze(0))
        A0 = prob.Z0[:, p_t.unsqueeze(1), p_t.unsqueeze(0)] if False else prob.Z0[:, p_t][:, :, p_t]   # (F,k,k)
        zc, zr = zc[0], zr[0]
        c0 = float(self._relaxed_peak(torch.zeros(k, 3, device=DEVICE, dtype=torch.float64), m_t, A0, zc, zr, z00).max())
        phi = torch.zeros(k, 3, device=DEVICE, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=POLISH_LR)
        best, phi_best = c0, phi.detach().clone()
        phases = [30.0, 100.0, 300.0]
        per = max(1, POLISH_STEPS // len(phases))
        for beta in phases:
            for _ in range(per):
                opt.zero_grad()
                a = self._relaxed_peak(phi, m_t, A0, zc, zr, z00)
                hard = float(a.max())
                if hard < best:
                    best, phi_best = hard, phi.detach().clone()
                loss = torch.logsumexp(beta * a / c0, dim=0) / beta
                loss.backward()
                opt.step()
                with torch.no_grad():
                    phi.clamp_(-PHI_BOUND, PHI_BOUND)
        with torch.no_grad():
            a = self._relaxed_peak(phi, m_t, A0, zc, zr, z00)
            if float(a.max()) < best:
                best, phi_best = float(a.max()), phi.detach().clone()
        # ---- snap: nearest library model to the relaxed parameters
        tgt = torch.stack([torch.log(self.R[m_t]) + phi_best[:, 0], torch.log(self.L[m_t]) + phi_best[:, 1],
                           torch.log(self.Ci[m_t]) + phi_best[:, 2]], dim=1) / torch.from_numpy(self.lp_sd).to(DEVICE)
        dist = torch.cdist(tgt, self.lib)                                  # (k,C)
        near = dist.topk(SNAP_CAND, dim=1, largest=False).indices          # (k,SNAP_CAND)
        snap = near[:, 0].tolist()
        c_pso = float(self._exact([models], [ports])[0])
        c_near = float(self._exact([snap], [ports])[0])
        # ---- local exact sweep over the nearest candidates (start from the better of PSO best / nearest snap)
        cur = list(snap) if c_near < c_pso else list(models)
        c_cur = min(c_near, c_pso)
        for _ in range(SWEEPS):
            improved = False
            for j in range(k):
                cands = sorted(set(near[j].tolist()) | {models[j], snap[j]})
                cfg = np.tile(np.array(cur), (len(cands), 1))
                cfg[:, j] = cands
                costs = self._exact(cfg, np.tile(np.array(ports), (len(cands), 1)))
                b = int(costs.argmin())
                if costs[b] < c_cur - 1e-15:
                    c_cur, cur, improved = float(costs[b]), cfg[b].tolist(), True
            if not improved:
                break
        cuda_sync()
        out = {"k": k, "pso": c_pso, "relaxed": best, "nearest": c_near, "local": c_cur, "time_s": time.perf_counter() - t0,
               "models_local": cur, "models_near": snap}
        self.log.append(out)
        return out


def make_hook(prob, pol, thr):
    valid = prob.valid_ports

    def hook(n_caps, gbest, gbest_particle, prev_best):
        if gbest > POLISH_RATIO * thr:
            return None
        m, p = prob.decode_particle(gbest_particle, n_caps)
        m, p = prob.resolve_adjacent_ports(m, p)
        r = pol.polish(m.tolist(), p.tolist())
        final = min(r["pso"], r["nearest"], r["local"])
        if final >= gbest:                              # never worse than the PSO best (exact cost re-evaluated in polish())
            return None
        use = r["models_local"] if r["local"] <= r["nearest"] else r["models_near"]
        frac_m = (np.array(use) + 0.5) / prob.C
        frac_p = (np.searchsorted(valid, np.array(p)) + 0.5) / len(valid)
        return final, np.concatenate([frac_m, frac_p])
    return hook


def main():
    run = C.Run(EXPERIMENT, __file__, {"POLISH_RATIO": POLISH_RATIO, "POLISH_STEPS": POLISH_STEPS, "POLISH_LR": POLISH_LR,
                                       "PHI_BOUND": PHI_BOUND, "SNAP_CAND": SNAP_CAND, "SWEEPS": SWEEPS})
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    pol = Polisher(prob)
    with open(os.path.join(run.run_dir, "summary", "rlc_fit.txt"), "w") as fh:
        fh.write(f"series R-L-C fit of the capacitor library ({prob.C} models, weighted LS on 1/D)\n"
                 f"median over models of the median relative |Z| error = {np.median(pol.fit_med):.4f}\n"
                 f"90th percentile of the median error = {np.percentile(pol.fit_med, 90):.4f}\n"
                 f"median of the max relative |Z| error = {np.median(pol.fit_max):.4f}\n")
    ev = C.SwarmEvaluator(prob, True, True)
    C.warmup(prob, [ev])
    pol.polish([0, 1], [int(prob.valid_ports[0]), int(prob.valid_ports[1])])        # touch autograd kernels (untimed)
    pol.log.clear()
    base = C.load_baseline()
    plog = []
    for thr in S["TARGETS"]:
        for name in ("fast", "polish"):
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", name, thr, r)
                pol.log.clear()
                try:
                    if name == "fast":
                        rec = C.run_pso(prob, ev, [thr], r, label=name)
                    else:
                        rec = C.run_pso(prob, ev, [thr], r, label=name, on_stage_end=make_hook(prob, pol, thr))
                except Exception as e:                                       # noqa: BLE001
                    run.fail(name, thr, r, e)
                    continue
                rec["polish_log"] = [{k: v for k, v in d.items() if not k.startswith("models")} for d in pol.log]
                rec["n_polish"] = len(pol.log)
                rec["polish_time_s"] = sum(d["time_s"] for d in pol.log)
                imp = [(d["pso"] - min(d["pso"], d["nearest"], d["local"])) / d["pso"] for d in pol.log]
                rec["polish_gain_mean"] = float(np.mean(imp)) if imp else None
                rec["polish_gain_max"] = float(np.max(imp)) if imp else None
                for d in pol.log:
                    plog.append({"threshold": thr, "run": r, **{k: v for k, v in d.items() if not k.startswith("models")}})
                # re-score the final placement with the exact model on the original grid
                if rec["best_models"] is not None:
                    rec["rescored_minz"] = float(prob.wb_peak(torch.tensor([rec["best_models"]], device=DEVICE),
                                                              torch.tensor([rec["best_ports"]], device=DEVICE))[0][0])
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, base.get((round(thr, 6), r)))
                run.add(rec)
                run.result_line(rec, polishes=rec["n_polish"], gain=rec["polish_gain_mean"])
    run.summarise(extra_cols=("n_polish", "polish_time_s", "polish_gain_mean", "polish_gain_max", "rescored_minz"))
    import pandas as pd
    import json
    json.dump(plog, open(os.path.join(run.run_dir, "output", "polish_events.json"), "w"))
    lines = []
    if plog:
        pdf = pd.DataFrame(plog)
        lines.append(f"polish_events = {len(pdf)}")
        for c in ("relaxed", "nearest", "local"):
            g = (pdf.pso - pdf[c]) / pdf.pso
            lines.append(f"{c}: mean rel. improvement over PSO stage best = {g.mean():.4f} | max = {g.max():.4f} | "
                         f"fraction of events better than PSO = {(pdf[c] < pdf.pso * (1 - 1e-12)).mean():.3f}")
        fin = pdf[["pso", "nearest", "local"]].min(axis=1)
        lines.append(f"final (best of PSO/nearest/local) mean rel. improvement = {((pdf.pso - fin) / pdf.pso).mean():.4f}; "
                     f"worse-than-PSO events (must be 0) = {int((fin > pdf.pso * (1 + 1e-12)).sum())}")
        lines.append(f"mean polish time = {pdf.time_s.mean():.3f}s")
    df = pd.DataFrame(run.records)
    f_, p_ = df[df.method == "fast"], df[df.method == "polish"]
    if len(f_) and len(p_):
        lines.append(f"success: fast {f_.success.mean():.3f} polish {p_.success.mean():.3f}")
        lines.append(f"mean caps_at_target: fast {f_.caps_at_target.mean():.3f} polish {p_.caps_at_target.mean():.3f}")
        lines.append(f"mean best_minz: fast {f_.best_minz.mean():.6f} polish {p_.best_minz.mean():.6f}")
        pj = p_.dropna(subset=["rescored_minz"])
        if len(pj):
            lines.append(f"max |rescored - reported| best_minz (polish) = {(pj.rescored_minz - pj.best_minz).abs().max():.3e}")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
    plots(run, plog)
    run.finish()


def plots(run, plog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(run.records)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df.groupby("method"):
        gg = g.groupby("threshold")
        ax[0].plot(gg.success.mean().index, gg.success.mean().values, "o-", label=m)
    ax[0].set_xlabel("target |Z11| (ohm)")
    ax[0].set_ylabel("success rate")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    if plog:
        pdf = pd.DataFrame(plog)
        ax[1].scatter(pdf.pso, pdf.relaxed, s=10, label="relaxed (continuous)")
        ax[1].scatter(pdf.pso, pdf.nearest, s=10, label="snapped nearest")
        ax[1].scatter(pdf.pso, pdf.local, s=10, label="snapped + local sweep")
        lim = [pdf.pso.min() * 0.9, pdf.pso.max() * 1.05]
        ax[1].plot(lim, lim, "k--", lw=.8)
        ax[1].set_xlabel("PSO stage best (ohm)")
        ax[1].set_ylabel("after polish, exact score (ohm)")
        ax[1].legend()
        ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "polish.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
