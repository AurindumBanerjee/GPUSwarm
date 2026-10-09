"""
Experiment 6 -- GPU-resident PSO.

The whole PSO loop lives on the device: floor(frac*K) decoding, vectorised port-collision resolution (same rule as
GPUBench2.resolve_adjacent_ports: nearest free pad, the higher pad first on ties), batched exact Woodbury fitness, pbest/gbest
updates, the immediate-stop scan (first particle in order whose cost reaches the target) and the random numbers (torch.Generator on
the GPU, seeded BASE_SEED + run*10000 + n_caps per stage). One device->host transfer per iteration (the stop flag).

Modes (arms)
    gpu_rng      all randomness from the GPU generator. Cannot replay numpy's stream -> judged on OUTCOMES (success rate, caps,
                 evaluations, time) against the baseline arms.
    gpu_debug    the CPU-generated numpy draws (seed_particles + the two rand() per iteration) are copied to the device, everything
                 else runs on the device -> the trajectory must equal the baseline's EXACTLY (correctness check of decode /
                 collision / pbest / gbest / stop logic and velocity update on the device).
    woodbury_batch, fast   host-driven reference arms of GPUBench2 (same settings), for timing.
Also checked: decode_check.txt -- the device decode + collision resolution vs the CPU one on random and on heavily colliding swarms.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync

EXPERIMENT = "gpu_resident"


class GPUPSO:
    def __init__(self, prob, debug_numpy_rng=False):
        self.prob = prob
        self.debug = debug_numpy_rng
        self.name = "gpu_debug" if debug_numpy_rng else "gpu_rng"
        self.sequential = False
        self.valid = torch.from_numpy(prob.valid_ports).to(DEVICE)
        self.node_idx = torch.arange(prob.n, device=DEVICE)

    # ---- decode on the device
    def decode(self, x, n_caps):
        prob = self.prob
        models = torch.clamp(torch.floor(x[:, :n_caps] * prob.C).long(), 0, prob.C - 1)
        pidx = torch.clamp(torch.floor(x[:, n_caps:] * len(prob.valid_ports)).long(), 0, len(prob.valid_ports) - 1)
        ports = self.valid[pidx]
        return models, self.resolve(ports)

    def resolve(self, ports):
        """Vectorised resolve_adjacent_ports: sequential over the k slots, parallel over particles. A free pad is chosen by
        key = 2*|pad - p| + (pad < p): nearest first, the higher pad first on ties (original: cand p+off before p-off)."""
        prob = self.prob
        B, k = ports.shape
        ar = torch.arange(B, device=DEVICE)
        big = 4 * prob.n + 8
        node = self.node_idx.view(1, 1, -1)
        keyall = (node - ports.unsqueeze(2)).abs() * 2 + (node < ports.unsqueeze(2)).long()      # (B,k,n)
        pen = torch.zeros(B, prob.n, dtype=torch.long, device=DEVICE)                            # big where the pad is used
        pen[:, prob.target_port] = big
        out = ports.clone()
        for i in range(k):
            p = ports[:, i]
            kmin, newp = (keyall[:, i] + pen).min(dim=1)
            newp = torch.where(kmin >= big, p, newp)                                             # no free pad: keep (original)
            q = torch.where(pen[ar, p] > 0, newp, p)
            out[:, i] = q
            pen[ar, q] = big
        return out

    # ---- seeding on the device
    def seed_particles(self, n_caps, prev_best, gen):
        NP = S["N_PARTICLES"]
        n_warm = min(int(round(S["WARM_START_FRACTION"] * NP)), NP)
        jit = S["WARM_START_JITTER"]
        rnd = lambda *shape: torch.rand(*shape, generator=gen, device=DEVICE, dtype=torch.float64)
        if prev_best is None or n_caps == 1 or n_warm == 0:
            return rnd(NP, 2 * n_caps)
        pn = n_caps - 1
        warm = torch.cat([prev_best[:pn].repeat(n_warm, 1), rnd(n_warm, 1), prev_best[pn:].repeat(n_warm, 1), rnd(n_warm, 1)], dim=1)
        if jit > 0 and n_warm > 1:
            carried = torch.ones(2 * n_caps, dtype=torch.bool, device=DEVICE)
            carried[pn] = False
            carried[-1] = False
            noise = torch.randn(warm.shape, generator=gen, device=DEVICE, dtype=torch.float64) * jit
            noise[0, :] = 0.0
            noise[:, ~carried] = 0.0
            warm = torch.clamp(warm + noise, 0.0, 1.0)
        if NP - n_warm > 0:
            warm = torch.cat([warm, rnd(NP - n_warm, 2 * n_caps)], dim=0)
        return warm

    def run(self, thresholds, run_id):
        prob = self.prob
        thresholds = sorted(thresholds, reverse=True)
        thr = thresholds[-1]
        NP, NI, MAXC = S["N_PARTICLES"], S["N_ITERATIONS"], S["MAX_CAPS"]
        gen = torch.Generator(device=DEVICE)
        cuda_sync()
        start = time.perf_counter()
        time_to_target = caps_at_target = None
        n_evals = 0
        prev_best = None
        stage_best, best_cfg, final_caps = [], None, None
        for n_caps in range(1, MAXC + 1):
            seed = S["BASE_SEED"] + run_id * 10000 + n_caps
            if self.debug:
                np.random.seed(seed)
                x = torch.from_numpy(C.seed_particles(n_caps, None if prev_best is None else prev_best.cpu().numpy())).to(DEVICE)
            else:
                gen.manual_seed(seed)
                x = self.seed_particles(n_caps, prev_best, gen)
            v = torch.zeros_like(x)
            pbest = x.clone()
            pbest_val = torch.full((NP,), float("inf"), dtype=torch.float64, device=DEVICE)
            gbest = float("inf")
            gbest_p = None
            stop = False
            for it in range(NI):
                models, ports = self.decode(x, n_caps)
                cost = prob.wb_peak(models, ports)[0]
                better = cost < pbest_val
                pbest_val = torch.where(better, cost, pbest_val)
                pbest = torch.where(better.unsqueeze(1), x, pbest)
                hit = cost <= thr
                j_hit_t = hit.to(torch.int8).argmax()                         # first particle meeting the target
                j_min_t = cost.argmin()                                       # first minimum
                stats = torch.stack([hit.any().to(torch.float64), cost[j_min_t], j_hit_t.to(torch.float64),
                                     j_min_t.to(torch.float64), cost[j_hit_t]]).cpu().numpy()   # ONE device->host transfer per iteration
                any_hit, cmin, j_hit, j_min, cj = bool(stats[0]), float(stats[1]), int(stats[2]), int(stats[3]), float(stats[4])
                if any_hit:
                    n_evals += j_hit + 1
                    if cj < gbest:
                        gbest, gbest_p = cj, x[j_hit].clone()
                    stop = True
                    break
                n_evals += NP
                if cmin < gbest:
                    gbest, gbest_p = cmin, x[j_min].clone()
                w = S["W_MAX"] - (S["W_MAX"] - S["W_MIN"]) * (it / NI)
                if self.debug:
                    r1 = torch.from_numpy(np.random.rand(*x.shape)).to(DEVICE)
                    r2 = torch.from_numpy(np.random.rand(*x.shape)).to(DEVICE)
                else:
                    r1 = torch.rand(x.shape, generator=gen, device=DEVICE, dtype=torch.float64)
                    r2 = torch.rand(x.shape, generator=gen, device=DEVICE, dtype=torch.float64)
                v = w * v + S["C1"] * r1 * (pbest - x) + S["C2"] * r2 * (gbest_p - x)
                x = torch.clamp(x + v, 0.0, 1.0)
            if gbest_p is None:
                prev_best = None
                continue
            prev_best = gbest_p
            m, p = self.decode(gbest_p.unsqueeze(0), n_caps)
            best_cfg = (m[0].tolist(), p[0].tolist())
            stage_best.append([n_caps, gbest])
            final_caps = n_caps
            if stop:
                cuda_sync()
                time_to_target = time.perf_counter() - start
                caps_at_target = n_caps
                break
        cuda_sync()
        total = time.perf_counter() - start
        best = min(b for _, b in stage_best) if stage_best else float("inf")
        return {"method": self.name, "run": run_id, "seed": S["BASE_SEED"] + run_id * 10000, "threshold": thr,
                "success": time_to_target is not None, "time_s": time_to_target if time_to_target is not None else -1.0,
                "total_time_s": total, "caps_at_target": caps_at_target, "final_n_caps": final_caps, "best_minz": best,
                "n_evals": n_evals, "stage_best": stage_best, "best_models": best_cfg[0] if best_cfg else None,
                "best_ports": best_cfg[1] if best_cfg else None, "crossings": {}, "gpu_full_evals": n_evals,
                "gpu_screen_evals": 0, "cache_hits": 0, "pruned": 0, "subset_size": 0}

    # interface shims so warmup() can touch the kernels
    def reset(self):
        pass


def decode_check(prob, gp, n_swarms=200):
    """Device decode + collision resolution vs the CPU implementation: number of differing (model, port) entries."""
    rng = np.random.default_rng(7)
    bad = tot = 0
    for t in range(n_swarms):
        k = int(rng.integers(1, min(S["MAX_CAPS"], len(prob.valid_ports)) + 1))
        if t % 2:       # heavy collisions: all particles crowd a few pads
            x = rng.random((S["N_PARTICLES"], 2 * k))
            x[:, k:] = 0.5 + 0.03 * rng.random((S["N_PARTICLES"], k))
        else:
            x = rng.random((S["N_PARTICLES"], 2 * k))
        M, P = prob.decode_swarm(x, k)
        mt, pt = gp.decode(torch.from_numpy(x).to(DEVICE), k)
        bad += int((mt.cpu().numpy() != M).sum() + (pt.cpu().numpy() != P).sum())
        tot += M.size * 2
    return bad, tot


def main():
    run = C.Run(EXPERIMENT, __file__)
    C.init_device_log(run)
    prob = C.Problem.from_files(S["Y_PATH"], S["D_PATH"], S["FREQ_PATH"])
    host = {"woodbury_batch": C.SwarmEvaluator(prob, False, False), "fast": C.SwarmEvaluator(prob, True, True)}
    gp_rng, gp_dbg = GPUPSO(prob, False), GPUPSO(prob, True)
    bad, tot = decode_check(prob, gp_rng)
    run.lg.info("STAGE | decode_check | differing_entries=%d of %d", bad, tot)
    open(os.path.join(run.run_dir, "summary", "decode_check.txt"), "w").write(
        f"device decode + collision resolution vs CPU decode_swarm on 200 swarms (half heavily colliding)\n"
        f"differing_entries = {bad}\nentries_compared = {tot}\n")
    C.warmup(prob, list(host.values()))
    gp_rng.run([0.05], 1)     # touch the device kernels once (untimed)
    base = C.load_baseline()
    wb_ref = {}
    arms = list(host.items()) + [("gpu_rng", gp_rng), ("gpu_debug", gp_dbg)]
    for thr in S["TARGETS"]:
        for name, ev in arms:
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", name, thr, r)
                try:
                    rec = ev.run([thr], r) if isinstance(ev, GPUPSO) else C.run_pso(prob, ev, [thr], r, label=name)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(name, thr, r, e)
                    continue
                if name == "woodbury_batch":
                    wb_ref[(thr, r)] = rec
                ref = base.get((round(thr, 6), r)) or wb_ref.get((thr, r))
                rec["traj_match"], rec["traj_match_strict"] = C.match(rec, ref)
                if name == "gpu_debug":
                    rec["traj_match_vs_woodbury_batch"], rec["traj_match_strict_vs_woodbury_batch"] = C.match(rec, wb_ref.get((thr, r)))
                run.add(rec)
                run.result_line(rec, match=rec["traj_match"], strict=rec["traj_match_strict"])
    run.summarise(extra_cols=("traj_match", "traj_match_strict", "traj_match_vs_woodbury_batch", "traj_match_strict_vs_woodbury_batch"))
    import pandas as pd
    df = pd.DataFrame(run.records)
    lines = [f"decode_check_differing_entries = {bad} of {tot}"]
    t = df.groupby("method").total_time_s.sum()
    for ref in ("woodbury_batch", "fast"):
        for m in ("gpu_rng", "gpu_debug"):
            if ref in t and m in t:
                lines.append(f"total_time {ref} = {t[ref]:.2f}s {m} = {t[m]:.2f}s speedup_{m}_over_{ref} = {t[ref] / t[m]:.3f}")
    d = df[df.method == "gpu_debug"]
    if len(d):
        lines.append(f"gpu_debug trajectory match vs numpy baseline (best_minz, caps_at_target) = {d.traj_match.mean():.4f}"
                     f" (strict incl. n_evals: {d.traj_match_strict.mean():.4f}); vs woodbury_batch = {d.traj_match_vs_woodbury_batch.mean():.4f}"
                     f" (strict {d.traj_match_strict_vs_woodbury_batch.mean():.4f})")
    with open(os.path.join(run.run_dir, "summary", "statistics.txt"), "a") as fh:
        fh.write("\n".join(lines) + "\n")
    plots(run)
    run.finish(notes={"decode_check_differing": bad})


def plots(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(run.records)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in df.groupby("method"):
        gg = g.groupby("threshold").total_time_s
        ax[0].errorbar(gg.mean().index, gg.mean().values, yerr=[C.ci95(x) for _, x in gg], marker="o", capsize=3, label=m)
        ax[1].plot(g.groupby("threshold").success.mean().index, g.groupby("threshold").success.mean().values, "o-", label=m)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("target |Z11| (ohm)")
    ax[0].set_ylabel("wall time per run (s)")
    ax[1].set_xlabel("target |Z11| (ohm)")
    ax[1].set_ylabel("success rate")
    for a in ax:
        a.legend()
        a.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "time_and_success.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
