"""
Search quality Q4 -- is PSO the right search strategy? Same fitness, same budget, other optimisers.

The exact 'fast' evaluator is shared, so the comparison is about the search only. Every optimiser uses the PSO's staged scheme:
capacitor count n = 1..MAX_CAPS, a population of N_PARTICLES evaluated N_ITERATIONS times per stage, the same warm start from the
previous stage best (seed_particles), the same per-stage seeding and the same immediate stop (first individual, in order, that
meets the target). Particles are vectors in [0,1]^(2n) decoded exactly like the PSO.

    pso   gpuexp_common.run_pso (reference)
    de    differential evolution rand/1/bin, F = DE_F, CR = DE_CR; a trial replaces its parent if not worse
          (the parent's value is a valid pruning bound -> pruning stays exact)
    ga    tournament (3) selection, uniform crossover (0.9), Gaussian mutation (rate 1/dim, sigma GA_SIGMA), 2 elites
    sa    N_PARTICLES parallel simulated-annealing chains, Gaussian proposals with shrinking step, geometric cooling
Outputs: success, time per success, caps, evaluations, best |Z11|, Fisher p vs PSO, before/after plots.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import gpuexp_common as C
from gpuexp_common import S, cuda_sync

EXPERIMENT = "q4_optimizer_comparison"
DATASET = os.environ.get("DATASET", "ref21")
OPTIMISERS = R.env_list("OPTIMISERS", "pso,de,ga,sa")
DE_F, DE_CR = float(os.environ.get("DE_F", "0.5")), float(os.environ.get("DE_CR", "0.9"))
GA_SIGMA = float(os.environ.get("GA_SIGMA", "0.1"))


def run_generic(prob, ev, algo, thr, run_id):
    """Staged population search with the PSO's seeding / warm start / immediate stop. Returns a run_pso-style record."""
    NP, NI, MAXC = S["N_PARTICLES"], S["N_ITERATIONS"], S["MAX_CAPS"]
    ev.reset()
    cuda_sync()
    start = time.perf_counter()
    n_evals, prev_best, stage_best, best_cfg = 0, None, [], None
    t_target = caps_t = final = None
    for n in range(1, MAXC + 1):
        np.random.seed(S["BASE_SEED"] + run_id * 10000 + n)
        ev.begin_stage(n)
        pop = C.seed_particles(n, prev_best)
        dim = 2 * n
        val = np.full(NP, np.inf)
        gbest, gpart, stop = np.inf, None, False

        def score(X, bound):
            """Evaluate X in order with immediate stop. Returns (costs or None for unevaluated, stop)."""
            nonlocal n_evals, gbest, gpart
            M, Pp = prob.decode_swarm(X, n)
            c = ev.evaluate(M, Pp, bound)
            for i in range(len(X)):
                n_evals += 1
                if c[i] < gbest:
                    gbest, gpart = c[i], X[i].copy()
                if gbest <= thr:
                    c[i + 1:] = np.inf
                    return c, True
            return c, False

        T0 = None
        for it in range(NI):
            if it == 0:
                val, stop = score(pop, np.full(NP, np.inf))
                if stop:
                    break
                T0 = 0.1 * np.median(val[np.isfinite(val)]) if np.isfinite(val).any() else 1.0
                continue
            if algo == "de":
                idx = np.array([np.random.choice(np.delete(np.arange(NP), i), 3, replace=False) for i in range(NP)])
                mut = np.clip(pop[idx[:, 0]] + DE_F * (pop[idx[:, 1]] - pop[idx[:, 2]]), 0, 1)
                cross = np.random.rand(NP, dim) < DE_CR
                cross[np.arange(NP), np.random.randint(0, dim, NP)] = True
                trial = np.where(cross, mut, pop)
                tv, stop = score(trial, val.copy())                  # parent value = exact pruning bound
                better = tv <= val
                pop[better], val[better] = trial[better], tv[better]
            elif algo == "ga":
                order = np.argsort(val)
                elite = pop[order[:2]].copy()
                def tour():
                    c = np.random.randint(0, NP, 3)
                    return pop[c[np.argmin(val[c])]]
                kids = []
                while len(kids) < NP - 2:
                    a, b = tour(), tour()
                    if np.random.rand() < 0.9:
                        mask = np.random.rand(dim) < 0.5
                        a = np.where(mask, a, b)
                    mu = np.random.rand(dim) < 1.0 / dim
                    kids.append(np.clip(a + mu * np.random.normal(0, GA_SIGMA, dim), 0, 1))
                kids = np.array(kids)
                kv, stop = score(kids, np.full(len(kids), np.inf))
                pop = np.vstack([elite, kids])
                val = np.concatenate([val[order[:2]], kv])
            elif algo == "sa":
                T = T0 * (1e-3 ** (it / max(1, NI - 1)))
                step = 0.2 * (1 - it / NI) + 0.01
                prop = np.clip(pop + np.random.normal(0, step, pop.shape), 0, 1)
                pv, stop = score(prop, np.full(NP, np.inf))
                with np.errstate(over="ignore", invalid="ignore"):
                    acc = (pv <= val) | (np.random.rand(NP) < np.exp(-(pv - val) / max(T, 1e-30)))
                acc &= np.isfinite(pv)
                pop[acc], val[acc] = prop[acc], pv[acc]
            if stop:
                break
        if stop:
            cuda_sync()
            t_target, caps_t = time.perf_counter() - start, n
        if gpart is None:
            prev_best = None
            continue
        prev_best = gpart
        m, p = prob.decode_particle(gpart, n)
        m, p = prob.resolve_adjacent_ports(m, p)
        best_cfg = (m.tolist(), p.tolist())
        stage_best.append([n, float(gbest)])
        final = n
        if stop:
            break
    cuda_sync()
    total = time.perf_counter() - start
    rec = {"method": algo, "run": run_id, "seed": S["BASE_SEED"] + run_id * 10000, "threshold": thr,
           "success": t_target is not None, "time_s": t_target if t_target is not None else -1.0, "total_time_s": total,
           "caps_at_target": caps_t, "final_n_caps": final, "best_minz": min(b for _, b in stage_best) if stage_best else float("inf"),
           "n_evals": n_evals, "stage_best": stage_best, "best_models": best_cfg[0] if best_cfg else None,
           "best_ports": best_cfg[1] if best_cfg else None}
    rec.update(ev.stats())
    return rec


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET, "OPTIMISERS": OPTIMISERS, "DE_F": DE_F, "DE_CR": DE_CR, "GA_SIGMA": GA_SIGMA})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    ev = C.SwarmEvaluator(prob, True, True)
    C.warmup(prob, [ev])
    targets = R.targets_for(info)
    recs = []
    for thr in targets:
        for algo in OPTIMISERS:
            for r in range(1, S["NUM_RUNS"] + 1):
                run.lg.info("RUN_START | method=%s | threshold=%.6f | run=%d", algo, thr, r)
                try:
                    rec = C.run_pso(prob, ev, [thr], r, label="pso") if algo == "pso" else run_generic(prob, ev, algo, thr, r)
                except Exception as e:                                       # noqa: BLE001
                    run.fail(algo, thr, r, e)
                    continue
                run.add(rec)
                run.result_line(rec)
                recs.append(rec)
    import pandas as pd
    df = pd.DataFrame(recs)
    lines = []
    for thr, g in df.groupby("threshold"):
        ref = g[g.method == "pso"]
        for algo, a in g.groupby("method"):
            p = R.fisher_success(int(a.success.sum()), len(a), int(ref.success.sum()), len(ref)) if algo != "pso" and len(ref) else float("nan")
            lines.append(f"target {thr:g} {algo:4s}: success {int(a.success.sum())}/{len(a)} | time/success {R.time_per_success(a):.2f}s | "
                         f"evals {a.n_evals.mean():.0f} | caps {a.caps_at_target.dropna().mean():.2f} | best {a.best_minz.mean():.5f} | "
                         f"Fisher p vs pso {p:.3g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for algo, g in df.groupby("method"):
        s = g.groupby("threshold").success.mean()
        b = g.groupby("threshold").best_minz.mean()
        ax[0].plot(s.index, s.values, "o-", label=algo)
        ax[1].plot(b.index, b.values, "o-", label=algo)
    ax[0].set_ylabel("success rate")
    ax[1].set_ylabel("mean best |Z11| (ohm)")
    for a in ax:
        a.set_xlabel("target |Z11| (ohm)")
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "optimisers.png"), dpi=150)
    plt.close(fig)
    run.summarise(extra_cols=("cache_hits", "pruned"))
    R.write_lines(run, "optimiser_summary.txt", lines)
    R.plot_best_runs(prob, recs, os.path.join(run.run_dir, "plots"))
    run.finish()


if __name__ == "__main__":
    main()
