"""
Speed S4 -- run many independent PSO runs (seeds) in lock-step and evaluate all their swarms in ONE GPU call.

One swarm (50 particles) leaves the GPU mostly idle. Here all NUM_RUNS runs of a threshold advance together: every lock-step
iteration gathers the swarms of all active runs (grouped by their current capacitor count k), evaluates them with a single batched
Woodbury call per group, and then each run does its own pbest/gbest scan, immediate stop and velocity update.
Each run keeps its own np.random.RandomState seeded exactly like run_pso (BASE_SEED + run*10000 + n_caps per stage), so every run
follows EXACTLY the trajectory of the sequential 'woodbury_batch' arm -- checked run by run.

What changes is throughput: total benchmark wall time for NUM_RUNS runs. Time-to-target of a single run is not meaningful in
lock-step mode (all runs share the clock), so only the total is compared.
Arms: woodbury_batch (sequential, one run after the other)  vs  lockstep (all runs together).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE, cuda_sync

EXPERIMENT = "s4_batched_seeds"
DATASET = os.environ.get("DATASET", "ref21")


def seed_particles_rs(rs, n_caps, prev_best):
    """C.seed_particles with an explicit RandomState (same draw order as the global-stream original)."""
    NP = S["N_PARTICLES"]
    n_warm = min(int(round(S["WARM_START_FRACTION"] * NP)), NP)
    jit = S["WARM_START_JITTER"]
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return rs.rand(NP, 2 * n_caps)
    pn = n_caps - 1
    warm = np.hstack([np.tile(prev_best[:pn], (n_warm, 1)), rs.rand(n_warm, 1),
                      np.tile(prev_best[pn:], (n_warm, 1)), rs.rand(n_warm, 1)])
    if jit > 0 and n_warm > 1:
        carried = np.ones(2 * n_caps, bool)
        carried[pn] = False
        carried[-1] = False
        noise = rs.normal(0.0, jit, warm.shape)
        noise[0, :] = 0.0
        noise[:, ~carried] = 0.0
        warm = np.clip(warm + noise, 0.0, 1.0)
    if NP - n_warm > 0:
        warm = np.vstack([warm, rs.rand(NP - n_warm, 2 * n_caps)])
    return warm


class RunState:
    def __init__(self, run_id, thr):
        self.run_id, self.thr = run_id, thr
        self.n_caps = 0
        self.prev_best = None
        self.n_evals = 0
        self.stage_best = []
        self.done = False
        self.success = False
        self.caps_at_target = None
        self.best_cfg = None
        self.new_stage()

    def new_stage(self):
        self.n_caps += 1
        if self.n_caps > S["MAX_CAPS"]:
            self.done = True
            return
        self.rs = np.random.RandomState(S["BASE_SEED"] + self.run_id * 10000 + self.n_caps)
        self.x = seed_particles_rs(self.rs, self.n_caps, self.prev_best)
        self.v = np.zeros_like(self.x)
        self.pbest = self.x.copy()
        self.pbest_val = np.full(S["N_PARTICLES"], np.inf)
        self.gbest, self.gpart, self.it = np.inf, None, 0

    def step(self, costs, prob):
        """One iteration given the swarm costs: scan with immediate stop, then velocity update or stage end."""
        stop = False
        for i in range(S["N_PARTICLES"]):
            c = costs[i]
            self.n_evals += 1
            if c < self.pbest_val[i]:
                self.pbest_val[i] = c
                self.pbest[i] = self.x[i]
            if c < self.gbest:
                self.gbest = c
                self.gpart = self.x[i].copy()
            if self.gbest <= self.thr:
                stop = True
                break
        if not stop:
            it = self.it
            w = S["W_MAX"] - (S["W_MAX"] - S["W_MIN"]) * (it / S["N_ITERATIONS"])
            self.v = w * self.v + S["C1"] * self.rs.rand(*self.x.shape) * (self.pbest - self.x) \
                + S["C2"] * self.rs.rand(*self.x.shape) * (self.gpart - self.x)
            self.x = np.clip(self.x + self.v, 0, 1)
            self.it += 1
            if self.it < S["N_ITERATIONS"]:
                return
        # stage end
        if self.gpart is None:
            self.prev_best = None
            self.new_stage()
            return
        self.prev_best = self.gpart
        m, p = prob.decode_particle(self.gpart, self.n_caps)
        m, p = prob.resolve_adjacent_ports(m, p)
        self.best_cfg = (m.tolist(), p.tolist())
        self.stage_best.append([self.n_caps, float(self.gbest)])
        if stop or self.gbest <= self.thr:
            self.done, self.success, self.caps_at_target = True, True, self.n_caps
            return
        self.new_stage()


def lockstep(prob, thr, run_ids):
    states = [RunState(r, thr) for r in run_ids]
    calls = 0
    cuda_sync()
    t0 = time.perf_counter()
    while True:
        act = [s for s in states if not s.done]
        if not act:
            break
        for k in sorted({s.n_caps for s in act}):
            grp = [s for s in act if s.n_caps == k]
            Ms, Ps = zip(*(prob.decode_swarm(s.x, k) for s in grp))
            mt = torch.from_numpy(np.concatenate(Ms)).to(DEVICE)
            pt = torch.from_numpy(np.concatenate(Ps)).to(DEVICE)
            cost = prob.wb_peak(mt, pt)[0].cpu().numpy()
            calls += 1
            NP = S["N_PARTICLES"]
            for j, s in enumerate(grp):
                s.step(cost[j * NP:(j + 1) * NP], prob)
    cuda_sync()
    total = time.perf_counter() - t0
    recs = []
    for s in states:
        best = min(b for _, b in s.stage_best) if s.stage_best else float("inf")
        recs.append({"method": "lockstep", "run": s.run_id, "threshold": thr, "success": s.success, "time_s": -1.0,
                     "total_time_s": total / len(states), "caps_at_target": s.caps_at_target,
                     "final_n_caps": s.stage_best[-1][0] if s.stage_best else None, "best_minz": best, "n_evals": s.n_evals,
                     "best_models": s.best_cfg[0] if s.best_cfg else None, "best_ports": s.best_cfg[1] if s.best_cfg else None,
                     "lockstep_total_s": total, "lockstep_gpu_calls": calls})
    return recs, total


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASET": DATASET})
    C.init_device_log(run)
    prob, info = R.load_problem(DATASET)
    run.lg.info("STAGE | %s", info)
    ev = C.SwarmEvaluator(prob, False, False)
    C.warmup(prob, [ev])
    lines = []
    runs = list(range(1, S["NUM_RUNS"] + 1))
    for thr in R.targets_for(info):
        cuda_sync()
        t0 = time.perf_counter()
        seq = R.loop_methods(run, prob, {"woodbury_batch": ev}, [thr], S["NUM_RUNS"], reference="woodbury_batch")
        cuda_sync()
        t_seq = time.perf_counter() - t0
        ls, t_ls = lockstep(prob, thr, runs)
        ref = {r["run"]: r for r in seq}
        for r in ls:
            b = ref.get(r["run"])
            r["traj_match"], r["traj_match_strict"] = C.match(r, b)
            run.add(r)
            run.result_line(r, match=r["traj_match"])
        strict = np.mean([bool(r["traj_match_strict"]) for r in ls])
        lines.append(f"target {thr:g}: sequential {t_seq:.2f}s for {len(runs)} runs | lock-step {t_ls:.2f}s "
                     f"({ls[0]['lockstep_gpu_calls']} GPU calls) -> throughput gain {t_seq / t_ls:.2f}x | "
                     f"identical trajectories (best, caps, evals, final stage) {strict:.3f}")
        R.plot_best_runs(prob, ls, os.path.join(run.run_dir, "plots"))
    run.summarise(extra_cols=("traj_match", "traj_match_strict"))
    R.write_lines(run, "lockstep_summary.txt", lines)
    run.finish()


if __name__ == "__main__":
    main()
