#!/usr/bin/env python3
"""
Warm-start comparison on the ORIGINAL 21-port benchmark PDN.

Question: does SB.py's FIX 11 refinement of the warm start -- an exact elite
particle plus Gaussian jitter on the carried genes -- change the outcome versus
the naive warm start it replaced, where 40% of the swarm carried the previous
stage's best verbatim?

  arm B (naive)    40% of particles carry prev_best EXACTLY, 60% random.
                   Every carried gene is identical across the warm block, so
                   both PSO velocity terms vanish there and those dimensions
                   cannot move within a stage.
  arm C (current)  40% warm, particle 0 an exact elite copy, the rest jittered
                   by N(0, 0.02) on the carried genes only, 60% random.
                   This is what SB.py and PSO_SP use today.

Everything else is held constant: numpy only, SB.py's PSO constants, the same
discretisation and collision resolver, and PAIRED SEEDS -- both arms at run r and
capacitor count n start from the same RNG state, so any difference is the
seeding rule and nothing else.

Measured per (arm, threshold, run): whether the target was met, time to target,
capacitors needed, and fitness evaluations spent.

Data: Data/Reference/{y2,decaps,freq2}.mat -- the dataset ScratchBench and SB.py
were written against.

    python warmstart_thresholds.py                 # full sweep
    python warmstart_thresholds.py --estimate      # cost estimate only, no PSO
    NUM_RUNS=5 THREADS=16 python warmstart_thresholds.py
"""
import os, sys, json, time

THREADS = int(os.environ.get("THREADS", "1"))
if THREADS > 1:
    for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np
import scipy.io as sio
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PDN_DATA", os.path.join(os.path.dirname(HERE), "Data", "Reference"))
OUT = os.path.join(HERE, "warmstart_results.jsonl")

# ---- constants, all held fixed across arms (SB.py values) ----
N_PARTICLES   = 50
N_ITERATIONS  = 15
W_MAX, W_MIN  = 0.9, 0.4
C1, C2        = 1.5, 1.5
WARM_FRACTION = 0.40
WARM_JITTER   = 0.02
MAX_CAPS      = 20
BASE_SEED     = 12345
TARGET_PORT   = 0
NUM_RUNS      = int(os.environ.get("NUM_RUNS", "10"))
THRESHOLDS    = [0.05, 0.045, 0.04, 0.03]        # SB.py's TARGETS

y = np.transpose(sio.loadmat(os.path.join(DATA, "y2.mat"))["y"], (2, 0, 1))
d = sio.loadmat(os.path.join(DATA, "decaps.mat"))["decaps"]
freq = sio.loadmat(os.path.join(DATA, "freq2.mat"))["freq"].ravel()
N_FREQS, N_NODES, _ = y.shape
N_CAP_MODELS = d.shape[0]
VALID_PORTS = np.array([p for p in range(N_NODES) if p != TARGET_PORT])


# ---- machinery, ported unchanged from SB.py ----
def frac_to_index(frac, K):                                   # FIX 2
    return np.clip(np.floor(np.asarray(frac) * K).astype(int), 0, K - 1)


def decode_particle(vec, n_caps):                             # FIX 3
    return (frac_to_index(vec[:n_caps], N_CAP_MODELS),
            VALID_PORTS[frac_to_index(vec[n_caps:], len(VALID_PORTS))])


def resolve_adjacent_ports(models, ports):                    # FIX 12
    used, ports = {TARGET_PORT}, [int(p) for p in ports]
    for i in range(len(ports)):
        if ports[i] not in used:
            used.add(ports[i]); continue
        placed = False
        for off in range(1, N_NODES):
            for cand in (ports[i] + off, ports[i] - off):
                if 0 <= cand < N_NODES and cand != TARGET_PORT and cand not in used:
                    ports[i] = cand; used.add(cand); placed = True; break
            if placed: break
        if not placed:
            used.add(ports[i])
    return models, ports


def evaluate(config):
    """Peak |Z11| over the band, ohm. numpy is the only method here."""
    A = y.copy()
    for cap, port in config:
        A[:, port, port] += d[cap]
    try:
        return float(np.abs(np.linalg.inv(A)[:, TARGET_PORT, TARGET_PORT]).max())
    except np.linalg.LinAlgError:
        return 1e200


# ---- the only thing that differs between arms ----
def _warm_block(n_caps, prev_best, n_warm):
    pn = n_caps - 1
    return np.hstack([
        np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
        np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1),
    ])


def seed_naive(n_caps, prev_best):
    n_warm = int(round(WARM_FRACTION * N_PARTICLES))
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(N_PARTICLES, 2 * n_caps)
    return np.vstack([_warm_block(n_caps, prev_best, n_warm),
                      np.random.rand(N_PARTICLES - n_warm, 2 * n_caps)])


def seed_current(n_caps, prev_best):
    n_warm = int(round(WARM_FRACTION * N_PARTICLES))
    if prev_best is None or n_caps == 1 or n_warm == 0:
        return np.random.rand(N_PARTICLES, 2 * n_caps)
    warm = _warm_block(n_caps, prev_best, n_warm)
    if WARM_JITTER > 0 and n_warm > 1:
        pn = n_caps - 1
        carried = np.ones(2 * n_caps, bool); carried[pn] = False; carried[-1] = False
        noise = np.random.normal(0.0, WARM_JITTER, warm.shape)
        noise[0, :] = 0.0                      # elite particle untouched
        noise[:, ~carried] = 0.0               # never perturb the new genes
        warm = np.clip(warm + noise, 0.0, 1.0)
    return np.vstack([warm, np.random.rand(N_PARTICLES - n_warm, 2 * n_caps)])


ARMS = {"naive": seed_naive, "current": seed_current}


def run_once(seed_fn, run_id, threshold, pool=None):
    """One PSO run. Stops the instant the running best meets the threshold.
    Also records the spread of the carried genes inside the warm block, which
    is the quantity FIX 11 claims a naive warm start drives to zero."""
    prev_best, t0, n_evals = None, time.perf_counter(), 0
    best = np.inf
    time_to_target = caps_at_target = None
    frozen_stages = total_stages = 0

    for n_caps in range(1, MAX_CAPS + 1):
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)        # FIX 1: paired
        particles = seed_fn(n_caps, prev_best)
        velocities = np.zeros_like(particles)
        pbest, pbest_val = particles.copy(), np.full(N_PARTICLES, np.inf)
        gbest, gbest_particle = np.inf, None
        stop = False

        if prev_best is not None and n_caps > 1:
            nw = int(round(WARM_FRACTION * N_PARTICLES))
            carried = np.ones(2 * n_caps, bool)
            carried[n_caps - 1] = False; carried[-1] = False
            total_stages += 1
            # std() of identical floats is ~1e-17, never exactly 0 -- use a
            # tolerance well below the jitter scale (0.02) instead.
            if particles[:nw][:, carried].std(axis=0).max() < 1e-12:
                frozen_stages += 1

        for it in range(N_ITERATIONS):
            cfgs = []
            for i in range(N_PARTICLES):
                m, p = decode_particle(particles[i], n_caps)
                m, p = resolve_adjacent_ports(m, p)
                cfgs.append(list(zip(m, p)))
            costs = list(pool.map(evaluate, cfgs)) if pool else None
            for i in range(N_PARTICLES):
                cost = costs[i] if pool else evaluate(cfgs[i])
                n_evals += 1
                if cost < pbest_val[i]:
                    pbest_val[i], pbest[i] = cost, particles[i]
                if cost < gbest:
                    gbest, gbest_particle = cost, particles[i].copy()
                best = min(best, cost)
                if gbest <= threshold:                            # immediate stop
                    stop = True
                    break
            if stop:
                time_to_target = time.perf_counter() - t0
                caps_at_target = n_caps
                break
            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)
            velocities = (w * velocities
                          + C1 * np.random.rand(*particles.shape) * (pbest - particles)
                          + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles))
            particles = np.clip(particles + velocities, 0.0, 1.0)

        if stop:
            break
        prev_best = gbest_particle                                 # FIX 13 if None

    return {"best_ohm": best, "success": time_to_target is not None,
            "time_to_target_s": time_to_target, "caps_at_target": caps_at_target,
            "n_evals": n_evals, "wall_s": time.perf_counter() - t0,
            "frozen_stages": frozen_stages, "warm_stages": total_stages}


def estimate():
    t = time.perf_counter()
    for _ in range(5):
        evaluate([(100, 3), (200, 5), (300, 7)])
    ev = (time.perf_counter() - t) / 5
    worst = MAX_CAPS * N_PARTICLES * N_ITERATIONS
    cells = len(ARMS) * len(THRESHOLDS) * NUM_RUNS
    print(f"PDN {N_NODES} ports x {N_FREQS} freqs | bare peak "
          f"{np.abs(np.linalg.inv(y)[:, 0, 0]).max():.6g} ohm")
    print(f"per evaluation      : {ev*1e3:.1f} ms  (threads={THREADS})")
    print(f"evals per run (max) : {worst:,}  = {MAX_CAPS} stages x {N_PARTICLES} x {N_ITERATIONS}")
    print(f"runs                : {cells}  = {len(ARMS)} arms x {len(THRESHOLDS)} thresholds x {NUM_RUNS} runs")
    print(f"WORST CASE (nothing stops early) : {cells*worst*ev/3600:.2f} h")
    print(f"typical (early stop ~1/3 budget) : {cells*worst*ev/3600/3:.2f} h")
    print("Runs stop the moment the threshold is met, so the loose thresholds "
          "cost a fraction of the worst case.")


def main():
    if "--estimate" in sys.argv:
        estimate(); return
    estimate(); print()
    pool = ThreadPoolExecutor(THREADS) if THREADS > 1 else None
    rows = []
    try:
        for thr in THRESHOLDS:
            for arm, fn in ARMS.items():
                for r in range(1, NUM_RUNS + 1):
                    out = run_once(fn, r, thr, pool)
                    out.update(arm=arm, threshold=thr, run_id=r,
                               particles=N_PARTICLES, iterations=N_ITERATIONS,
                               warm_fraction=WARM_FRACTION, jitter=WARM_JITTER,
                               method="numpy", seed=BASE_SEED + r * 10000)
                    rows.append(out)
                    with open(OUT, "a") as fh:
                        fh.write(json.dumps(out) + "\n")
                    print(f"thr={thr:<6} {arm:8s} run {r:2d}  "
                          f"{'HIT ' if out['success'] else 'miss'} "
                          f"best={out['best_ohm']:.6f} "
                          f"t={out['time_to_target_s'] or out['wall_s']:7.1f}s "
                          f"caps={out['caps_at_target']} evals={out['n_evals']:,}",
                          flush=True)
    finally:
        if pool: pool.shutdown()

    import statistics as st
    print("\n" + "=" * 78)
    print(f"{'threshold':>10} {'arm':>9} {'success':>8} {'med t2t':>9} "
          f"{'med caps':>9} {'med evals':>10} {'frozen stages':>14}")
    for thr in THRESHOLDS:
        for arm in ARMS:
            g = [x for x in rows if x["threshold"] == thr and x["arm"] == arm]
            ok = [x for x in g if x["success"]]
            fz = sum(x["frozen_stages"] for x in g), sum(x["warm_stages"] for x in g)
            print(f"{thr:>10} {arm:>9} {len(ok)}/{len(g):<6} "
                  f"{(st.median([x['time_to_target_s'] for x in ok]) if ok else float('nan')):9.1f} "
                  f"{(st.median([x['caps_at_target'] for x in ok]) if ok else float('nan')):9.1f} "
                  f"{(st.median([x['n_evals'] for x in ok]) if ok else float('nan')):10.0f} "
                  f"{fz[0]:>6}/{fz[1]:<7}")
    print("\nPaired per-run comparison (same seed, same threshold):")
    for thr in THRESHOLDS:
        a = {x["run_id"]: x for x in rows if x["threshold"] == thr and x["arm"] == "naive"}
        b = {x["run_id"]: x for x in rows if x["threshold"] == thr and x["arm"] == "current"}
        same = sum(1 for k in a if abs(a[k]["best_ohm"] - b[k]["best_ohm"]) < 1e-12)
        cw = sum(1 for k in a if b[k]["best_ohm"] < a[k]["best_ohm"])
        nw = sum(1 for k in a if a[k]["best_ohm"] < b[k]["best_ohm"])
        print(f"  thr={thr:<6} identical {same}/{len(a)} | current better {cw} | naive better {nw}")
    print(f"\nRecords -> {OUT}")


if __name__ == "__main__":
    main()
