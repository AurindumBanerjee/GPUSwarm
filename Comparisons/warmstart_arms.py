#!/usr/bin/env python3
"""
Three-arm warm-start comparison on the ORIGINAL 21-port benchmark PDN.

Companion to warmstart_thresholds.py. That script asks "how fast does each warm
start reach a target"; this one asks "does the warm start change the answer at
all", by running every arm to its full capacitor budget with NO threshold.

  arm A  none      every stage reseeds a fully random swarm
                   -- the original ScratchBench / PythonBench behaviour
  arm B  naive     40% of particles carry prev_best EXACTLY, 60% random
                   -- the warm start before SB.py's FIX 11
  arm C  current   40% warm, particle 0 an exact elite, the rest jittered by
                   N(0, 0.02) on the carried genes only, 60% random
                   -- SB.py FIX 11, and what PSO_SP uses today

FIX 11's claim is specific: with every warm particle identical in the carried
dimensions, both PSO velocity terms -- (pbest - x) and (gbest - x) -- vanish
there, so those dimensions are frozen for the whole stage. This script measures
that directly rather than taking it on faith: `frozen` counts stages whose warm
block began with exactly zero spread across the carried genes, and `spread_end`
reports whether they ever moved.

numpy is the only inversion method. PSO constants are SB.py's and are held
identical across arms. Seeds are set per (run, n_caps) as in FIX 1, so arm A, B
and C at run r and capacitor count n start from the same RNG state -- any
difference is the seeding rule and nothing else.

Data: Data/Reference/{y2,decaps,freq2}.mat

    python warmstart_arms.py --estimate     # cost projection only, no PSO
    python warmstart_arms.py                # full sweep
    NUM_RUNS=5 THREADS=16 python warmstart_arms.py

Writes warmstart_arms_results.jsonl and warmstart_arms.png.
"""
import os, sys, json, time, statistics as st

THREADS = int(os.environ.get("THREADS", "1"))
if THREADS > 1:
    for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PDN_DATA", os.path.join(os.path.dirname(HERE), "Data", "Reference"))
OUT_JSONL = os.path.join(HERE, "warmstart_arms_results.jsonl")
OUT_PNG = os.path.join(HERE, "warmstart_arms.png")

# ---- constants, identical across arms (SB.py values) ----
N_PARTICLES   = 50
N_ITERATIONS  = 15
W_MAX, W_MIN  = 0.9, 0.4
C1, C2        = 1.5, 1.5
WARM_FRACTION = 0.40
WARM_JITTER   = 0.02
MAX_CAPS      = 20
BASE_SEED     = 12345
TARGET_PORT   = 0
NUM_RUNS      = int(os.environ.get("NUM_RUNS", "5"))

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
    """Peak |Z11| over the band, in ohm."""
    A = y.copy()
    for cap, port in config:
        A[:, port, port] += d[cap]
    try:
        return float(np.abs(np.linalg.inv(A)[:, TARGET_PORT, TARGET_PORT]).max())
    except np.linalg.LinAlgError:
        return 1e200


# ---- the three seeding rules: the only difference between arms ----
def _warm_block(n_caps, prev_best, n_warm):
    pn = n_caps - 1
    return np.hstack([
        np.tile(prev_best[:pn], (n_warm, 1)), np.random.rand(n_warm, 1),
        np.tile(prev_best[pn:], (n_warm, 1)), np.random.rand(n_warm, 1),
    ])


def seed_none(n_caps, prev_best):
    return np.random.rand(N_PARTICLES, 2 * n_caps)


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


ARMS = {"A none": seed_none, "B naive": seed_naive, "C current": seed_current}


def run_once(seed_fn, run_id, pool=None):
    """Full capacitor budget, no threshold. Returns the per-stage curve and the
    warm-block spread diagnostic."""
    prev_best, curve, t0, n_evals = None, [], time.perf_counter(), 0
    spread_start, spread_end, frozen, warm_stages = [], [], 0, 0

    for n_caps in range(1, MAX_CAPS + 1):
        np.random.seed(BASE_SEED + run_id * 10000 + n_caps)        # FIX 1: paired
        particles = seed_fn(n_caps, prev_best)
        velocities = np.zeros_like(particles)
        pbest, pbest_val = particles.copy(), np.full(N_PARTICLES, np.inf)
        gbest, gbest_particle = np.inf, None

        carried = None
        if prev_best is not None and n_caps > 1:
            nw = int(round(WARM_FRACTION * N_PARTICLES))
            carried = np.ones(2 * n_caps, bool)
            carried[n_caps - 1] = False; carried[-1] = False
            s0 = float(particles[:nw][:, carried].std(axis=0).mean())
            spread_start.append(s0); warm_stages += 1
            # std() of identical floats is ~1e-17, never exactly 0 -- use a
            # tolerance well below the jitter scale (0.02) instead.
            if particles[:nw][:, carried].std(axis=0).max() < 1e-12:
                frozen += 1

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
            w = W_MAX - (W_MAX - W_MIN) * (it / N_ITERATIONS)
            velocities = (w * velocities
                          + C1 * np.random.rand(*particles.shape) * (pbest - particles)
                          + C2 * np.random.rand(*particles.shape) * (gbest_particle - particles))
            particles = np.clip(particles + velocities, 0.0, 1.0)

        if carried is not None:
            nw = int(round(WARM_FRACTION * N_PARTICLES))
            spread_end.append(float(particles[:nw][:, carried].std(axis=0).mean()))

        curve.append(gbest)
        prev_best = gbest_particle                                 # FIX 13 if None

    return {"curve": curve, "best_ohm": min(curve), "n_evals": n_evals,
            "wall_s": time.perf_counter() - t0,
            "spread_start": st.mean(spread_start) if spread_start else None,
            "spread_end": st.mean(spread_end) if spread_end else None,
            "frozen_stages": frozen, "warm_stages": warm_stages}


def estimate():
    t = time.perf_counter()
    for _ in range(5):
        evaluate([(100, 3), (200, 5), (300, 7)])
    ev = (time.perf_counter() - t) / 5
    per_run = MAX_CAPS * N_PARTICLES * N_ITERATIONS
    runs = len(ARMS) * NUM_RUNS
    print(f"PDN {N_NODES} ports x {N_FREQS} freqs | bare peak "
          f"{np.abs(np.linalg.inv(y)[:, 0, 0]).max():.6g} ohm")
    print(f"per evaluation : {ev*1e3:.1f} ms  (threads={THREADS})")
    print(f"evals per run  : {per_run:,}  ({MAX_CAPS} stages x {N_PARTICLES} x {N_ITERATIONS})")
    print(f"runs           : {runs}  ({len(ARMS)} arms x {NUM_RUNS} runs)")
    print(f"TOTAL          : {runs*per_run*ev/3600:.2f} h")
    print("No thresholds here, so nothing stops early -- this is the full cost.")


def main():
    if "--estimate" in sys.argv:
        estimate(); return
    estimate(); print()

    pool = ThreadPoolExecutor(THREADS) if THREADS > 1 else None
    results = {}
    try:
        for name, fn in ARMS.items():
            results[name] = []
            for r in range(1, NUM_RUNS + 1):
                out = run_once(fn, r, pool)
                out.update(arm=name, run_id=r, method="numpy",
                           particles=N_PARTICLES, iterations=N_ITERATIONS,
                           warm_fraction=WARM_FRACTION, jitter=WARM_JITTER,
                           max_caps=MAX_CAPS, seed=BASE_SEED + r * 10000)
                results[name].append(out)
                with open(OUT_JSONL, "a") as fh:
                    fh.write(json.dumps(out) + "\n")
                print(f"{name:10s} run {r}  best {out['best_ohm']:.6f} ohm  "
                      f"{out['wall_s']:.1f}s  {out['n_evals']:,} evals", flush=True)
    finally:
        if pool: pool.shutdown()

    print("\n" + "=" * 70)
    print(f"{'arm':12s} {'median best':>12s} {'min':>10s} {'max':>10s} {'spread':>9s}")
    for name in ARMS:
        b = [x["best_ohm"] for x in results[name]]
        print(f"{name:12s} {st.median(b):12.6f} {min(b):10.6f} {max(b):10.6f} "
              f"{100*(max(b)-min(b))/min(b):8.2f}%")

    print("\nPaired per-run difference vs arm C (negative = that arm did better):")
    cb = [x["best_ohm"] for x in results["C current"]]
    for name in ("A none", "B naive"):
        ab = [x["best_ohm"] for x in results[name]]
        diffs = [100 * (a - c) / c for a, c in zip(ab, cb)]
        print(f"  {name:10s} median {st.median(diffs):+6.3f}%  "
              f"range {min(diffs):+.3f}% to {max(diffs):+.3f}%  "
              f"beats C in {sum(a < c for a, c in zip(ab, cb))}/{NUM_RUNS}")

    print("\nWarm-block carried-gene spread (the FIX 11 mechanism):")
    for name in ("B naive", "C current"):
        g = results[name]
        print(f"  {name:10s} start {st.mean([x['spread_start'] for x in g]):.3e}  "
              f"end {st.mean([x['spread_end'] for x in g]):.3e}  "
              f"frozen stages {sum(x['frozen_stages'] for x in g)}"
              f"/{sum(x['warm_stages'] for x in g)}")

    xs = np.arange(1, MAX_CAPS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in ARMS:
        c = np.array([x["curve"] for x in results[name]])
        ax[0].plot(xs, np.median(c, axis=0), marker="o", ms=3, lw=1.6, label=name)
        ax[0].fill_between(xs, c.min(axis=0), c.max(axis=0), alpha=.15)
        ax[1].scatter([name] * len(results[name]),
                      [x["best_ohm"] for x in results[name]], s=40, alpha=.75)
    ax[0].set_xlabel("Decaps"); ax[0].set_ylabel(r"peak $|Z_{11}|$ ($\Omega$)")
    ax[0].set_yscale("log"); ax[0].set_title(f"Median of {NUM_RUNS} runs, band = min-max")
    ax[0].grid(True, which="both", alpha=.3); ax[0].legend()
    ax[1].set_ylabel(r"best peak $|Z_{11}|$ ($\Omega$)")
    ax[1].set_title("Best reached, per run"); ax[1].grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=160)
    print(f"\nRecords -> {OUT_JSONL}\nFigure  -> {OUT_PNG}")


if __name__ == "__main__":
    main()
