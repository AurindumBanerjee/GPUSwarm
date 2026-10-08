"""
Do the five inversion methods agree on the COST value?

Takes a fixed set of (model, pad) configurations (default 200, 1..max_caps capacitors,
seeded), evaluates each with numpy, solve, sm, iterative and pure_python on one
problem/band in the same per-frequency "loop" mode Grid 2 times, and reports the max
relative difference in the cost against numpy, per method.

    ~1e-12  -> agree; a placement divergence between those methods is float noise
               forking the PSO trajectory
    >~1e-9  -> not float noise: a bug, or (iterative) an approximation by construction

Needs no targets (the cost of a raw-objective problem is the peak impedance in ohm), so it
can run before targets.json is filled. Writes results/cost_agreement.json only.

    python check_cost_agreement.py                       # ddr21, B2, 200 configs
    python check_cost_agreement.py --n-configs 40 --problem mphy_mer2
"""
import os
import sys
import json
import time
import argparse

import study_config as C
import numpy as np
import study_core as S

METHODS = ["numpy", "solve", "sm", "iterative", "pure_python"]


def make_configs(prob, n, seed):
    rng = np.random.default_rng(seed)
    cfgs = []
    for i in range(n):
        n_caps = 1 + (i % prob.max_caps)                     # cycles 1..max_caps
        pads = rng.choice(prob.pads, size=n_caps, replace=False)
        models = rng.integers(0, S.decaps().shape[0], size=n_caps)
        cfgs.append([(int(m), int(p)) for m, p in zip(models, pads)])
    return cfgs


def iterative_diagnostics(prob, nf, cfgs):
    """How good is the 2nd-order Neumann series? Per frequency: ||E||_1 with
    E = A B - I (B = base inverse); the series needs ||E||_1 < 1 and its error is O(||E||^3);
    otherwise the method falls back to a full inverse."""
    d = S.decaps()
    yinv = prob.y_inv_base
    norms = []
    for cfg in cfgs:
        for f in range(nf):
            A = prob.y[f].copy()
            for cap, port in cfg:
                A[port, port] += d[cap, f]
            norms.append(np.linalg.norm(A @ yinv[f] - np.eye(prob.N), 1))
    norms = np.array(norms)
    return {"n_freq_evals": int(norms.size), "frac_fallback_to_full_inverse": float((norms >= 1).mean()),
            "median_norm": float(np.median(norms)), "p95_norm": float(np.percentile(norms, 95)),
            "max_norm": float(norms.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="ddr21")
    ap.add_argument("--band", default="B2", choices=list(C.BANDS))
    ap.add_argument("--n-configs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--out", default=os.path.join(C.OUT_ROOT, "cost_agreement.json"))
    args = ap.parse_args()

    S.set_require_targets(False)               # raw cost needs no targets
    prob = S.build_problem(args.problem)
    if prob.objective != "raw":
        sys.exit(f"{prob.name} uses the ratio objective, which needs targets; pick a raw problem")
    bs = S.band_spec(prob, args.band)
    cfgs = make_configs(prob, args.n_configs, args.seed)
    print(f"{prob.name}: N={prob.N}, n_obs={prob.n_obs}, band {args.band} ({bs[0]} frequencies), "
          f"{len(cfgs)} configurations of 1..{prob.max_caps} capacitors, seed {args.seed}", flush=True)

    ev = lambda cfg, m, mode="loop": S.evaluate(prob, bs, cfg, m, mode)[0]
    costs, secs = {}, {}
    # the batched numpy path is what Grid 1 used; include it as an extra reference
    t = time.perf_counter()
    costs["numpy_batched"] = np.array([ev(c, "numpy", "batched") for c in cfgs])
    secs["numpy_batched"] = time.perf_counter() - t
    for m in args.methods:
        t = time.perf_counter()
        out = []
        for i, c in enumerate(cfgs):
            out.append(ev(c, m))
            if m == "pure_python" and (i + 1) % 20 == 0:
                print(f"  pure_python {i + 1}/{len(cfgs)}  {time.perf_counter() - t:.0f}s", flush=True)
        costs[m] = np.array(out)
        secs[m] = time.perf_counter() - t
        print(f"  {m:<14} done in {secs[m]:.0f}s", flush=True)

    ref = costs["numpy"]
    report = {}
    print(f"\ncost = peak |Z| (ohm) on {args.band}; relative difference vs numpy (loop mode):")
    print(f"{'method':<15}{'max rel diff':>14}{'median':>12}{'worst at n_caps':>17}{'failed(1e200)':>15}  verdict")
    for m in ["numpy_batched"] + [x for x in args.methods if x != "numpy"]:
        c = costs[m]
        failed = int((c >= 1e199).sum())
        ok = c < 1e199
        rel = np.abs(c[ok] - ref[ok]) / np.abs(ref[ok])
        worst = int(np.argmax(np.where(ok, np.abs(c - ref) / np.abs(ref), -1)))
        mx = float(rel.max()) if rel.size else float("nan")
        verdict = ("agrees (float noise)" if mx <= 1e-12 else
                   "agrees to < 1e-9" if mx <= 1e-9 else "OFF by more than 1e-9")
        signed = (c[ok] - ref[ok]) / ref[ok]
        report[m] = {"max_rel_diff": mx, "median_rel_diff": float(np.median(rel)) if rel.size else None,
                     "worst_config_index": worst, "worst_config_n_caps": len(cfgs[worst]),
                     "n_failed": failed, "verdict": verdict,
                     "mean_signed_rel_diff": float(signed.mean()) if signed.size else None,
                     "frac_underestimated": float((signed < 0).mean()) if signed.size else None,
                     "seconds": secs[m]}
        print(f"{m:<15}{mx:>14.3e}{report[m]['median_rel_diff']:>12.2e}{len(cfgs[worst]):>17}{failed:>15}  {verdict}")
    if "iterative" in args.methods:
        diag = iterative_diagnostics(prob, bs[0], cfgs[:10])
        report["iterative"]["series_diagnostics_first10_configs"] = diag
        print(f"\niterative series (first 10 configs): {diag['frac_fallback_to_full_inverse']:.1%} of "
              f"(config, frequency) cases fall back to a full inverse; ||E||_1 median "
              f"{diag['median_norm']:.3g}, p95 {diag['p95_norm']:.3g}, max {diag['max_norm']:.3g}")
        print(f"iterative underestimates the cost in {report['iterative']['frac_underestimated']:.0%} of configs "
              f"(mean signed rel diff {report['iterative']['mean_signed_rel_diff']:+.2e})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"problem": prob.name, "band": args.band, "n_configs": len(cfgs), "seed": args.seed,
                   "cost_unit": "ohm (peak |Z| on the band)", "methods": report}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
