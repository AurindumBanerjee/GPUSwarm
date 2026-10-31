#!/usr/bin/env python3
"""AllCPU dry check -- writes nothing outside a temp dir, runs no study.  For each PDN: shapes, port rules, library/grid agreement, bands (points per band, bare peaks),
method agreement on identical placements (core.selftest), measured seconds per evaluation of every method (single thread), target status, and a tiny 2-stage smoke PSO.
   python validate.py [--pdns pdn1,pdn2,pdn3,pdn4] [--no-smoke]
Exit 3 when PDN 3 / PDN 4 floors are still null (floor_pass.py not run yet) -- everything else is still checked."""
import os, sys, time, tempfile, argparse
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
ap = argparse.ArgumentParser(); ap.add_argument("--pdns", default="pdn1,pdn2,pdn3,pdn4"); ap.add_argument("--no-smoke", action="store_true"); a = ap.parse_args()
import numpy as np, allcpu_pdns as R
missing = False
for key in a.pdns.split(","):
    cell = R.Cell(key, 24, 2); core = cell.core; s = cell.spec; p = cell.prob; f = core.library()["freq"]
    print(f"\n=== {s['label']} ({p.name}) ===")
    print(f"N={p.N} n_obs={p.n_obs} obs={list(p.obs)} pads={len(p.pads)} (first pad idx {p.pads[0]}) max_caps={p.max_caps}  F={p.F}  grid {f[0]:.2f} Hz - {f[-1]:.6g} Hz  library {core.library()['d'].shape}")
    assert p.max_caps == p.N - p.n_obs and p.n_obs == s["n_obs"] and p.F == len(f) == core.library()["d"].shape[1]
    core.OBJ_WEIGHTS = None; bare = cell.bare_peaks()
    for b in core.BANDS: print(f"  band {b:7} {R.BAND_LABEL.get(b, ''):34} {int(core.band_mask(b).sum()):5d} pts   bare peak per obs [ohm]: " + ", ".join(f"{x:.4g}" for x in bare[b]))
    core.RANK_BAND = s["opt_bands"][0]; core.selftest(p, ["numpy", "solve", "sm", "iterative"])
    rng = np.random.default_rng(5); d = core.library()["d"]; tim = {}
    cfgs = [[(int(rng.integers(d.shape[0])), int(rng.choice(p.pads))) for _ in range(8)] for _ in range(3)]
    for m in ["numpy", "solve", "sm", "iterative"]:
        core.peaks_all_bands(p, cfgs[0], m); t0 = time.perf_counter()
        for c in cfgs: core.peaks_all_bands(p, c, m)
        tim[m] = (time.perf_counter() - t0) / len(cfgs)
    print("  s/eval (8 caps, 1 thread, this machine): " + "  ".join(f"{m} {v:.4f}" for m, v in tim.items()))
    for b in s["opt_bands"]:
        try:
            lv, info = cell.select_band(b); print(f"  targets {b:7} floor {info['floor']:.6g} bare {info['bare']:.6g} [{info['unit']}]  levels " + "  ".join(f"{k}={lv[k]:.6g}" for k in cell.order(lv)))
        except RuntimeError as e: print(f"  targets {b:7} NOT READY: {e}"); missing = True
    if not a.no_smoke:
        tmp = tempfile.mkdtemp(prefix="allcpu_smoke_"); core.OBJ_WEIGHTS = None; core.RANK_BAND = s["opt_bands"][0]
        r = core.run_pso(p, "numpy", 1, os.path.join(tmp, "r"), os.path.join(tmp, "res.jsonl"), pool=None, out_root=tmp, threshold=None, max_caps=2)
        print(f"  smoke ok: {r['n_fitness_evals']} evals, warm {r['n_warm_effective']}/24, best {r['best_cost']:.5g} @ {r['best_n_caps']} caps ({tmp})")
print("\nvalidate: done" + (" -- PDN 3/4 floors missing, run floor_pass.py" if missing else " -- all targets ready")); sys.exit(3 if missing else 0)
