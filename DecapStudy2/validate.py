"""
--validate: load every dataset, check shapes and the port rules, print the
resolved targets and the planned cell list, then exit. Runs NO PSO and writes
nothing.

Exit codes: 0 all checks passed and targets resolved; 1 a check failed;
3 checks passed but targets.json still has null/invalid targets (the study would
refuse to start).
"""
import os
import numpy as np

import study_config as C
import study_core as S

EXPECTED_N = {          # port counts from the Dataset READMEs; a sanity check on the loaded matrices
    "y_sp_mer1.mat": 16, "y_sp_mer2.mat": 16, "y_sp_pll1v0.mat": 15,
    "y_sp_pll1v8.mat": 16, "y_sp_mphyvdd.mat": 10, "y_sp_mphy_full.mat": 73,
    "y_sp_ddr21.mat": 21, "y_sp_ddr_full.mat": 79,
}
EXPECTED_MODELS = 3347

_fails = []


def check(cond, msg):
    print(("  [ ok ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        _fails.append(msg)
    return cond


def _files():
    return ([("MPHY", r[1]) for r in C.MPHY_RAILS] + [("MPHY", "y_sp_mphy_full.mat")]
            + [("VDDQ_DDR3", "y_sp_ddr21.mat"), ("VDDQ_DDR3", "y_sp_ddr_full.mat")])


def check_files_and_shapes():
    print(f"\n[1] data under {C.DATA_ROOT}")
    need = [("MPHY", "freq_sp.mat"), ("Decaps", "decaps_sp.mat")] + _files()
    missing = [f"{d}/{f}" for d, f in need if not os.path.exists(os.path.join(C.DATA_ROOT, d, f))]
    if not check(not missing, f"all {len(need)} required files present" +
                 (f" (missing: {missing})" if missing else "")):
        return False
    f = S.freqs()
    F = len(f)
    check(bool(np.all(np.diff(f) > 0)), f"frequency grid ascending, F={F}, "
                                        f"{f[0]:.4g} Hz .. {f[-1]:.6g} Hz")
    d = S.decaps()
    check(d.ndim == 2 and d.shape == (EXPECTED_MODELS, F),
          f"decaps_sp shape {d.shape} == ({EXPECTED_MODELS}, {F})")
    check(bool(np.isfinite(d).all()), "decaps_sp finite")
    for sub, fn in _files():
        y = S._load_y(sub, fn)                         # (F, N, N) after transpose
        N = y.shape[1]
        n_obs = S._n_obs(fn)
        ok = (y.ndim == 3 and y.shape == (F, N, N) and N == EXPECTED_N[fn]
              and bool(np.isfinite(y).all()) and np.iscomplexobj(y))
        check(ok, f"{sub}/{fn}: y {tuple(y.shape)} complex, N={N} (expected {EXPECTED_N[fn]}), "
                  f"n_obs={n_obs} -> {N - n_obs} pads")
    return True


def check_port_rules():
    print("\n[2] port rules (observation ports first, never a capacitor site; "
          "max caps = N - n_obs)")
    S.set_require_targets(False)                        # targets may still be null here
    keys = [f"mphy_{r[0]}" for r in C.MPHY_RAILS] + ["mphy_full", "mphy_improvable",
                                                       "ddr21", "ddr_full"]
    rng = np.random.default_rng(0)
    probs = {}
    for key in keys:
        p = S.build_problem(key)
        probs[key] = p
        ok = (np.array_equal(p.obs, np.arange(p.n_obs))
              and np.array_equal(p.pads, np.arange(p.n_obs, p.N))
              and not set(p.obs.tolist()) & set(p.pads.tolist())
              and len(p.obs) + len(p.pads) == p.N and p.max_caps == p.N - p.n_obs)
        check(ok, f"{p.name:<16} N={p.N:<3} n_obs={p.n_obs}  obs={p.obs.tolist()}  "
                  f"pads={p.pads[0]}..{p.pads[-1]} ({len(p.pads)})  max_caps={p.max_caps}  "
                  f"objective={p.objective}")
        # decode -> resolve must never produce an observation port or a duplicate pad
        bad = 0
        for n_caps in sorted({1, max(1, p.max_caps // 2), p.max_caps}):
            for _ in range(25):
                models, ports = S.decode_particle(rng.random(2 * n_caps), n_caps, p)
                models, ports = S.resolve_adjacent_ports(models, ports, p)
                pl = [int(x) for x in ports]
                if (set(pl) & set(p.obs.tolist()) or not set(pl) <= set(p.pads.tolist())
                        or len(set(pl)) != len(pl)):
                    bad += 1
        check(bad == 0, f"{p.name:<16} decode+resolve: no observation port, no duplicate pad "
                        f"(75 random particles)")
    full = probs["mphy_full"]
    check(full.n_obs == len(C.MPHY_RAILS) == 5, "mphy_full has five observation ports (indices 0-4)")
    blocks = []
    for r in C.MPHY_RAILS:
        rp = probs[f"mphy_{r[0]}"]
        okr = (rp.n_obs == 1 and len(rp.pads) == r[4] and r[2] < full.n_obs
               and int(rp.global_index[0]) == r[2]
               and [int(g) for g in rp.global_index[1:]] == list(range(r[3], r[3] + r[4])))
        check(okr, f"rail {r[0]:<8} file N-1={len(rp.pads)} == layout count {r[4]}, "
                   f"pads map to mphy_full {r[3]}..{r[3] + r[4] - 1}")
        blocks += list(range(r[3], r[3] + r[4]))
    check(sorted(blocks) == list(range(full.n_obs, full.N)),
          f"rail pad blocks tile mphy_full pads {full.n_obs}..{full.N - 1} exactly "
          f"({len(blocks)} pads)")
    imp = probs["mphy_improvable"]
    check(imp.n_obs == len(C.IMPROVABLE_RAILS)
          and imp.max_caps == sum(_r[4] for _r in C.MPHY_RAILS if _r[0] in C.IMPROVABLE_RAILS),
          f"mphy_improvable: {imp.n_obs} observation ports, {imp.max_caps} pads")
    return probs


def show_targets():
    print("\n[3] targets (targets.json)")
    try:
        levels, table = S.load_targets()
    except S.TargetsNotSet as e:
        print(f"  [NOT READY] {e}")
        print("  The study REFUSES to start until every target is filled "
              "(python derive_targets.py, after Exp 1 has finished).")
        return False
    print("  levels (multiplier on the Exp 1 floor): "
          + ", ".join(f"x{lab}" for lab, _ in levels) + "   [stop at the tightest = first]")
    flags = S.get_flags()
    print(f"  {'key':<10}" + "".join(f"{'x' + lab + ' [ohm]':>14}" for lab, _ in levels))
    for k, row in table.items():
        print(f"  {k:<10}" + "".join(
            f"{row[lab]:>13.5g}" + ("*" if flags.get(k, {}).get(lab) else " ") for lab, _ in levels))
    flagged = [f"{k} x{lab}" for k, row in flags.items() for lab, v in row.items() if v]
    print("  * = noise-flagged level (target below the median Exp 1 floor, i.e. inside the floor spread): "
          + (", ".join(flagged) if flagged else "none"))
    return True


def show_plan(probs, ready):
    print("\n[4] planned cells")
    n_runs_total = 0
    for stage in (1, 2):
        sc = C.STAGES[stage]
        print(f"  Stage {stage}: {sc['runs']} runs/cell, {sc['n_particles']} particles x "
              f"{sc['n_iters']} iterations" + ("" if stage == 1 else
              f"  (survivor cells from Stage 1 only; {len(C.SCOPES) * len(C.BANDS)} cells max)"))
    print(f"  {'scope':<5}{'label':<22}{'problems':<46}{'objective':<10}{'max_caps':<9}")
    cells = 0
    for scope in C.SCOPES:
        ps = [probs[k] for k in ({"A1": [f"mphy_{r[0]}" for r in C.MPHY_RAILS],
                                  "A2": ["mphy_full"], "A3": ["mphy_improvable"],
                                  "A4": ["ddr21"], "A5": ["ddr_full"]}[scope])]
        desc = ", ".join(f"{p.name}(N={p.N},obs={p.n_obs})" for p in ps)
        print(f"  {scope:<5}{C.SCOPE_LABEL[scope]:<22}{desc[:44]:<46}"
              f"{ps[0].objective:<10}{'/'.join(str(p.max_caps) for p in ps):<9}")
        for band in C.BANDS:
            cells += 1
            n_runs_total += C.STAGES[1]["runs"] * len(ps)
    print(f"  Stage 1 cells (scope x band): {cells}   ->   {n_runs_total} PSO runs "
          f"(A1 counts its five rails separately)")
    print("  bands: " + ", ".join(f"{b}={C.BAND_LABEL[b]}" for b in C.BANDS)
          + f"   ranking band: {C.CARE_BAND}")
    print("  Grid 2 (band = Grid 1 winner): sizes "
          + ", ".join(f"N={n}:{k}" for n, k in C.GRID2_SIZES.items()))
    print(f"    PRIMARY: wall-time for {C.FIXED_EVALS} fixed evaluations (identical configs, 1.."
          f"{C.FIXED_EVAL_CAPS} capacitors) for {C.GRID2_TIMING_METHODS}; "
          f"{[m for m in C.APPROXIMATE_METHODS]} is an APPROXIMATION: timing + series fallback rate only")
    print(f"    SECONDARY: PSO runs of the exact methods {C.GRID2_PSO_METHODS}, {C.GRID2_RUNS} runs each "
          f"(pure_python {C.GRID2_PURE_RUNS} runs at N in {C.GRID2_PURE_SIZES} only, other sizes "
          f"extrapolated by n^3), stop level '{C.GRID2_STOP_LEVEL}', cap {C.GRID2_TIME_LIMIT_S}s/run, "
          f"{'skip a size whose first numpy run hits the cap' if C.GRID2_SKIP_UNCONVERGED else 'no skipping'}")
    if ready:
        S.set_require_targets(True)
        for n, k in C.GRID2_SIZES.items():
            p = S.build_problem(k)
            lab = p.stop_label(C.GRID2_STOP_LEVEL)
            print(f"      N={n:<3} {p.name:<12} objective={p.objective:<5} PSO stop level x{lab}"
                  + ("  [all levels flagged]" if p.level_flagged[lab] else ""))
    print(f"  threads: STUDY_THREADS={C.THREADS}; BLAS threads: "
          + ", ".join(f"{v}={os.environ.get(v)}" for v in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS")))
    print(f"  writes only under {C.HERE}")


def run():
    print("DecapStudy2 --validate (no PSO is run, nothing is written)")
    check_files_and_shapes()
    if _fails:
        print(f"\nRESULT: FAILED ({len(_fails)} check(s)); fix the data before going on.")
        return 1
    probs = check_port_rules()
    ready = show_targets()
    show_plan(probs, ready)
    if _fails:
        print(f"\nRESULT: FAILED ({len(_fails)} check(s)):")
        for m in _fails:
            print("  - " + m)
        return 1
    if not ready:
        print("\nRESULT: data and port rules OK, but targets.json is not filled in -- "
              "the study will refuse to start.")
        return 3
    print("\nRESULT: OK -- data, port rules and targets are all in place.")
    return 0
