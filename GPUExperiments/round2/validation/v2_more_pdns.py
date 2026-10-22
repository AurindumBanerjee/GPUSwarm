"""
Validation V2 -- do the speed-ups and the search hold on the real SP datasets, not only on the reference PDN?

Part A: for every dataset in DATASETS (default mer2, pll1v0, mphyvdd, ddr21), numpy (NUM_RUNS_NUMPY runs) and fast (NUM_RUNS
runs) at targets relative to the bare peak (TARGET_FRACS x bare peak, default 0.6, 0.5, 0.4), band limited to SP_FMAX
(default 50 MHz, band-edge problem). Reported: speed-up on the paired runs, identical-trajectory check, success, caps,
before/after impedance plots.
Part B (MPHY only): the five rails are electrically separate but share one package. The best fast placement of every MPHY rail
in Part A (lowest target reached) is mapped into the full 73-port network (y_sp_mphy_full; port map from Data/MPHY/README.md:
rail r observation = port r, pads contiguous after the 5 observation ports), ALL rails' capacitors are installed at once and
|Z| at every rail's observation port is compared with the single-rail result. Plot: bare full network vs all caps installed.
MER1 and PLL1V8 are weakly coupled to their pads (decaps barely move them); include them with DATASETS=mer1,... if wanted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r2_common as R  # noqa: E402
import numpy as np
import scipy.io as sio
import torch
import gpuexp_common as C
from gpuexp_common import S, DEVICE

EXPERIMENT = "v2_more_pdns"
DATASETS = R.env_list("DATASETS", "mer2,pll1v0,mphyvdd,ddr21")
NUM_RUNS_NUMPY = int(os.environ.get("NUM_RUNS_NUMPY", "2" if S["QUICK"] else "5"))
RAIL_ORDER = ["mer1", "mer2", "pll1v0", "pll1v8", "mphyvdd"]          # port-block order in the full 73-port file


def floor_targets(name, info):
    """TARGET_MODE=floor: FLOOR_MULTS x the achievable floor of the rail (DecapStudy2/targets.json, peak |Z| below 50 MHz from full-budget
    runs). Relative-to-bare targets proved unreachable for the MPHY rails (their decaps cannot lower the peak that much)."""
    import json
    fl = json.load(open(os.path.join(R.ROOT, "DecapStudy2", "targets.json")))["floors_ohm"]
    key = {"ddr79": "ddr_full"}.get(name, name)
    return [round(m * fl[key], 6) for m in R.env_list("FLOOR_MULTS", "1.5,1.25,1.1", float)]


def part_b(run, best, lines):
    full_path = R._D("MPHY", "y_sp_mphy_full.mat")
    if not best or not os.path.exists(full_path):
        return
    # rail sizes -> pad offsets in the full network
    sizes = {r: sio.whosmat(R.DATASETS[r][0])[0][1][0] for r in RAIL_ORDER}
    offs, o = {}, len(RAIL_ORDER)
    for r in RAIL_ORDER:
        offs[r] = o
        o += sizes[r] - 1
    any_rail = next(iter(best))
    yfull = np.transpose(sio.loadmat(full_path)["y"], (2, 0, 1))
    rp, _ = R.load_problem(any_rail)                                  # same band / library as the rail runs
    full_freq = sio.loadmat(R.DATASETS[any_rail][2])["freq"].ravel()
    keep = np.isin(full_freq, rp.freq)
    fullp = C.Problem(y=yfull[keep], d=rp.d_np, freq=rp.freq, label="mphy_full")
    models, ports = [], []
    for r, (m, p, thr, single) in best.items():
        models += list(m)
        ports += [offs[r] + q - 1 for q in p]                          # rail-file port q (1..n-1) -> full index
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(best), figsize=(4.2 * len(best), 3.8), squeeze=False)
    for a, (r, (m, p, thr, single)) in zip(ax[0], best.items()):
        fullp.target_port = RAIL_ORDER.index(r)
        bare = R.z11_curve(fullp, None, None)
        own = R.z11_curve(fullp, list(m), [offs[r] + q - 1 for q in p])
        allc = R.z11_curve(fullp, models, ports)
        lines.append(f"Part B {r}: bare {bare.max():.4g} | own caps only (full net) {own.max():.4g} | single-rail run {single:.4g} | "
                     f"ALL rails' caps {allc.max():.4g} | target {thr:.4g} | still met: {allc.max() <= thr * (1 + 1e-9)}")
        a.loglog(fullp.freq, bare, color="0.45", label="bare (full network)")
        a.loglog(fullp.freq, allc, label=f"all {len(models)} caps installed")
        a.axhline(thr, color="crimson", ls="--", lw=1, label="target")
        a.set_title(r)
        a.grid(True, which="both", alpha=.3)
        a.legend(fontsize=7)
        a.set_xlabel("frequency (Hz)")
    ax[0][0].set_ylabel("|Z| at rail observation port (ohm)")
    fig.tight_layout()
    fig.savefig(os.path.join(run.run_dir, "plots", "mphy_all_rails_combined.png"), dpi=150)
    plt.close(fig)


def main():
    run = C.Run(EXPERIMENT, __file__, {"DATASETS": DATASETS, "NUM_RUNS_NUMPY": NUM_RUNS_NUMPY, "SP_FMAX": R.SP_FMAX,
                                       "LIB_SOURCE": R.LIB_SOURCE, "TARGET_MODE": os.environ.get("TARGET_MODE", "relative")})
    C.init_device_log(run)
    import pandas as pd
    lines, best = [], {}
    max_caps0 = S["MAX_CAPS"]
    for name in DATASETS:
        prob, info = R.load_problem(name)
        S["MAX_CAPS"] = min(max_caps0, prob.n - 1)          # never more capacitors than pads (stages beyond would stack parts on one pad)
        run.lg.info("STAGE | %s", info)
        lines.append(f"{name}: {info['ports']} ports, {info['models']} models, {info['n_freq']} freqs "
                     f"{info['f_lo']:.3g}-{info['f_hi']:.3g} Hz, bare peak {info['bare_peak']:.4g} ohm @ {info['bare_peak_f']:.3g} Hz")
        targets = floor_targets(name, info) if os.environ.get("TARGET_MODE") == "floor" else R.targets_for(info)
        npy, fast = C.SeqNumpyEvaluator(prob), C.SwarmEvaluator(prob, True, True)
        C.warmup(prob, [npy, fast])
        recs = R.loop_methods(run, prob, {"numpy": npy}, targets, NUM_RUNS_NUMPY, reference="numpy",
                              label_prefix=f"{name}/", extra={"dataset": name})
        recs += R.loop_methods(run, prob, {"fast": fast}, targets, S["NUM_RUNS"], reference="__none__",
                               label_prefix=f"{name}/", extra={"dataset": name})
        df = pd.DataFrame(recs)
        for thr, g in df.groupby("threshold"):
            a, b = g[g.method == f"{name}/numpy"], g[g.method == f"{name}/fast"]
            bb = b[b.run.isin(a.run)].set_index("run")
            same = sum(abs(bb.best_minz[r] - v) <= 1e-9 * max(1, v) for r, v in a.set_index("run").best_minz.items())
            sp = a.total_time_s.mean() / bb.total_time_s.mean()
            lines.append(f"  target {thr:.4g} ({thr / info['bare_peak']:.2f} x bare): fast success {int(b.success.sum())}/{len(b)}, "
                         f"caps {b.caps_at_target.dropna().mean():.2f}, best {b.best_minz.mean():.4g} | speed-up vs numpy {sp:.1f}x "
                         f"(identical on {same}/{len(a)})")
        R.plot_best_runs(prob, recs, os.path.join(run.run_dir, "plots"))
        # Part B input: best fast placement at the lowest target that was met (else the best overall)
        if name in RAIL_ORDER:
            f = df[(df.method == f"{name}/fast") & df.best_models.notna()]
            ok = f[f.success == True]                                         # noqa: E712
            pick = (ok[ok.threshold == ok.threshold.min()] if len(ok) else f).sort_values("best_minz").iloc[0]
            best[name] = (pick.best_models, pick.best_ports, pick.threshold, pick.best_minz)
    part_b(run, best, lines)
    run.summarise(extra_cols=("cache_hits", "pruned"))
    R.write_lines(run, "pdn_summary.txt", lines)
    run.finish()


if __name__ == "__main__":
    main()
