"""
Analysis: Stage-1 screening, Grid 1 matrix (3-band re-scoring), Grid 2 timing,
plots, report. Reads results_long.jsonl only, so it can run at any point.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import study_config as C
import experiments as X

BANDS = list(C.BANDS)
FIGS = C.FIG_DIR


def md(df, floatfmt="{:.4g}"):
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                         else ("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ============================================================ tables

def per_rail_long():
    """One row per (run record, observation rail) with peak/target on each band."""
    rows = []
    for r in X.load_records("run"):
        if r.get("grid") == "G2":
            continue
        for k, rail in enumerate(r["rails"]):
            row = {"stage": r["stage"], "scope": r["scope"], "band": r["band"],
                   "run_id": r["run_id"], "rec_rail": r["rail"], "rail": rail,
                   "target_ohm": r["targets_ohm"][k]}
            for b in BANDS:
                pk = r["rescored"][b]["peaks_ohm"][k]
                row[f"peak_{b}"] = pk
                row[f"ratio_{b}"] = pk / r["targets_ohm"][k]
            rows.append(row)
    return pd.DataFrame(rows)


def run_level():
    """One row per (stage, scope, band, run_id). A1's five rail runs are combined."""
    recs = [r for r in X.load_records("run") if r.get("grid") != "G2"]
    if not recs:
        return pd.DataFrame()
    labels = list(recs[0]["levels"])                  # threshold levels, tightest first
    rows = []
    for r in recs:
        row = {
            "stage": r["stage"], "scope": r["scope"], "band": r["band"], "run_id": r["run_id"],
            "rec_rail": r["rail"], "success": r["success"], "caps_used": r["caps_used"],
            "caps_needed": r["caps_needed"], "conv_iter": r["conv_iter"],
            "n_evals": r["n_evals"], "wall_s": r["wall_s"]}
        for lab in labels:
            row[f"hit_{lab}"] = r["levels"][lab]["hit"]
            row[f"caps_{lab}"] = r["levels"][lab]["caps_needed"]
            row[f"flag_{lab}"] = r["levels"][lab].get("inside_noise", False)
        rows.append(row)
    df = pd.DataFrame(rows)
    keys = ["stage", "scope", "band", "run_id"]
    # A1 runs five rails per run: a level counts as hit only if every rail hit it,
    # and the capacitors needed add up across the five independent budgets
    _sum_all = lambda s: s.sum() if s.notna().all() else np.nan
    agg = df.groupby(keys).agg(
        success=("success", "all"), caps_used=("caps_used", "sum"),
        caps_needed=("caps_needed", _sum_all),
        conv_iter=("conv_iter", "max"), n_evals=("n_evals", "sum"),
        wall_s=("wall_s", "sum"), n_rec=("rec_rail", "count"),
        **{f"hit_{lab}": (f"hit_{lab}", "all") for lab in labels},
        **{f"caps_{lab}": (f"caps_{lab}", _sum_all) for lab in labels},
        **{f"flag_{lab}": (f"flag_{lab}", "any") for lab in labels}).reset_index()
    pl = per_rail_long()
    for b in BANDS:
        g = pl.groupby(keys)[f"ratio_{b}"].max().rename(f"R_{b}")
        agg = agg.merge(g.reset_index(), on=keys, how="left")
        imp = pl[pl["rail"].isin(C.IMPROVABLE_RAILS)]
        gi = imp.groupby(keys)[f"ratio_{b}"].max().rename(f"IMP_{b}")
        agg = agg.merge(gi.reset_index(), on=keys, how="left")
    # A1 is only complete when all five rails are present for that run
    need = agg["scope"].map(lambda s: 5 if s == "A1" else 1)
    return agg[agg["n_rec"] >= need].reset_index(drop=True)


def stage1_table(care=None):
    care = care or C.CARE_BAND
    rl = run_level()
    rl = rl[rl["stage"] == 1] if not rl.empty else rl
    if rl.empty:
        return pd.DataFrame(), {}
    g = rl.groupby(["scope", "band"]).agg(
        n_runs=("run_id", "count"),
        **{f"med_R_{b}": (f"R_{b}", "median") for b in BANDS},
        success=("success", "mean"), caps_used=("caps_used", "median")).reset_index()
    g["score"] = g[f"med_R_{care}"]
    g["row_best"] = g.groupby("scope")["score"].transform("min")
    g["x_vs_best"] = g["score"] / g["row_best"]
    g["keep"] = g["x_vs_best"] <= C.SCREEN_FACTOR
    surv = {s: sorted(g[(g.scope == s) & g.keep]["band"]) for s in C.SCOPES}
    return g, surv


def write_stage1(care=None):
    care = care or C.CARE_BAND
    g, surv = stage1_table(care)
    if g.empty:
        return None
    os.makedirs(C.OUT_ROOT, exist_ok=True)
    g.to_csv(os.path.join(C.OUT_ROOT, "stage1_table.csv"), index=False)
    with open(os.path.join(C.OUT_ROOT, "survivors.json"), "w") as fh:
        json.dump({"care_band": care, "survivors": surv}, fh, indent=2)
    show = g[["scope", "band", "n_runs"] + [f"med_R_{b}" for b in BANDS]
             + ["x_vs_best", "keep", "success", "caps_used"]].copy()
    show.insert(1, "label", show["scope"].map(C.SCOPE_LABEL))
    text = (f"# Stage 1 screening\n\nRanking band: **{care}** (median over runs of the "
            f"re-scored peak/Ztarget; <=1 meets target). Cells with x_vs_best > "
            f"{C.SCREEN_FACTOR} are dropped.\n\n" + md(show))
    with open(os.path.join(C.OUT_ROOT, "stage1_table.md"), "w") as fh:
        fh.write(text)
    return text


def survivors():
    p = os.path.join(C.OUT_ROOT, "survivors.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        s = json.load(fh)["survivors"]
    return [(sc, b) for sc in C.SCOPES for b in s.get(sc, [])]


def grid1_matrix(stage=2):
    rl = run_level()
    rl = rl[rl["stage"] == stage] if not rl.empty else rl
    if rl.empty:
        return pd.DataFrame()
    labels = [c[4:] for c in rl.columns if c.startswith("hit_")]
    g = rl.groupby(["scope", "band"]).agg(
        n_runs=("run_id", "count"), caps_used=("caps_used", "median"),
        **{f"R_{b}": (f"R_{b}", "median") for b in BANDS},
        **{f"IMP_{b}": (f"IMP_{b}", "median") for b in BANDS},
        caps_needed=("caps_needed", "median"), conv_iter=("conv_iter", "median"),
        success_rate=("success", "mean"),
        # one success rate and median capacitor count per threshold level (x = multiplier on the floor)
        **{f"succ_x{lab}": (f"hit_{lab}", "mean") for lab in labels},
        **{f"caps_x{lab}": (f"caps_{lab}", "median") for lab in labels},
        # True when the level's target is below the median Exp 1 floor (inside the floor spread)
        **{f"flag_x{lab}": (f"flag_{lab}", "any") for lab in labels}).reset_index()
    return g


NOISE_NOTE = ("\n\u2020 = noise-flagged level: its target lies below the median achieved Exp 1 floor, i.e. "
              "inside the floor spread. A low success rate there means *target inside measurement "
              "noise*, not a difference between scopes or bands.\n")


def grid1_display(g):
    """Grid 1 matrix for the report: success rate at a noise-flagged level gets a dagger and the
    flag columns are folded away (they stay in grid1_matrix.csv)."""
    d = g.copy()
    for c in [c for c in g.columns if c.startswith("succ_x")]:
        lab = c[len("succ_x"):]
        fc = f"flag_x{lab}"
        flagged = list(g[fc]) if fc in g.columns else [False] * len(g)
        d[c] = [f"{v:.2f}\u2020" if bool(f) else f"{v:.2f}" for v, f in zip(g[c], flagged)]
    return d.drop(columns=[c for c in g.columns if c.startswith("flag_x")])


def rails_ohm_table(stage=2):
    pl = per_rail_long()
    if pl.empty:
        return pl
    pl = pl[pl["stage"] == stage]
    return pl.groupby(["scope", "band", "rail"]).agg(
        target_ohm=("target_ohm", "first"),
        **{f"peak_{b}_ohm": (f"peak_{b}", "median") for b in BANDS}).reset_index()


def winning_band(care=None, stage=2):
    care = care or C.CARE_BAND
    g = grid1_matrix(stage)
    if g.empty:
        g = grid1_matrix(1)
    if g.empty:
        return None
    col = f"R_{care}"
    g = g.copy()
    g["rel"] = g[col] / g.groupby("scope")[col].transform("min")
    sc = g.groupby("band")["rel"].apply(lambda s: float(np.exp(np.log(s).mean())))
    return sc.idxmin(), sc.to_dict()


def grid2_tables():
    """(primary, secondary, info).

    primary   -- wall-time for a FIXED evaluation count, every method (iterative = approximation:
                 timing + series fallback rate only, no quality column)
    secondary -- PSO runs of the exact methods: time-to-target (converged runs only) and placement
                 match against numpy (runs not truncated by the cap only)
    """
    tim = pd.DataFrame(X.load_records("eval_timing"))
    runs = [r for r in X.load_records("run") if r.get("grid") == "G2"]
    skips = {r["size_N"]: r for r in X.load_records("grid2_skip")}
    if tim.empty:
        return pd.DataFrame(), pd.DataFrame(), {"pure_python_exponent": float("nan")}
    pp = tim[tim.method == "pure_python"].sort_values("size_N")
    exponent = float("nan")
    if len(pp) >= 2:
        a_, b_ = pp.iloc[0], pp.iloc[-1]
        exponent = float(np.log(b_.sec_per_eval / a_.sec_per_eval) / np.log(b_.size_N / a_.size_N))
    ref = pp.iloc[-1] if len(pp) else None

    def t_pp(N):
        row = pp[pp.size_N == N]
        if len(row):
            return float(row.sec_per_eval.iloc[0]), "measured"
        if ref is None:
            return float("nan"), "n/a"
        return float(ref.sec_per_eval * (N / ref.size_N) ** 3), "extrapolated n^3"

    rows = []
    for N in sorted(tim.size_N.unique()):
        tpp, how = t_pp(N)
        K = int(tim[tim.size_N == N].n_evals.iloc[0])
        for m in C.GRID2_TIMING_METHODS:
            label = f"{m} (approximation)" if m in C.APPROXIMATE_METHODS else m
            te = tim[(tim.size_N == N) & (tim.method == m)]
            if m == "pure_python" and how != "measured":
                rows.append({"N": N, "method": label, "evals": K, "wall_s": tpp * K,
                             "ms_per_eval": 1e3 * tpp, "speedup_vs_pure": 1.0, "source": how})
                continue
            if te.empty:
                continue
            t = te.iloc[0]
            row = {"N": N, "method": label, "evals": K, "wall_s": float(t.total_s),
                   "ms_per_eval": 1e3 * float(t.sec_per_eval),
                   "speedup_vs_pure": tpp / float(t.sec_per_eval), "source": "measured"}
            if m in C.APPROXIMATE_METHODS:                     # no quality figure for an approximation
                row["series_fallback_rate"] = float(t.series["fallback_rate"])
            else:
                row["max_cost_dev_vs_numpy"] = float(t.max_rel_cost_dev_vs_numpy)
            rows.append(row)
    primary = pd.DataFrame(rows)

    exact = [r for r in runs if r["method"] not in C.APPROXIMATE_METHODS]
    rdf = pd.DataFrame([{
        "N": r["size_N"], "method": r["method"], "run_id": r["run_id"], "stop": r.get("stop_level"),
        "success": r["success"], "timed_out": r["timed_out"], "n_evals": r["n_evals"],
        "ttt": r["time_to_target_s"],
        "placement": tuple(sorted((c["pad"], c["model"]) for c in r["placement"]))} for r in exact])
    srows = []
    sizes = sorted(set(rdf.N) | set(skips)) if not rdf.empty else sorted(skips)
    for N in sizes:
        for m in C.GRID2_PSO_METHODS:
            mm = rdf[(rdf.N == N) & (rdf.method == m)] if not rdf.empty else rdf
            if mm.empty:
                continue
            nm = rdf[(rdf.N == N) & (rdf.method == "numpy")]
            conv = mm[mm.success]
            row = {"N": N, "method": m, "stop level": f"x{mm.stop.iloc[0]}", "runs": len(mm),
                   "reached": int(mm.success.sum()), "timeouts": int(mm.timed_out.sum()),
                   "n_evals_med": float(mm.n_evals.median()),
                   "time_to_target_s_med": float(conv.ttt.median()) if len(conv) else float("nan")}
            if m != "numpy":
                both = mm.merge(nm, on="run_id", suffixes=("", "_np"))
                both = both[~both.timed_out & ~both.timed_out_np]     # skip cap-truncated runs
                row["placement_match_vs_numpy"] = (
                    f"{int((both.placement == both.placement_np).sum())}/{len(both)}"
                    if len(both) else "n/a (cap-truncated)")
            srows.append(row)
        if N in skips:
            srows.append({"N": N, "method": "(PSO runs skipped)",
                          "stop level": f"x{skips[N]['stop_level']}",
                          "placement_match_vs_numpy": skips[N]["reason"]})
    return primary, pd.DataFrame(srows), {"pure_python_exponent": exponent}


# ============================================================ plots

def _stage_for_plots():
    df = run_level()
    return 2 if (not df.empty and (df.stage == 2).any()) else 1


def plot_per_run(stage):
    recs = [r for r in X.load_records("run") if r["stage"] == stage and r.get("grid") != "G2"]
    os.makedirs(os.path.join(FIGS, "per_run"), exist_ok=True)
    os.makedirs(os.path.join(FIGS, "convergence"), exist_ok=True)
    for scope in C.SCOPES:
        for band in BANDS:
            rr = [r for r in recs if r["scope"] == scope and r["band"] == band]
            if not rr:
                continue
            rails = sorted({r["rail"] for r in rr}, key=lambda x: x)
            for kind in ("per_run", "convergence"):
                fig, axes = plt.subplots(1, len(rails), figsize=(5.2 * len(rails), 4), squeeze=False)
                for ax, rail in zip(axes[0], rails):
                    sub = [r for r in rr if r["rail"] == rail]
                    s0 = sub[0]
                    raw, single = s0["objective"] == "raw", s0["n_obs"] == 1
                    unit = (("peak |Z11| (ohm)" if single else "max_k peak |Z_kk| (ohm)") if raw
                            else "max_k peak / Ztarget (tightest level)")
                    for r in sub:
                        if kind == "per_run":
                            ax.plot(r["curve_caps"], r["curve_cost"], marker="o", ms=3, lw=1, alpha=.7)
                        else:
                            h = [v for hh in r["histories"] for v in hh]
                            ax.plot(range(1, len(h) + 1), h, lw=1, alpha=.7)
                    # threshold levels: ohm lines for single-observation raw problems,
                    # ratio lines (multiplier / tightest multiplier) for the ratio objective
                    lv0 = list(s0["levels"].values())[0]["multiplier"]
                    for lab, lv in s0["levels"].items():
                        if raw and single:
                            ax.axhline(lv["target_ohm"][0], ls="--", lw=1, c="k", alpha=.6)
                        elif not raw:
                            ax.axhline(lv["multiplier"] / lv0, ls="--", lw=1, c="k", alpha=.6)
                    ax.set_yscale("log")
                    ax.set_xlabel("capacitors" if kind == "per_run" else "cumulative PSO iteration")
                    ax.set_ylabel(unit)
                    ax.set_title(f"{scope} {band} {rail if scope == 'A1' else ''}".strip())
                    ax.grid(alpha=.3)
                fig.tight_layout()
                fig.savefig(os.path.join(FIGS, kind, f"{scope}_{band}.png"), dpi=130)
                plt.close(fig)


def plot_heatmap(stage):
    g = grid1_matrix(stage)
    if g.empty:
        return
    nb = len(BANDS)
    fig, axes = plt.subplots(1, nb, figsize=(5 * nb, 3.8), sharey=True)
    allv = np.array([g[f"R_{b}"] for b in BANDS]).ravel()
    norm = LogNorm(vmin=max(np.nanmin(allv), 1e-3), vmax=np.nanmax(allv))
    for ax, sb in zip(axes, BANDS):
        M = np.full((len(C.SCOPES), len(BANDS)), np.nan)
        for i, s in enumerate(C.SCOPES):
            for j, ob in enumerate(BANDS):
                row = g[(g.scope == s) & (g.band == ob)]
                if len(row):
                    M[i, j] = row[f"R_{sb}"].iloc[0]
        im = ax.imshow(M, norm=norm, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(nb)); ax.set_xticklabels([f"opt {b}" for b in BANDS])
        ax.set_yticks(range(len(C.SCOPES))); ax.set_yticklabels(C.SCOPES)
        ax.set_title(f"scored on {sb} ({C.BAND_LABEL[sb]})")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=axes, label="median peak / Ztarget (<=1 meets target)")
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, "heatmap_scope_band.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_impedance(stage, band):
    """Bare vs optimised |Z_kk|(f), log-log, best placement of each scope at `band`."""
    import study_core as S
    f = S.freqs()
    recs = [r for r in X.load_records("run")
            if r["stage"] == stage and r["band"] == band and r.get("grid") != "G2"]
    os.makedirs(FIGS, exist_ok=True)
    for scope in C.SCOPES:
        rr = [r for r in recs if r["scope"] == scope]
        if not rr:
            continue
        best = {}
        for r in rr:
            if r["rail"] not in best or r["best_cost_own_band"] < best[r["rail"]]["best_cost_own_band"]:
                best[r["rail"]] = r
        items = []
        for rail, r in sorted(best.items()):
            prob = (S.build_problem(f"mphy_{rail}") if scope == "A1"
                    else S.scope_problems(scope)[0])
            cfg = [(int(c["model"]), int(c["pad"])) for c in r["placement"]]
            zb = np.abs(S.diag_z(prob, len(f), []))
            zo = np.abs(S.diag_z(prob, len(f), cfg))
            for k, name in enumerate(prob.rails):
                items.append((f"{scope} {name}", zb[:, k], zo[:, k], prob.targets[k], len(cfg)))
        n = len(items)
        cols = min(n, 3)
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(5.2 * cols, 3.8 * ((n + cols - 1) // cols)),
                                 squeeze=False)
        for ax, (title, zb, zo, tz, nc) in zip(axes.ravel(), items):
            ax.loglog(f, zb, label="bare")
            ax.loglog(f, zo, label=f"optimised ({nc} caps)")
            ax.axhline(tz, ls="--", c="k", lw=1, label="Ztarget")
            for hi in (20e6, 50e6):
                ax.axvline(hi, c="gray", lw=.6, ls=":")
            ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|Z| (ohm)")
            ax.set_title(f"{title}  (opt {band})"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"zf_{scope}_{band}.png"), dpi=130)
        plt.close(fig)


# ============================================================ report

def recommendation(care, stage):
    g = grid1_matrix(stage)
    if g.empty:
        return "_(no Grid 1 data yet)_"
    wb = winning_band(care, stage)
    band = wb[0]
    lines = [f"Ranking band: **{care}**. Winning band by geometric-mean relative score: "
             f"**{band}** ({C.BAND_LABEL[band]}); scores {json.dumps({k: round(v, 3) for k, v in wb[1].items()})}.\n"]
    at = g[g.band == band]
    mp = at[at.scope.isin(["A1", "A2", "A3"])]
    if len(mp):
        col = f"IMP_{care}"
        b = mp.sort_values(col).iloc[0]
        lines.append(f"- MPHY (compared on the improvable rails mer2/pll1v0/mphyvdd, the only set "
                     f"common to A1-A3): best scope **{b.scope} {C.SCOPE_LABEL[b.scope]}** at {band}, "
                     f"median {col}={b[col]:.3f}, success rate {b.success_rate:.2f}, median caps used "
                     f"{b.caps_used:.0f}.\n")
    dd = at[at.scope.isin(["A4", "A5"])]
    if len(dd):
        col = f"R_{care}"
        b = dd.sort_values(col).iloc[0]
        lines.append(f"- DDR3: best scope **{b.scope} {C.SCOPE_LABEL[b.scope]}** at {band}, median "
                     f"{col}={b[col]:.3f}, success rate {b.success_rate:.2f}, median caps used "
                     f"{b.caps_used:.0f}.\n")
    return "\n".join(lines)


def build_report():
    care = C.CARE_BAND
    stage = _stage_for_plots()
    parts = ["# Decap placement study (Exp 2) -- report\n",
             f"Ranking band (CARE_BAND): {care}. Targets come from targets.json (Exp 1 floors on "
             "the <=50 MHz band x each multiplier); the run stops at the tightest level and "
             "every level is reported (succ_x<m>, caps_x<m> columns). R_x / ratio = peak / "
             "Ztarget at the tightest level, so <=1 means the tightest level is met.\n"]
    try:
        import study_core as S
        levels, table = S.load_targets()
        flags = S.get_flags()
        tt = pd.DataFrame([{"target key": k,
                            **{f"x{lab} [ohm]": (f"{v[lab]:.5g}" + ("\u2020" if flags.get(k, {}).get(lab) else ""))
                               for lab, _ in levels}}
                           for k, v in table.items()])
        parts.append("\n## Targets used\n" + md(tt) + NOISE_NOTE)
    except Exception as e:                      # report must still build from the JSONL
        parts.append(f"\n_(targets.json not readable here: {e})_\n")
    s1 = os.path.join(C.OUT_ROOT, "stage1_table.md")
    if os.path.exists(s1):
        parts.append(open(s1).read())
    g = grid1_matrix(stage)
    if not g.empty:
        parts.append(f"\n## Grid 1 matrix (stage {stage}; R_x = median peak/Ztarget re-scored on band x)\n")
        parts.append(md(grid1_display(g)) + NOISE_NOTE)
        parts.append("\n### Per-rail median peaks (ohm)\n" + md(rails_ohm_table(stage)))
        g.to_csv(os.path.join(C.OUT_ROOT, "grid1_matrix.csv"), index=False)
        rails_ohm_table(stage).to_csv(os.path.join(C.OUT_ROOT, "grid1_rails_ohm.csv"), index=False)
    rr = os.path.join(C.OUT_ROOT, "rail_response.json")
    if os.path.exists(rr):
        rows = [{"rail": k, **{f"{b}_bare_ohm": v[b]["bare_peak_ohm"] for b in BANDS},
                 **{f"{b}_gain_%": v[b]["gain_pct"] for b in BANDS}}
                for k, v in json.load(open(rr))["rails"].items()]
        parts.append("\n## Rail responsiveness (best single model on every pad vs bare)\n"
                     "mer1 and pll1v8 barely respond: the observation port sits at the die and "
                     "package inductance hides the pads. They stay in A1/A2; A3 exists because "
                     "of them.\n" + md(pd.DataFrame(rows)))
    t1, t2, info = grid2_tables()
    if not t1.empty:
        parts.append("\n## Grid 2 -- wall-time for a FIXED evaluation count (primary)\n"
                     f"Every method evaluates the same {int(t1.evals.iloc[0])} (model, pad) configurations per size "
                     "in per-frequency loop mode. `iterative` is an APPROXIMATION (second-order series, "
                     "falls back to a full inverse where it diverges): timing and series fallback rate only, "
                     "no placement-match or quality claim. pure_python at N=73/79 is extrapolated by n^3 "
                     "from its measured scaling.\n" + md(t1))
        parts.append(f"\nMeasured pure_python scaling exponent (N=16 -> 21): "
                     f"{info['pure_python_exponent']:.2f} (n^3 assumed for N=73/79)\n")
        t1.to_csv(os.path.join(C.OUT_ROOT, "grid2_fixed_eval_timing.csv"), index=False)
    if not t2.empty:
        parts.append("\n## Grid 2 -- PSO runs, exact methods (secondary)\n"
                     "Stop level = the tightest level that is not noise-flagged. Time-to-target is the median "
                     "over converged runs only (blank = none converged within the 1800 s cap). Placement match "
                     "counts only runs that neither method had truncated by the cap; `iterative` is not run.\n"
                     + md(t2))
        t2.to_csv(os.path.join(C.OUT_ROOT, "grid2_pso_runs.csv"), index=False)
    for b in BANDS:
        p = os.path.join(C.OUT_ROOT, f"consistency_{b}.json")
        if os.path.exists(p):
            c = json.load(open(p))
            parts.append(f"\n## Consistency check ({b})\n{c['verdict']}  reduction check "
                         f"{c['reduction_check_max_rel_dev']:.2e}, worst combined peak deviation "
                         f"{c['worst_peak_rel_dev_combined']:.2e}\n" + md(pd.DataFrame(c["rails"])))
    parts.append("\n## Recommendation\n" + recommendation(care, stage))
    text = "\n".join(parts)
    with open(os.path.join(C.OUT_ROOT, "report.md"), "w") as fh:
        fh.write(text)
    return text


def make_plots():
    stage = _stage_for_plots()
    plot_per_run(stage)
    plot_heatmap(stage)
    wb = winning_band(C.CARE_BAND, stage)
    if wb:
        plot_impedance(stage, wb[0])


if __name__ == "__main__":
    write_stage1()
    make_plots()
    print(build_report())
