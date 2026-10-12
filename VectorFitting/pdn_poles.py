"""
Vector-fit a PDN's input impedance and read its resonances off the poles.

This is where vector fitting earns its place. A PDN seen from one port IS a
rational function of s -- it is a finite LC network with loss -- so Z11(s) has
genuine pole pairs, and each pair is an anti-resonance: the peaks the decoupling
capacitors have to damp.

Peak-picking a sampled |Z11| curve gives you frequency and height. The poles give
you frequency, damping (Q) and residue, which is strictly more:

  * Q says how narrow the peak is, and therefore whether one capacitor near that
    frequency can cover it or whether it needs several spread around it.
  * The residue says how much of the peak height that pole is responsible for,
    so overlapping resonances can be separated instead of read as one hump.
  * The pole frequency is not quantised to the frequency grid, so a sharp peak
    between two samples is still located correctly.

Those three numbers are what the decap buckets get matched against.

Note on direction: the PDN's Z11 has poles at its anti-resonances, while a
capacitor's Z has a ZERO at its self-resonance (the pole pair is in its Y). So
fit Z for the PDN and Y for a part -- see decap_classify.py.

    python pdn_poles.py                                # default: ddr21
    python pdn_poles.py --board MPHY --prob mer1 --order 20
    python pdn_poles.py --prob ddr_full --plot
"""
import os, sys, json
import numpy as np
import scipy.io as sio
import vectorfit as vf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PDN_DATA", os.path.join(os.path.dirname(HERE), "Data"))

BOARDS = {"VDDQ_DDR3": ["ddr21", "ddr_full"],
          "MPHY": ["mer1", "mer2", "pll1v0", "pll1v8", "mphyvdd", "mphy_full"]}


def load_z11(board, prob, port=0):
    y = np.transpose(sio.loadmat(os.path.join(DATA, board, f"y_sp_{prob}.mat"))["y"],
                     (2, 0, 1))
    f = sio.loadmat(os.path.join(DATA, board, "freq_sp.mat"))["freq"].ravel()
    return f, np.linalg.inv(y)[:, port, port]


def fit_z11(freq, z11, order=24, band=None, n_iter=12, verbose=False):
    """Fit over `band` = (lo_hz, hi_hz), default the whole grid.

    Order is in POLE PAIRS. A PDN of N ports has at most a few dozen significant
    resonances in band; 20-30 pairs is usually enough, and the fit error tells
    you when it is not. `fit_lin=True` keeps the s*e term, which absorbs the
    series inductance that makes |Z| rise at the top of the band.
    """
    m = np.ones(len(freq), bool) if band is None else \
        (freq >= band[0]) & (freq <= band[1])
    return vf.vector_fit(freq[m], z11[m], n_pairs=order, n_iter=n_iter,
                         fit_const=True, fit_lin=True, weight="inverse",
                         verbose=verbose), m


def dominant_peaks(result, top=None, min_contrib_frac=0.01):
    """Poles ranked by how much they contribute to the impedance.

    `peak_contrib` = |residue| / sigma is the height a weakly damped pole adds at
    its own frequency. Poles contributing less than `min_contrib_frac` of the
    largest are dropped: a high-order fit always produces a few near-cancelling
    pole/zero pairs that are numerical, not physical.
    """
    rows = [r for r in result.summary() if not r["is_real"] and r["f0_hz"] > 0]
    if not rows:
        return []
    mx = max(r["peak_contrib"] for r in rows)
    rows = [r for r in rows if r["peak_contrib"] >= min_contrib_frac * mx]
    rows.sort(key=lambda r: -r["peak_contrib"])
    rows = rows[:top] if top else rows
    # numpy scalars are not JSON-serialisable; cast on the way out
    return [{k: (bool(v) if isinstance(v, (bool, np.bool_))
                 else float(v) if isinstance(v, (int, float, np.floating, np.integer))
                 else v) for k, v in r.items()} for r in rows]


def measured_peaks(freq, z11):
    """Local maxima of |Z11| on the sampled grid, for comparison."""
    a = np.abs(z11)
    k = np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1
    return [{"f_hz": float(freq[i]), "z_ohm": float(a[i])} for i in
            sorted(k, key=lambda i: -a[i])]


def main():
    args = sys.argv[1:]
    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default
    board = opt("--board", "VDDQ_DDR3")
    prob = opt("--prob", "ddr21")
    order = int(opt("--order", 24))
    lo = float(opt("--lo", 0)) or None
    hi = float(opt("--hi", 0)) or None

    freq, z11 = load_z11(board, prob)
    band = (lo or freq[0], hi or freq[-1])
    print(f"{board}/{prob}: {len(freq)} points, {freq[0]:.4g}-{freq[-1]:.4g} Hz")
    print(f"bare |Z11|: min {np.abs(z11).min():.5g}  peak {np.abs(z11).max():.5g} "
          f"ohm @ {freq[np.abs(z11).argmax()]:.4g} Hz")

    res, m = fit_z11(freq, z11, order=order, band=band, verbose="--verbose" in args)
    print(f"\nvector fit: {order} pole pairs over {band[0]:.4g}-{band[1]:.4g} Hz, "
          f"relative RMS {res.rms:.4e}, {res.iters} iterations")
    if res.rms > 0.05:
        print("  WARNING: >5% relative error -- raise --order before trusting "
              "these poles")

    peaks = dominant_peaks(res, top=12)
    print(f"\ndominant resonances ({len(peaks)} above 1% of the largest)")
    print(f"{'f0 (Hz)':>12} {'Q':>9} {'bandwidth':>12} {'contrib':>12}  suggested decap SRF")
    for r in peaks:
        bw = r["bandwidth_hz"]
        print(f"{r['f0_hz']:>12.5g} {r['Q']:>9.2f} {bw:>12.4g} "
              f"{r['peak_contrib']:>12.5g}  {r['f0_hz']:.3g} Hz "
              f"({_bucket(r['f0_hz'])})")

    # restrict to the fitted band -- comparing in-band poles against maxima
    # outside the fit is meaningless
    mp = [p for p in measured_peaks(freq, z11)
          if band[0] <= p["f_hz"] <= band[1]][:6]
    print(f"\nsampled-curve local maxima, for comparison")
    for p in mp:
        near = min(peaks, key=lambda r: abs(np.log10(r["f0_hz"] / p["f_hz"]))) if peaks else None
        tag = (f"  nearest pole {near['f0_hz']:.4g} Hz "
               f"({100*abs(near['f0_hz']-p['f_hz'])/p['f_hz']:.1f}% off)") if near else ""
        print(f"  {p['f_hz']:>12.5g} Hz  |Z| {p['z_ohm']:.5g} ohm{tag}")

    out = os.path.join(HERE, f"poles_{board}_{prob}.json")
    with open(out, "w") as fh:
        json.dump({"board": board, "problem": prob, "order_pairs": order,
                   "band_hz": list(band), "rms": res.rms,
                   "poles_real": [float(p.real) for p in res.poles],
                   "poles_imag": [float(p.imag) for p in res.poles],
                   "residues_real": [float(r.real) for r in res.residues],
                   "residues_imag": [float(r.imag) for r in res.residues],
                   "d": res.d, "e": res.e,
                   "dominant": peaks, "sampled_maxima": mp[:12]}, fh, indent=1)
    print(f"\nwrote {out}")

    if "--plot" in args:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fit = res(freq)
        plt.figure(figsize=(10, 5))
        plt.loglog(freq, np.abs(z11), lw=1.8, label="measured")
        plt.loglog(freq, np.abs(fit), lw=1.2, ls="--", label=f"VF, {order} pairs")
        for r in peaks[:8]:
            plt.axvline(r["f0_hz"], color="0.6", lw=.8, ls=":")
        plt.xlabel("Frequency (Hz)"); plt.ylabel(r"$|Z_{11}|$ ($\Omega$)")
        plt.title(f"{board}/{prob} - vector fit, dotted = dominant poles")
        plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
        png = os.path.join(HERE, f"poles_{board}_{prob}.png")
        plt.savefig(png, dpi=160); print(f"wrote {png}")


def _bucket(f):
    for lo, hi, n in ((1e3, 1e4, "1-10kHz"), (1e4, 1e5, "10-100kHz"),
                      (1e5, 1e6, "100kHz-1MHz"), (1e6, 1e7, "1-10MHz"),
                      (1e7, 1e8, "10-100MHz"), (1e8, 1e9, "100MHz-1GHz")):
        if lo <= f < hi:
            return n
    return "out of band"


if __name__ == "__main__":
    main()
