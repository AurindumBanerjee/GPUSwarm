"""
Characterise every decap in the library and bucket it by self-resonant frequency.

A mounted decoupling capacitor behaves as a series RLC over the band that
matters: Z(f) = ESR + j*w*ESL + 1/(j*w*C). Its impedance falls as 1/wC, reaches
a minimum of ESR at the self-resonant frequency SRF = 1/(2*pi*sqrt(ESL*C)), and
rises as w*ESL above it. Above SRF the part is an inductor, not a capacitor.

That makes SRF the natural label: a capacitor is only useful for damping a PDN
resonance that sits near its own SRF. Bucketing the library by SRF, and the PDN
peaks by frequency, turns decap selection from a 6695-way search per slot into a
choice within one bucket.

Extraction is analytic rather than fitted: C from the capacitive asymptote, ESL
from the inductive asymptote, ESR from the minimum. That is faster than vector
fitting 6695 parts and gives the same three numbers, since a series RLC has
exactly one pole pair. Set --verify to cross-check a sample against
vectorfit.vector_fit, which should agree on SRF and Q.

    python decap_classify.py                 # write decap_classes.json + table
    python decap_classify.py --verify 25     # also VF-check 25 random parts
"""
import os, sys, json
import warnings
import numpy as np
import scipy.io as sio

warnings.filterwarnings('ignore', r'All-NaN', RuntimeWarning)
warnings.filterwarnings('ignore', r'Mean of empty slice', RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PDN_DATA", os.path.join(os.path.dirname(HERE), "Data"))
LIB = os.path.join(DATA, "Decaps", "combined_decaps.mat")
GRID = os.path.join(DATA, "Decaps", "sp_freq.mat")
OUT = os.path.join(HERE, "decap_classes.json")


def load_library():
    m = sio.loadmat(LIB)
    f = sio.loadmat(GRID)["freq"].ravel()
    return (f, m["decaps"], [str(x[0][0]) for x in m["names"]],
            np.asarray([str(x[0][0]) for x in m["source"]]))


def characterise(freq, Y):
    """(C, ESR, ESL, SRF, Q) per model, from the asymptotes of Z = 1/Y.

    C   : median of -1/(w*Im Z) over the decade below SRF where the part is
          still capacitive. Using a median rather than one point makes it
          robust to a noisy low-frequency sample.
    ESL : median of Im(Z)/w over the decade above SRF.
    ESR : Re(Z) at the |Z| minimum.
    SRF : the |Z| minimum, refined by parabolic interpolation in log-f so it is
          not quantised to the frequency grid.
    Q   : (1/ESR) * sqrt(ESL/C), the series-RLC quality factor. High Q means a
          deep, narrow notch; low Q a shallow, broad one.
    """
    w = 2 * np.pi * freq
    Z = 1.0 / Y
    absZ = np.abs(Z)
    n, F = Z.shape

    k = np.argmin(absZ, axis=1)
    lf = np.log10(freq)
    srf = np.empty(n)
    for i in range(n):                                  # parabolic refinement
        j = k[i]
        if 0 < j < F - 1:
            y0, y1, y2 = np.log10(absZ[i, j - 1:j + 2])
            den = (y0 - 2 * y1 + y2)
            dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            dx = float(np.clip(dx, -1, 1))
            srf[i] = 10 ** (lf[j] + dx * (lf[j] - lf[j - 1]))
        else:
            srf[i] = freq[j]

    imZ, w2 = Z.imag, w[None, :]
    cap_band = (freq[None, :] < srf[:, None] / 2) & (freq[None, :] > srf[:, None] / 30)
    ind_band = (freq[None, :] > srf[:, None] * 2) & (freq[None, :] < srf[:, None] * 30)
    fall = freq[None, :] < srf[:, None]
    rise = freq[None, :] > srf[:, None]
    cap_band = np.where(cap_band.any(axis=1, keepdims=True), cap_band, fall)
    ind_band = np.where(ind_band.any(axis=1, keepdims=True), ind_band, rise)

    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = np.where(cap_band, -1.0 / (w2 * imZ), np.nan)
        Lm = np.where(ind_band, imZ / w2, np.nan)
    C = np.nanmedian(np.where(Cm > 0, Cm, np.nan), axis=1)
    L = np.nanmedian(np.where(Lm > 0, Lm, np.nan), axis=1)
    ESR = Z[np.arange(n), k].real
    ESR = np.where(ESR > 0, ESR, absZ[np.arange(n), k])

    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.sqrt(L / C) / ESR
    return C, ESR, L, srf, Q


def decade_buckets(srf, edges=None):
    """Label each part by the decade its SRF falls in."""
    if edges is None:
        edges = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    idx = np.digitize(srf, edges)
    names = []
    for i in idx:
        lo = edges[i - 1] if i > 0 else 0.0
        hi = edges[i] if i < len(edges) else np.inf
        names.append(f"{_hz(lo)}-{_hz(hi)}")
    return np.array(names), idx


def _hz(v):
    if v == 0: return "DC"
    if not np.isfinite(v): return "inf"
    for s, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if v >= s: return f"{v/s:g}{u}"
    return f"{v:g}Hz"


def bucket_for_peaks(srf, peak_freqs, tol_octaves=1.0):
    """Which parts are usable against each PDN peak.

    A capacitor can damp a resonance only near its own SRF: below SRF it is
    capacitive and still has useful low impedance, above SRF it is an inductor
    and makes things worse. `tol_octaves` sets how far either side counts.
    Returns {peak_index: array of model indices}, nearest-SRF first.
    """
    out = {}
    for i, fp in enumerate(peak_freqs):
        lo, hi = fp / 2 ** tol_octaves, fp * 2 ** tol_octaves
        sel = np.where((srf >= lo) & (srf <= hi))[0]
        out[i] = sel[np.argsort(np.abs(np.log10(srf[sel] / fp)))] if len(sel) else sel
    return out


def main():
    freq, Y, names, source = load_library()
    print(f"library {Y.shape[0]} models x {Y.shape[1]} freqs, "
          f"{freq[0]:.4g}-{freq[-1]:.4g} Hz")
    C, ESR, L, srf, Q = characterise(freq, Y)

    in_band = (srf > freq[0] * 1.5) & (srf < freq[-1] / 1.5)
    print(f"SRF inside the measured band for {in_band.sum()} of {len(srf)} models; "
          f"{(~in_band).sum()} sit at or beyond an edge and their SRF is a bound, "
          f"not a measurement")

    lab, idx = decade_buckets(srf)
    print("\nSRF buckets")
    print(f"{'bucket':>14} {'count':>7} {'med C':>10} {'med ESR':>10} "
          f"{'med ESL':>10} {'med Q':>8}")
    order = sorted(set(zip(idx, lab)))
    for i, name in order:
        m = idx == i
        nL = np.nanmedian(L[m]) if np.isfinite(L[m]).any() else np.nan
        nQ = np.nanmedian(Q[m]) if np.isfinite(Q[m]).any() else np.nan
        lstr = f"{nL*1e12:>9.1f}pH" if np.isfinite(nL) else f"{'n/a':>11}"
        qstr = f"{nQ:>8.2f}" if np.isfinite(nQ) else f"{'n/a':>8}"
        note = "" if np.isfinite(nL) else "   (SRF at the band edge: no inductive decade to fit)"
        print(f"{name:>14} {m.sum():>7} {np.nanmedian(C[m])*1e6:>9.3f}uF "
              f"{np.nanmedian(ESR[m])*1e3:>9.2f}m {lstr} {qstr}{note}")

    if "--verify" in sys.argv:
        import vectorfit as vf
        k = int(sys.argv[sys.argv.index("--verify") + 1]) if len(sys.argv) > sys.argv.index("--verify") + 1 else 20
        rng = np.random.default_rng(0)
        pick = rng.choice(np.where(in_band)[0], size=min(k, in_band.sum()), replace=False)
        # Fit the ADMITTANCE, not the impedance. For a series RLC,
        #   Z(s) = sL + R + 1/(sC)
        # has a single pole at the origin and its resonance is a ZERO; the
        # complex pole pair lives in Y = 1/Z = sC/(s^2 LC + sRC + 1). Fitting Z
        # returns only real poles and finds nothing.
        print(f"\nVF cross-check on {len(pick)} parts (RLC asymptotes vs VF of Y)")
        errs = []
        for i in pick:
            # Fit a WINDOW around the SRF, not the whole 7-decade band. A
            # 2-pair model cannot describe Y from 110 Hz to 1 GHz, and a global
            # fit spends its order on the extremes and misses the notch.
            m = (freq > srf[i] / 30) & (freq < srf[i] * 30)
            if m.sum() < 20:
                continue
            r = vf.vector_fit(freq[m], Y[i][m], n_pairs=2, n_iter=12, fit_lin=False)
            rows = [q for q in r.summary() if not q["is_real"] and q["f0_hz"] > 0]
            if not rows:
                continue
            f_vf = max(rows, key=lambda q: q["peak_contrib"])["f0_hz"]
            errs.append(abs(f_vf - srf[i]) / srf[i])
        errs = np.array(errs)
        if errs.size:
            print(f"  SRF agreement: median {np.median(errs)*100:.2f}%  "
                  f"max {errs.max()*100:.2f}%  (n={errs.size})")
        else:
            print("  no complex pole pair recovered on any sampled part")

    out = {"n_models": int(Y.shape[0]),
           "freq_hz": [float(freq[0]), float(freq[-1])],
           "buckets": {name: int((idx == i).sum()) for i, name in order},
           "models": [{"i": int(i), "name": names[i], "source": str(source[i]),
                       "C_F": float(C[i]), "ESR_ohm": float(ESR[i]),
                       "ESL_H": float(L[i]), "SRF_hz": float(srf[i]),
                       "Q": float(Q[i]), "bucket": str(lab[i]),
                       "srf_in_band": bool(in_band[i])}
                      for i in range(Y.shape[0])]}
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
