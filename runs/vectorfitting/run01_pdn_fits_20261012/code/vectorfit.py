"""
Vector fitting (Gustavsen & Semlyen 1999, relaxed variant Gustavsen 2006).

Fits sampled frequency-domain data with a rational function

    H(s) ~= sum_k  r_k / (s - p_k)  +  d  +  s * e

by iteratively relocating poles. Each iteration solves one linear least-squares
problem for the residues of H*sigma and of sigma simultaneously, where sigma is
an auxiliary rational function with the same poles; the zeros of sigma are the
improved poles. Converges in a handful of iterations from a spread starting set.

Why this is here: the poles ARE the resonances. A pole at s = -a + j*w gives a
resonance at w/2pi with quality factor Q = sqrt(a^2 + w^2) / (2a). Reading peaks
off a sampled |Z| curve gives you frequency and height; vector fitting gives you
frequency, damping and the residue, which is what decides how much capacitance
it takes to damp a peak and how wide a band that capacitor has to cover.

Self-contained: numpy only.
"""
import numpy as np

__all__ = ["vector_fit", "VFResult", "starting_poles", "rational_eval",
           "pole_summary"]


class VFResult:
    """poles, residues, d, e, plus fit quality and per-pole resonance data."""

    def __init__(self, poles, residues, d, e, freq, data, rms, iters):
        self.poles = poles              # (n,) complex, in rad/s
        self.residues = residues        # (n,) complex
        self.d = d                      # constant term
        self.e = e                      # s-proportional term
        self.freq = freq                # (F,) Hz, the data it was fitted to
        self.data = data                # (F,) complex, the data
        self.rms = rms                  # relative RMS error of the final fit
        self.iters = iters

    @property
    def order(self):
        return len(self.poles)

    def __call__(self, freq):
        return rational_eval(freq, self.poles, self.residues, self.d, self.e)

    def summary(self):
        return pole_summary(self.poles, self.residues)

    def __repr__(self):
        return (f"<VFResult order={self.order} rms={self.rms:.3e} "
                f"iters={self.iters}>")


def starting_poles(freq, n_pairs, real_frac=0.01):
    """Complex-conjugate pairs spread logarithmically over the band.

    Gustavsen's recommendation: imaginary parts spanning the frequency range,
    real parts small and negative (weakly damped), here imag/100. Starting from
    weakly damped poles lets the relocation move them outward if the data is
    lossier, which converges better than starting over-damped.
    """
    w = 2 * np.pi * np.asarray(freq, float)
    beta = np.logspace(np.log10(w[0]), np.log10(w[-1]), n_pairs)
    alpha = real_frac * beta
    return np.concatenate([-alpha + 1j * beta, -alpha - 1j * beta])


def _scaled_lstsq(A, b):
    """Least squares with columns normalised to unit 2-norm, then unscaled.

    The VF design matrix mixes columns differing by ~15 orders of magnitude
    (1/(s-p) against s). Solving it raw makes lstsq's rcond cutoff discard real
    information; normalising first puts every column on the same footing.
    """
    nrm = np.linalg.norm(A, axis=0)
    nrm[nrm == 0] = 1.0
    x, *_ = np.linalg.lstsq(A / nrm, b, rcond=None)
    return x / nrm


def _basis(s, poles):
    """Partial-fraction basis, in real arithmetic.

    A complex pair (p, p*) contributes two REAL basis columns,
    1/(s-p) + 1/(s-p*) and j/(s-p) - j/(s-p*), so the residue pair stays
    conjugate by construction and the unknowns stay real. Real poles
    contribute one column each.
    """
    F, n = len(s), len(poles)
    A = np.zeros((F, n), complex)
    k = 0
    while k < n:
        p = poles[k]
        if abs(p.imag) < 1e-14:                      # real pole
            A[:, k] = 1.0 / (s - p)
            k += 1
        else:                                        # conjugate pair
            A[:, k] = 1.0 / (s - p) + 1.0 / (s - np.conj(p))
            A[:, k + 1] = 1j / (s - p) - 1j / (s - np.conj(p))
            k += 2
    return A


def _pair_mask(poles):
    """True at the first index of each conjugate pair."""
    m = np.zeros(len(poles), bool)
    k = 0
    while k < len(poles):
        if abs(poles[k].imag) < 1e-14:
            k += 1
        else:
            m[k] = True
            k += 2
    return m


def _unpack(x, poles):
    """Real LS unknowns -> complex residues."""
    r = np.zeros(len(poles), complex)
    k = 0
    while k < len(poles):
        if abs(poles[k].imag) < 1e-14:
            r[k] = x[k]
            k += 1
        else:
            r[k] = x[k] + 1j * x[k + 1]
            r[k + 1] = np.conj(r[k])
            k += 2
    return r


def _relocate(s, H, poles, weight, fit_const, fit_lin):
    """One pole-relocation iteration. Returns the new pole set.

    Solves  sum_k c_k/(s-p_k) + d + s*e  -  H * (sum_k ct_k/(s-p_k) + 1) ~= 0
    (relaxed form: the sigma constant is fixed at 1 and the equation is
    weighted), then takes the zeros of sigma as the new poles. Those zeros are
    the eigenvalues of A - b*ct^T with A the pole matrix in real block form.
    """
    n = len(poles)
    Ab = _basis(s, poles)
    cols = [Ab]
    if fit_const:
        cols.append(np.ones((len(s), 1)))
    if fit_lin:
        cols.append(s[:, None])
    left = np.hstack(cols)
    right = -(H[:, None] * Ab)
    M = np.hstack([left, right])
    rhs = H

    W = weight[:, None]
    Mr = np.vstack([np.real(M * W), np.imag(M * W)])
    br = np.concatenate([np.real(rhs * weight), np.imag(rhs * weight)])
    # Column scaling is not optional here. The s*e column has magnitude ~1e9
    # while the partial-fraction columns are ~1e-6, so without it the system is
    # conditioned at ~1e15, lstsq truncates the small columns away, and the
    # returned sigma residues are garbage -- the poles then never move.
    x = _scaled_lstsq(Mr, br)
    ct = _unpack(x[left.shape[1]:], poles)

    # zeros of sigma = eig(A - b ct^T), A real-block diagonal
    A = np.zeros((n, n))
    b = np.zeros(n)
    k = 0
    while k < n:
        p = poles[k]
        if abs(p.imag) < 1e-14:
            A[k, k] = p.real
            b[k] = 1.0
            k += 1
        else:
            A[k, k] = A[k + 1, k + 1] = p.real
            A[k, k + 1] = p.imag
            A[k + 1, k] = -p.imag
            b[k], b[k + 1] = 2.0, 0.0
            k += 2
    ctr = np.zeros(n)
    k = 0
    while k < n:
        if abs(poles[k].imag) < 1e-14:
            ctr[k] = ct[k].real
            k += 1
        else:
            ctr[k], ctr[k + 1] = ct[k].real, ct[k].imag
            k += 2

    new = np.linalg.eigvals(A - np.outer(b, ctr))
    new = np.where(new.real > 0, -new.real + 1j * new.imag, new)   # flip unstable
    return _canonical(new)


def _canonical(poles):
    """Sort so each conjugate pair is adjacent with +imag first."""
    real = np.sort(poles[np.abs(poles.imag) < 1e-14].real)
    cplx = poles[np.abs(poles.imag) >= 1e-14]
    cplx = cplx[cplx.imag > 0]
    cplx = cplx[np.argsort(cplx.imag)]
    out = [np.asarray(real, complex)]
    for p in cplx:
        out.append(np.array([p, np.conj(p)]))
    return np.concatenate(out) if out else np.array([], complex)


def _residues(s, H, poles, weight, fit_const, fit_lin):
    Ab = _basis(s, poles)
    cols = [Ab]
    if fit_const:
        cols.append(np.ones((len(s), 1)))
    if fit_lin:
        cols.append(s[:, None])
    M = np.hstack(cols)
    W = weight[:, None]
    Mr = np.vstack([np.real(M * W), np.imag(M * W)])
    br = np.concatenate([np.real(H * weight), np.imag(H * weight)])
    x = _scaled_lstsq(Mr, br)
    r = _unpack(x[:len(poles)], poles)
    i = len(poles)
    d = x[i] if fit_const else 0.0
    e = x[i + (1 if fit_const else 0)] if fit_lin else 0.0
    return r, float(d), float(e)


def rational_eval(freq, poles, residues, d=0.0, e=0.0):
    s = 2j * np.pi * np.asarray(freq, float)
    out = np.full(len(s), complex(d))
    out = out + e * s
    for p, r in zip(poles, residues):
        out = out + r / (s - p)
    return out


def vector_fit(freq, data, n_pairs=12, n_iter=8, fit_const=True, fit_lin=True,
               weight="inverse", poles0=None, tol=1e-6, verbose=False):
    """Fit `data` sampled at `freq` (Hz) with a rational function.

    weight : 'inverse'  -> 1/|H|, equalises relative error across decades. This
                           is what you want for impedance data spanning orders
                           of magnitude: unweighted fitting would spend all its
                           accuracy on the largest peak and ignore the floor.
             'unity'    -> uniform
             'sqrt'     -> 1/sqrt(|H|), a compromise
             array      -> your own, length F

    Returns a VFResult. Stops early if the relative RMS error stops improving
    by more than `tol`.
    """
    freq = np.asarray(freq, float)
    H = np.asarray(data, complex)
    s = 2j * np.pi * freq

    if isinstance(weight, str):
        mag = np.abs(H)
        mag[mag == 0] = np.finfo(float).tiny
        w = {"inverse": 1.0 / mag,
             "sqrt": 1.0 / np.sqrt(mag),
             "unity": np.ones_like(mag)}[weight]
    else:
        w = np.asarray(weight, float)
    w = w / np.mean(w)

    poles = _canonical(starting_poles(freq, n_pairs) if poles0 is None
                       else np.asarray(poles0, complex))

    prev, used = np.inf, 0
    for it in range(n_iter):
        poles = _relocate(s, H, poles, w, fit_const, fit_lin)
        r, d, e = _residues(s, H, poles, w, fit_const, fit_lin)
        fit = rational_eval(freq, poles, r, d, e)
        rms = float(np.sqrt(np.mean(np.abs((fit - H) / np.where(
            np.abs(H) > 0, np.abs(H), 1.0)) ** 2)))
        used = it + 1
        if verbose:
            print(f"  iter {used:2d}  rel-rms {rms:.4e}")
        if prev - rms < tol * max(rms, 1e-300):
            break
        prev = rms

    r, d, e = _residues(s, H, poles, w, fit_const, fit_lin)
    fit = rational_eval(freq, poles, r, d, e)
    rms = float(np.sqrt(np.mean(np.abs((fit - H) / np.where(
        np.abs(H) > 0, np.abs(H), 1.0)) ** 2)))
    return VFResult(poles, r, d, e, freq, H, rms, used)


def pole_summary(poles, residues=None):
    """One row per conjugate pair / real pole, as resonance data.

    For p = -a + j*w:  f0 = w/2pi,  Q = |p| / (2a),  and the half-power
    bandwidth is f0/Q. A high-Q pole is a sharp anti-resonance that needs a
    narrow-band, low-ESR capacitor; a low-Q pole is a broad hump that a bulk
    capacitor can cover.
    """
    rows = []
    for k, p in enumerate(poles):
        if p.imag < 0:                       # keep one of each pair
            continue
        a, wq = -p.real, abs(p.imag)
        f0 = wq / (2 * np.pi)
        Q = (abs(p) / (2 * a)) if a > 0 else np.inf
        row = {"index": k, "f0_hz": f0, "sigma": a, "Q": Q,
               "bandwidth_hz": (f0 / Q) if np.isfinite(Q) and Q > 0 else np.inf,
               "is_real": abs(p.imag) < 1e-14}
        if residues is not None:
            r = residues[k]
            row["residue_abs"] = float(abs(r))
            # |contribution| at the pole frequency, |r|/a for a weakly damped pole
            row["peak_contrib"] = float(abs(r) / a) if a > 0 else np.inf
        rows.append(row)
    rows.sort(key=lambda d: d["f0_hz"])
    return rows
