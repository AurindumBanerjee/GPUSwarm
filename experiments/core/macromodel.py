"""
macromodel -- rational models for experiments 04 and 05.

fit_rlc(d, freq)
    Series R-L-C fit of every SHUNT decap: Z_cap(s) = R + sL + 1/(sC),
    linear least squares with relative weighting (|Z| spans decades).
    Returns R, L, C arrays and the max relative fit error per part.

vector_fit(H, freq, n_pairs, n_iter)
    Fast (QR-decoupled) Vector Fitting with a COMMON pole set for many
    responses (Gustavsen & Semlyen 1999; fast VF; the none-splitting
    multiport strategy used by GVF / TC-GVF). Fits
        H_m(s) ~= sum_n r_mn / (s - p_n) + d_m + s e_m
    with real poles and complex-conjugate pairs (real system), relative
    weighting and frequency normalisation. Returns (RationalModel, err) where
    err = max over responses of max|fit-H| / max|H|.
"""

import numpy as np


# ============================================================
# decap RLC fit
# ============================================================

def fit_rlc(d, freq, min_L=1e-13):
    d = np.asarray(d)
    w = 2 * np.pi * np.asarray(freq, float)
    Z = 1.0 / d                                     # (C,F) cap impedance
    wt = 1.0 / np.abs(Z)                            # relative weighting
    R = (np.real(Z) * wt ** 2).sum(1) / (wt ** 2).sum(1)
    a1 = w[None, :] * wt                            # Im Z = w L - Cinv / w
    a2 = -(1.0 / w)[None, :] * wt
    b = np.imag(Z) * wt
    s11, s12, s22 = (a1 * a1).sum(1), (a1 * a2).sum(1), (a2 * a2).sum(1)
    t1, t2 = (a1 * b).sum(1), (a2 * b).sum(1)
    det = s11 * s22 - s12 ** 2
    L = np.maximum((t1 * s22 - t2 * s12) / det, min_L)
    C = 1.0 / np.maximum((s11 * t2 - s12 * t1) / det, 1e-30)
    R = np.maximum(R, 1e-6)
    Zf = R[:, None] + 1j * w[None, :] * L[:, None] + 1.0 / (1j * w[None, :] * C[:, None])
    err = np.max(np.abs(Zf - Z) / np.abs(Z), axis=1)
    return R, L, C, err


# ============================================================
# Vector fitting
# ============================================================

class RationalModel:
    def __init__(self, poles, residues, d, e, scale):
        self.poles = poles            # (Np,) complex, normalised, incl. conjugates
        self.residues = residues      # (M, Np) complex
        self.d = d                    # (M,)
        self.e = e                    # (M,)
        self.scale = scale            # s_norm = s / scale

    def eval(self, s):
        sn = np.asarray(s) / self.scale
        return (self.residues[None] / (sn[:, None, None] - self.poles[None, None, :])).sum(-1) \
            + self.d[None, :] + sn[:, None] * self.e[None, :]


def _initial_poles(wn, n_pairs):
    beta = np.logspace(np.log10(wn.min()), np.log10(wn.max()), n_pairs)
    return list(-beta / 100 + 1j * beta)


def _basis(sn, poles):
    cols = []
    for p in poles:
        if p.imag == 0:
            cols.append(1 / (sn - p.real))
        else:
            a = 1 / (sn - p)
            b = 1 / (sn - np.conj(p))
            cols += [a + b, 1j * a - 1j * b]
    return np.stack(cols, 1)


def _realization(poles):
    n = sum(1 if p.imag == 0 else 2 for p in poles)
    A = np.zeros((n, n))
    B = np.zeros(n)
    i = 0
    for p in poles:
        if p.imag == 0:
            A[i, i] = p.real
            B[i] = 1.0
            i += 1
        else:
            A[i:i + 2, i:i + 2] = [[p.real, p.imag], [-p.imag, p.real]]
            B[i] = 2.0
            i += 2
    return A, B


def _rr(A):
    return np.vstack([A.real, A.imag])


def vector_fit(H, freq, n_pairs=20, n_iter=8, fit_e=True, verbose=False):
    H = np.asarray(H, complex)
    F, M = H.shape
    s = 1j * 2 * np.pi * np.asarray(freq, float)
    scale = 2 * np.pi * float(np.max(freq))
    sn = s / scale
    poles = _initial_poles(np.abs(sn), n_pairs)
    W = 1.0 / np.maximum(np.abs(H), 1e-30)
    nfix = 1 + (1 if fit_e else 0)

    def fixed_cols():
        return [np.ones((F, 1))] + ([sn[:, None]] if fit_e else [])

    for it in range(n_iter):
        Phi = _basis(sn, poles)
        nb = Phi.shape[1]
        Pres = np.hstack([Phi] + fixed_cols())
        ncr = nb + nfix
        rows, rhs = [], []
        for j in range(M):
            w = W[:, j:j + 1]
            A = np.hstack([Pres * w, -Phi * H[:, j:j + 1] * w])     # sigma*H = fit
            b = H[:, j] * W[:, j]
            Q, R = np.linalg.qr(_rr(A))
            rows.append(R[ncr:, ncr:])
            rhs.append((Q.T @ np.concatenate([b.real, b.imag]))[ncr:])
        csig = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        Am, Bv = _realization(poles)
        z = np.linalg.eigvals(Am - np.outer(Bv, csig))               # zeros of sigma
        z = np.where(z.real > 0, -z.real + 1j * z.imag, z)          # enforce stability
        new = []
        for zz in z:
            if abs(zz.imag) <= 1e-9 * max(abs(zz), 1e-30):
                new.append(complex(zz.real, 0.0))
            elif zz.imag > 0:
                new.append(complex(zz))
        poles = sorted(new, key=abs)
        if verbose:
            print(f"[VF] iter {it + 1}/{n_iter}: {len(poles)} poles")

    Phi = _basis(sn, poles)
    nb = Phi.shape[1]
    Pres = np.hstack([Phi] + fixed_cols())
    coef = np.zeros((M, nb + nfix))
    for j in range(M):
        w = W[:, j:j + 1]
        br = H[:, j] * W[:, j]
        coef[j] = np.linalg.lstsq(_rr(Pres * w), np.concatenate([br.real, br.imag]), rcond=None)[0]
    allp, allr = [], []
    c = 0
    for p in poles:
        if p.imag == 0:
            allp.append(p)
            allr.append(coef[:, c] + 0j)
            c += 1
        else:
            r = coef[:, c] + 1j * coef[:, c + 1]
            allp += [p, np.conj(p)]
            allr += [r, np.conj(r)]
            c += 2
    model = RationalModel(np.array(allp), np.stack(allr, 1), coef[:, nb],
                          coef[:, nb + 1] if fit_e else np.zeros(M), scale)
    # error normalised per response by its own peak magnitude (deep nulls in
    # weakly coupled transfer terms would otherwise dominate a pointwise
    # relative error while being irrelevant in absolute terms)
    err = np.max(np.abs(model.eval(s) - H).max(0) / np.maximum(np.abs(H).max(0), 1e-30))
    return model, float(err)
