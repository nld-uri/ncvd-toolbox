"""Iterative refinement of characteristic values under noise.

WHAT DOES AND DOES NOT HELP (measured; see the manual)
------------------------------------------------------
The canonical NCVD is the GLOBAL minimiser of the second-identity residual,
but that is a statement about one record, not about the estimator. Under
measurement noise its error turns out to be VARIANCE, not bias: the signed
damping bias on a 4-DOF chain at 45 dB is -0.29 +/- 0.33 %, i.e. not
significantly different from zero, against a median absolute error of 2.50 %.

Consequently the standard errors-in-variables repairs buy little here. Median
relative pole error over 12 realisations:

    SNR            60 dB          45 dB          30 dB
    estimator      f      zeta    f      zeta    f      zeta
    canonical     0.0016  0.45   0.0097  2.50   0.19   13.9   (%)
    TLS           0.0016  0.45   0.0102  2.50   0.100  13.8
    forward-back  0.0016  0.45   0.0086  2.51   0.104  21.4
    varpro        0.0000  0.01   0.0002  0.06   0.0011  0.36

Variable projection -- Gauss-Newton on the poles with the linear amplitudes
eliminated -- reduces the damping error by about 40x at 45 dB and 38x at 30 dB,
and the frequency error by 175x at 30 dB. It never differentiates and never
assumes a uniform grid, so it also handles unevenly sampled records. Warm-start
it from the canonical NCVD, which supplies both the starting poles and the
mode count.

"""
import numpy as np
from scipy.optimize import least_squares

__all__ = ['varpro', 'refine_poles']


def _design(t, sig, omg, order=0):
    cols = []
    for s, w in zip(sig, omg):
        lam = s + 1j*w
        e = np.exp(lam*t)*(lam**order)
        cols += [e.real, e.imag]
    return np.stack(cols, 1)


def varpro(data, t, lam0, channels=None, max_nfev=400, xtol=1e-11, ftol=1e-11):
    """Variable-projection refinement of poles for a sum of damped exponentials.

    Fits  data(t) ~ sum_i exp(lam_i t) * b_i  with the linear amplitudes b_i
    eliminated at every step, so only the poles are optimised.

    Parameters
    ----------
    data : (m, n) array -- one channel block (e.g. displacement at n stations).
    t : (m,) sample times. NEED NOT BE UNIFORM.
    lam0 : (k,) starting poles with positive imaginary part (from the canonical
        NCVD: take rec.lam[rec.lam.imag > 0]).
    channels : list of (array, order) or None
        Extra blocks that share the SAME poles, with their derivative order,
        e.g. [(vel, 1), (acc, 2)]. Including them enforces the derivative
        structure; displacement alone is usually enough and is the most robust
        choice when the derivatives were obtained numerically.

    Returns
    -------
    dict with lam (k,), amplitudes B (2k, n), residual, nfev, success.
    """
    data = np.atleast_2d(np.asarray(data, float))
    t = np.asarray(t, float).ravel()
    if data.shape[0] != t.size:
        raise ValueError('data must have one row per sample time')
    lam0 = np.atleast_1d(np.asarray(lam0, complex))
    if lam0.size == 0:
        raise ValueError('lam0 is empty')
    k = lam0.size
    p0 = np.concatenate([lam0.real, np.abs(lam0.imag)])

    def resid(p):
        sig, omg = p[:k], p[k:]
        E = _design(t, sig, omg, 0)
        r = (data - E @ np.linalg.lstsq(E, data, rcond=None)[0]).ravel()
        if channels:
            for Dx, od in channels:
                Dx = np.atleast_2d(np.asarray(Dx, float))
                Ex = _design(t, sig, omg, od)
                r = np.concatenate([r, (Dx - Ex @ np.linalg.lstsq(Ex, Dx, rcond=None)[0]).ravel()])
        return r

    sol = least_squares(resid, p0, method='trf', xtol=xtol, ftol=ftol, max_nfev=max_nfev)
    sig, omg = sol.x[:k], np.abs(sol.x[k:])
    E = _design(t, sig, omg, 0)
    B = np.linalg.lstsq(E, data, rcond=None)[0]
    return dict(lam=sig + 1j*omg, B=B, nfev=int(sol.nfev), success=bool(sol.success),
                residual=float(np.linalg.norm(data - E @ B)/max(np.linalg.norm(data), 1e-300)))


def refine_poles(rec, data, t, channels=None, **kw):
    """Refine the poles of an NCVDResult in place-safe fashion.

    Takes the oscillatory poles of `rec` as the starting point and returns a new
    dict from `varpro`. Does not modify `rec`.
    """
    lam = np.asarray(rec['lam'])
    lam = lam[lam.imag > 0]
    if lam.size == 0:
        raise ValueError('no oscillatory poles in rec to refine')
    return varpro(data, t, lam, channels=channels, **kw)
