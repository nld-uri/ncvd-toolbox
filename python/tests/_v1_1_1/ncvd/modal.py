"""Modal parameters, mode selection and shape comparison."""
import numpy as np

__all__ = ['modal_parameters', 'physical_modes', 'realify', 'mpc', 'mac',
           'mac_matrix', 'mac_phase_matched', 'relative_error', 'normalize_shape',
           'poles_and_shapes', 'match_modes']


def modal_parameters(lam):
    """Natural frequency, damped frequency and damping ratio of continuous-time poles.

    Returns dict with
        wn   |lam|               natural frequency (rad per time unit)
        wd   |Im lam|            damped frequency
        zeta -Re lam / |lam|     damping ratio
        fn, fd                   the same in cycles per time unit
    """
    lam = np.asarray(lam, complex)
    wn = np.abs(lam)
    with np.errstate(invalid='ignore', divide='ignore'):
        zeta = -lam.real/np.where(wn > 0, wn, np.nan)
    return dict(wn=wn, wd=np.abs(lam.imag), zeta=zeta,
                fn=wn/(2*np.pi), fd=np.abs(lam.imag)/(2*np.pi))


def physical_modes(lam, tol=1e-9):
    """Indices of poles with positive imaginary part (one per conjugate pair),
    ordered by ascending damped frequency."""
    lam = np.asarray(lam, complex)
    idx = np.where(lam.imag > tol*np.maximum(np.abs(lam), 1e-300))[0]
    return idx[np.argsort(lam.imag[idx])]


def realify(v):
    """Phase-rotate a complex mode to maximum collinearity; return (shape, MPC).

    MPC is the modal phase collinearity: the eccentricity of the scatter of
    (Re v, Im v), 1 for a purely normal mode and falling toward 0 as the motion
    becomes genuinely complex.
    """
    v = np.asarray(v, complex)
    R, I = v.real - v.real.mean(), v.imag - v.imag.mean()
    S = np.array([[R @ R, R @ I], [R @ I, I @ I]])
    ev, evec = np.linalg.eigh(S)
    m = ((ev[1] - ev[0])/max(ev[1] + ev[0], 1e-300))**2
    th = np.arctan2(evec[1, 1], evec[0, 1])
    w = (v*np.exp(-1j*th)).real
    nrm = np.linalg.norm(w)
    if nrm < 1e-300:
        w, nrm = v.real, max(np.linalg.norm(v.real), 1e-300)
    w = w/nrm
    return w*np.sign(w[np.argmax(np.abs(w))]), float(m)


def mpc(v):
    """Modal phase collinearity of a complex shape."""
    return realify(v)[1]


def normalize_shape(v):
    """Unit 2-norm, phase rotated so the largest entry is real and positive."""
    v = np.asarray(v, complex)
    k = np.argmax(np.abs(v))
    v = v*np.exp(-1j*np.angle(v[k]))
    return v/max(np.linalg.norm(v), 1e-300)


def mac(a, b):
    """Modal assurance criterion |a^H b|^2 / (|a|^2 |b|^2) for complex vectors."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(abs(np.vdot(a, b))**2/max(np.vdot(a, a).real*np.vdot(b, b).real, 1e-300))


def mac_matrix(A, B):
    """MAC between every column of A and every column of B."""
    A, B = np.asarray(A), np.asarray(B)
    num = np.abs(A.conj().T @ B)**2
    den = np.outer(np.sum(np.abs(A)**2, 0), np.sum(np.abs(B)**2, 0))
    return num/np.maximum(den, 1e-300)


def mac_phase_matched(est, true):
    """MAC after the phase-matching of Li, Stein & Chelidze (2025), Eq. (55).

    The true complex shape is projected (real least squares) onto the span of
    [Re est, Im est]; MAC is then evaluated against the projection. Returns
    (mac, matched_estimate).
    """
    est, true = np.asarray(est, complex), np.asarray(true, complex)
    Ph = np.column_stack([est.real, est.imag])
    # complex target written as two real columns
    coef = np.linalg.lstsq(Ph, np.column_stack([true.real, true.imag]), rcond=None)[0]
    P = Ph @ coef
    matched = P[:, 0] + 1j*P[:, 1]
    return mac(matched, true), matched


def relative_error(est, true):
    """Relative l2 error after aligning the complex scale of est to true (Eq. 56)."""
    est, true = np.asarray(est, complex), np.asarray(true, complex)
    a = np.vdot(est, true)/max(np.vdot(est, est).real, 1e-300)
    return float(np.linalg.norm(a*est - true)/max(np.linalg.norm(true), 1e-300))


def poles_and_shapes(disp, vel, acc, rank=None, center=False, legacy_balanced=False, **kw):
    """Adapter for window-tracking code: (lam, shapes, resid, singular values)."""
    from .core import ncvd
    r = ncvd(disp, vel, acc, rank=rank, center=center, legacy_balanced=legacy_balanced, **kw)
    return r['lam'], r['Phi'], r['resid'], r['s']


def match_modes(lam_est, Phi_est, lam_ref, Phi_ref, w_freq=1.0):
    """One-to-one match of estimated to reference modes.

    Cost (1 - MAC) + w_freq*|log(|lam_est|/|lam_ref|)|, solved as a linear
    assignment. Returns index array j such that estimate j[i] matches reference i
    (j[i] = -1 when there are fewer estimates than references).
    """
    from scipy.optimize import linear_sum_assignment
    Mc = mac_matrix(Phi_ref, Phi_est)
    lr, le = np.abs(np.asarray(lam_ref)), np.abs(np.asarray(lam_est))
    cost = (1 - Mc) + w_freq*np.abs(np.log(np.maximum(le[None, :], 1e-300)
                                           / np.maximum(lr[:, None], 1e-300)))
    # also penalise opposite sign of Im part (conjugate confusion)
    si = np.sign(np.asarray(lam_ref).imag)[:, None] != np.sign(np.asarray(lam_est).imag)[None, :]
    cost = cost + 10.0*si
    r, c = linear_sum_assignment(cost)
    out = np.full(len(lam_ref), -1, int)          # -1: no estimate available
    out[r] = c
    return out
