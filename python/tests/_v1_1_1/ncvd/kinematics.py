"""State records from displacement, and exact pole correction for FIR derivatives.

If the rates are produced by a linear, shift-invariant FIR derivative filter D
applied consistently,

    X1 = [q, D q],     X2 = [D q, D D q],

then X2 = X1 A~^T exactly for a free linear system, where A~ has eigenvectors
[psi; d(theta) psi] (the displacement half psi is that of A) and eigenvalues
d(theta_i) with theta_i = exp(lam_i dt) and

    d(theta) = fs * sum_j c_j theta^j          (the filter's symbol).

The NCVD therefore returns lam~ = d(theta), not lam. The true pole is
recovered by solving d(theta) = lam~ for the root theta closest to
exp(lam~ dt) and taking lam = log(theta)/dt. For the forward difference this is
Eq. (48) of Li, Stein & Chelidze (2025), lam = fs log(1 + lam~/fs); the same
inversion applies to every FIR stencil, including Savitzky-Golay.

The displacement half of each shape is unaffected.
"""
import numpy as np

__all__ = ['DerivFilter', 'deriv_filter', 'state_from_displacement', 'correct_poles',
           'apply_filter']


class DerivFilter:
    """FIR first-derivative stencil: (D q)_k = fs * sum_j c[j] q[k + offsets[j]]."""

    def __init__(self, coeffs, offsets, name='custom'):
        self.c = np.asarray(coeffs, float)
        self.off = np.asarray(offsets, int)
        self.name = name
        if self.c.shape != self.off.shape:
            raise ValueError('coeffs and offsets must have equal length')

    def symbol(self, theta, fs):
        """d(theta) = fs * sum c_j theta^j."""
        theta = np.asarray(theta, complex)
        return fs*np.sum(self.c[:, None]*theta.ravel()[None, :]**self.off[:, None], 0
                         ).reshape(theta.shape)

    def band(self):
        """Largest omega*dt below which the symbol is one-to-one on the
        imaginary axis: the first maximum of Im d(exp(i w))/fs on (0, pi].
        Poles with |Im lam| dt above this value cannot be corrected uniquely.
        Two-point stencils (forward, backward) are one-to-one up to Nyquist."""
        if self.off.max() - self.off.min() == 1:
            return float(np.pi)
        w = np.linspace(1e-4, np.pi, 20001)
        g = np.imag(self.symbol(np.exp(1j*w), 1.0))
        dg = np.diff(g)
        k = np.where(dg <= 0)[0]
        return float(w[k[0]]) if k.size else float(np.pi)

    def __repr__(self):
        return f"DerivFilter({self.name}, offsets={self.off.tolist()})"


def deriv_filter(method='central', window=7, polyorder=3):
    """Named derivative stencils.

    method : 'forward'  (q[k+1]-q[k])/dt
             'backward' (q[k]-q[k-1])/dt
             'central'  (q[k+1]-q[k-1])/(2 dt)
             'central4' fourth-order five-point central difference
             'savgol'   Savitzky-Golay derivative (window, polyorder)
    """
    m = method.lower()
    if m == 'forward':
        return DerivFilter([-1, 1], [0, 1], m)
    if m == 'backward':
        return DerivFilter([-1, 1], [-1, 0], m)
    if m == 'central':
        return DerivFilter([-0.5, 0.5], [-1, 1], m)
    if m == 'central4':
        return DerivFilter([1/12, -8/12, 8/12, -1/12], [-2, -1, 1, 2], m)
    if m == 'savgol':
        from scipy.signal import savgol_coeffs
        if window % 2 == 0:
            raise ValueError('savgol window must be odd')
        h = (window - 1)//2
        c = savgol_coeffs(window, polyorder, deriv=1, use='dot')
        return DerivFilter(c, np.arange(-h, h + 1), f'savgol({window},{polyorder})')
    raise ValueError(f'unknown method {method!r}')


def apply_filter(q, filt, fs):
    """Apply a DerivFilter along axis 0. Returns (Dq, k0) where Dq[i] belongs to
    sample k0 + i (only samples with a full stencil are returned)."""
    q = np.asarray(q)
    lo, hi = -min(filt.off.min(), 0), max(filt.off.max(), 0)
    m = q.shape[0]
    k = np.arange(lo, m - hi)
    out = np.zeros((len(k),) + q.shape[1:], dtype=np.result_type(q, float))
    for c, o in zip(filt.c, filt.off):
        out += c*q[k + o]
    return fs*out, lo


def state_from_displacement(q, fs, method='central', **kw):
    """Build a consistent (disp, vel, acc) record from displacement only.

    vel = D q and acc = D(D q) with the same stencil D, trimmed to the samples
    where both exist and aligned with disp. Feed the result to `ncvd` and pass
    its poles through `correct_poles(lam, filt, fs)`.

    Returns (disp, vel, acc, filt, k0) with k0 the index of the first retained
    sample in the original record.
    """
    filt = method if isinstance(method, DerivFilter) else deriv_filter(method, **kw)
    q = np.asarray(q, float)
    v, k1 = apply_filter(q, filt, fs)
    a, k2 = apply_filter(v, filt, fs)
    k0 = k1 + k2                                  # first sample with acc
    n_a = a.shape[0]
    d = q[k0:k0 + n_a]
    v = v[k2:k2 + n_a]
    return d, v, a, filt, k0


def correct_poles(lam_est, filt, fs, branch='principal'):
    """Map poles identified from FIR-derived rates back to continuous-time poles.

    Solves d(theta) = lam_est for theta (polynomial roots), picks the root with
    the smallest |log theta| (the lowest-frequency, least-damped preimage), and
    returns log(theta)*fs.

    The inversion is unique only where the stencil's symbol is one-to-one:
    forward/backward differences up to Nyquist (omega dt < pi); central-type
    stencils (central, central4, Savitzky-Golay) only below their symbol's
    first turning point (omega dt < pi/2 for the central difference), because
    omega dt and pi - omega dt give the same derivative estimate. Outside that
    band the lowest-frequency preimage is returned. Re-sort the corrected
    poles: distortion can change their frequency order.
    """
    if isinstance(filt, str):
        filt = deriv_filter(filt)
    lam_est = np.atleast_1d(np.asarray(lam_est, complex))
    out = np.empty_like(lam_est)
    jmin = min(filt.off.min(), 0)
    deg = filt.off.max() - jmin
    for i, le in enumerate(lam_est):
        # sum c_j theta^(j - jmin) - (le/fs) theta^(-jmin) = 0
        p = np.zeros(deg + 1, complex)              # ascending powers
        for c, o in zip(filt.c, filt.off):
            p[o - jmin] += c
        p[-jmin] -= le/fs
        roots = np.roots(p[::-1])
        roots = roots[np.abs(roots) > 1e-300]
        th = roots[np.argmin(np.abs(np.log(roots.astype(complex))))]
        out[i] = np.log(th)*fs
    return out if out.size > 1 else out[0]
