"""Amplitude-dependent identification by sliding-window NCVD.

A nonlinear free decay visits a sequence of energy levels. Decomposing short
windows returns, at each level, the best linear description of the local
dynamics: characteristic values (frequency, damping) and shapes. Following
each resonance through the windows traces its frequency-energy dependence,
the data-driven counterpart of a nonlinear normal mode backbone curve.

The residual ratio res = ||R||/||X2|| of each window measures how much of the
rate data the diagonal linear form cannot represent. For a noise-free record it
is zero for a linear system and grows with the strength of the nonlinearity
engaged in that window.
"""
import numpy as np
from .core import ncvd
from .modal import realify, physical_modes

__all__ = ['window_edges', 'track', 'band_frequency', 'TrackResult']


class TrackResult(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def window_edges(m, n_windows=None, length=None, step=None):
    """Start/stop indices. Either n_windows equal non-overlapping windows, or
    windows of `length` samples advanced by `step` (overlap allowed)."""
    if n_windows is not None:
        e = np.linspace(0, m, n_windows + 1).astype(int)
        return np.column_stack([e[:-1], e[1:]])
    if length is None:
        raise ValueError('give n_windows or length')
    step = length if step is None else step
    starts = np.arange(0, m - length + 1, step)
    return np.column_stack([starts, starts + length])


def track(disp, vel, acc, fs=1.0, n_windows=None, length=None, step=None,
          n_track=None, M=None, center=True, rank=None, operator='ls',
          time_scale='auto', w_freq=1.0, ref_station=-1):
    """Sliding-window NCVD with resonance tracking.

    Parameters
    ----------
    fs : sampling rate, used only for window centre times.
    n_windows / length, step : window layout (see `window_edges`).
    n_track : number of resonances to follow (default: all physical pairs of
        the first window).
    M : (n, n) mass matrix or None. With M, modal amplitude and energy are
        mass-normalised; without it, the amplitude is the RMS displacement
        contribution of the pair at `ref_station`.
    center : remove the window mean (needed for multi-equilibrium systems).
    w_freq : weight of |log(w/w_prev)| against (1 - MAC) in the matching cost.

    Returns TrackResult with arrays of shape (n_windows, n_track):
        wn, zeta, wd, lam, mpc, amp, energy, mac_prev
    and shape (n_windows, n_track, n) real shapes, complex_shape (complex),
    res (n_windows,), t (window centre times), edges.
    """
    from scipy.optimize import linear_sum_assignment
    disp, vel, acc = (np.asarray(a, float) for a in (disp, vel, acc))
    m, n = disp.shape
    ed = window_edges(m, n_windows, length, step)
    nw = len(ed)
    out = None
    prev = None
    res = np.full(nw, np.nan)
    for w, (a, b) in enumerate(ed):
        s = slice(a, b)
        r = ncvd(disp[s], vel[s], acc[s], center=center, rank=rank,
                 operator=operator, time_scale=time_scale)
        res[w] = r.res
        idx = physical_modes(r.lam)
        if out is None:
            k = len(idx) if n_track is None else min(n_track, len(idx))
            out = {key: np.full((nw, k), np.nan) for key in
                   ('wn', 'zeta', 'wd', 'mpc', 'amp', 'energy', 'mac_prev')}
            out['lam'] = np.full((nw, k), np.nan + 0j)
            out['shape'] = np.full((nw, k, n), np.nan)
            out['complex_shape'] = np.full((nw, k, n), np.nan + 0j)
        k = out['wn'].shape[1]
        cand = []
        for i in idx:
            phi = r.Psi[:n, i]               # eigenvector shape of lam[i]
            sh, mp = realify(phi)
            # displacement contribution of the conjugate pair: 2 Re(u psi)
            amp_t = 2*np.abs(r.U[:, i])
            cand.append(dict(lam=r.lam[i], phi=phi, shape=sh, mpc=mp, u=amp_t))
        if prev is None:
            sel = list(range(min(k, len(cand))))
        else:
            cost = np.array([[(1 - (c['shape'] @ p['shape'])**2)
                              + w_freq*abs(np.log(abs(c['lam'])/abs(p['lam'])))
                              for c in cand] for p in prev])
            _, sel = linear_sum_assignment(cost)
        dX = disp[s] - disp[s].mean(0) if center else disp[s]
        chosen = []
        for j, ci in enumerate(sel):
            c = cand[ci]
            lam = c['lam']
            out['lam'][w, j] = lam
            out['wn'][w, j] = abs(lam)
            out['wd'][w, j] = abs(lam.imag)
            out['zeta'][w, j] = -lam.real/abs(lam)
            out['shape'][w, j] = c['shape']
            out['complex_shape'][w, j] = c['phi']
            out['mpc'][w, j] = c['mpc']
            if prev is not None:
                out['mac_prev'][w, j] = (c['shape'] @ prev[j]['shape'])**2
            if M is not None:
                vm = c['shape']/np.sqrt(c['shape'] @ M @ c['shape'])
                qa = np.sqrt(2)*(dX @ (M @ vm)).std()     # modal amplitude
                out['amp'][w, j] = qa
                out['energy'][w, j] = 0.5*(abs(lam)*qa)**2
            else:
                phi = c['phi']
                out['amp'][w, j] = np.sqrt(np.mean(c['u']**2))*abs(phi[ref_station])
            chosen.append(c)
        prev = chosen
    out['res'] = res
    out['edges'] = ed
    out['t'] = ed.mean(1)/fs
    return TrackResult(out)


def band_frequency(sig, dt, lo, hi):
    """Dominant angular frequency of sig inside (lo, hi): Hann window and
    parabolic peak refinement. Use it to validate identified poles window by
    window."""
    s = np.asarray(sig, float) - np.mean(sig)
    S = np.abs(np.fft.rfft(s*np.hanning(len(s))))
    f = np.fft.rfftfreq(len(s), dt)*2*np.pi
    mk = (f > lo) & (f < hi)
    if not mk.any():
        return np.nan
    i = np.where(mk)[0][np.argmax(S[mk])]
    if i == 0 or i == len(S) - 1:
        return f[i]
    d = 0.5*(S[i-1] - S[i+1])/(S[i-1] - 2*S[i] + S[i+1])
    return (i + d)*(f[1] - f[0])
