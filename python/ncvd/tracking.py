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

DEFAULTS ARE THOSE OF 1.1.1 (release 1.1.2)
-------------------------------------------
Every default call of `track` runs the 1.1.1 code (`_track_v111`, unchanged)
and returns what 1.1.1 returns. That procedure has four known weaknesses, kept
on the default path so that published results reproduce:

  (1) seeding takes the lowest-frequency pole with Im(lam) > 0 of window 0,
      which can be a non-oscillatory static-offset pole (center=False) or a
      null-space eigenvalue of a rank-deficient window;
  (2) when a window has fewer pairs than tracks, the assignment's row indices
      are discarded and the candidates are written into tracks 0, 1, ... in
      order, so identity is lost (and never recovered);
  (3) rank=None keeps every direction of [disp, vel], so on rank-deficient
      windows the poles depend on the balanced quotient;
  (4) the zero-frequency (static-offset) pole is neither separated nor
      reported.

The 1.1.2 remedies are OPT-IN keywords of `track` (rank='auto', seed=...,
families=True, matching='identity', max_cost) and the function `windowed`,
which is `track` with all of them switched on (and center=False).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import ncvd, numerical_rank, RANK_RTOL
from .modal import realify, physical_modes

__all__ = ['window_edges', 'track', 'windowed', 'band_frequency', 'TrackResult']

REAL_TOL = 1e-9            # |Im lam| <= REAL_TOL*|lam| is a real pole (as physical_modes)
_SEEDS = ('lowest', 'energy', 'participation')
_MATCHING = ('legacy', 'identity')


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


def track(disp, vel, acc, fs=None, n_windows=None, length=None, step=None,
          n_track=None, M=None, center=True, rank=None, operator='ls',
          time_scale='auto', w_freq=1.0, ref_station=-1,
          seed='lowest', matching='legacy', families=False, max_cost=None,
          n_zero=1, min_cycles=1.0, w_freq_zero=0.1, rank_rtol=RANK_RTOL, rank_gap=None):
    """Sliding-window NCVD with resonance tracking.

    With every keyword at its default this is EXACTLY the 1.1.1 routine (same
    code, same seeding, same rank behavior, same ordering); see the module
    docstring for its known weaknesses and `windowed` for the corrected
    protocol.

    Parameters (1.1.1)
    ------------------
    fs : sampling rate. Used for window center times (None means 1.0, as the
        1.1.1 default) and, with families=True, for the window duration of the
        cycle criterion, where it is REQUIRED.
    n_windows / length, step : window layout (see `window_edges`).
    n_track : number of resonances to follow (default: all candidates of the
        seed window).
    M : (n, n) mass matrix or None. With M, modal amplitude and energy are
        mass-normalized; without it, the amplitude is the RMS displacement
        contribution of the pair at `ref_station`.
    center : remove the window mean (needed for multi-equilibrium systems).
        center=False is the windowed path for single-equilibrium free decays.
    rank : None (1.1.1 default: keep every direction), an int, or 'auto'
        (opt-in, 1.1.2): rank = min over windows of
        numerical_rank(s_w, rank_rtol, rank_gap), s_w the singular values of
        the window's [disp, vel] after centering. Returned as `rank`, with
        `rank_windows`, `rank_rule` and `sv`. An int is the explicit override.
    w_freq : weight of |log(w/w_prev)| against (1 - MAC) in the matching cost.

    Opt-in keywords (1.1.2; the defaults reproduce 1.1.1)
    ----------------------------------------------------
    seed : how the tracks are chosen in the seed window (the first window with
        a candidate).
        'lowest'        lowest damped frequencies (1.1.1 rule).
        'energy'        largest modal energy of the pole's own contribution,
                        E_i = mean_t v_i(t)^T M v_i(t), v_i = 2 Re(U_i Psi_v,i)
                        the velocity it contributes (M = I without a mass
                        matrix); for a harmonic contribution this is
                        0.5 omega^2 a^2, the energy of the mode.
        'participation' largest mean_t d_i^T M d_i, d_i the displacement
                        contribution.
        sequence of floats: one target damped frequency per track; the
                        nearest distinct candidates are taken, in the order
                        given.
        Selected tracks are stored in ascending seed-window frequency (except
        for targets); `seed_score` is each track's share of the seed-window
        harmonic energy (participation for seed='participation').
    families : False (1.1.1: candidates are all poles with Im(lam) > 0) or
        True: poles are split into a HARMONIC family (Im(lam) > 0 and at least
        `min_cycles` damped cycles in the window, |Im lam| T_w / 2 pi >=
        min_cycles, T_w = window samples / fs) and a ZERO-FREQUENCY family
        (real poles, and pairs with fewer cycles: the decaying static offset
        that center=False leaves in the data, or a trend). Only harmonic poles
        are tracked as resonances; the zero-frequency family is tracked
        separately in `zero` (n_zero tracks, seeded by participation, matched
        on shape with w_freq_zero) and is never seeded as a harmonic master.
    matching : 'legacy' (1.1.1) or 'identity': tracks keep their identity.
        Each window solves the rectangular assignment between tracks and
        candidates on the cost (1 - MAC(shape, shape_ref)) + w_freq
        |log(|lam|/|lam_ref|)|, where the reference is the track's LAST
        ASSIGNED window (so a gap does not erase it). A track with no
        candidate, or whose cost exceeds max_cost, is a gap: NaN in every
        per-window array and True in `gap`. The cost of every proposed
        assignment (accepted or rejected) is returned in `cost`. Required by
        any other opt-in seed or families=True.
    max_cost : None (no gate) or the largest accepted assignment cost.
    n_zero : number of zero-frequency tracks (None: all in its seed window).
    min_cycles : cycle threshold of the family split (default 1.0).
    w_freq_zero : frequency weight in the zero-family matching cost (default
        0.1; the decay rate of an offset is not a resonance frequency).
    rank_rtol, rank_gap : criterion of rank='auto' (see `core.numerical_rank`).

    Returns TrackResult with arrays of shape (n_windows, n_track):
        wn, zeta, wd, lam, mpc, amp, energy, mac_prev
    and shape (n_windows, n_track, n) real shapes, complex_shape (complex),
    res (n_windows,), t (window center times), edges.
    Opt-in calls add cost, gap (n_windows, n_track), n_harmonic (n_windows,),
    seed, seed_window, seed_score, matching, max_cost, family_rule, zero
    (TrackResult of the zero-frequency family, or None) and the rank keys.
    With identity matching, mac_prev is the MAC against the track's last
    assigned window.
    """
    if isinstance(rank, str) and rank != 'auto':
        raise ValueError(f"rank must be None, an int or 'auto', got {rank!r}")
    targets = None
    if isinstance(seed, str):
        if seed not in _SEEDS:
            raise ValueError(f'seed must be one of {_SEEDS} or a sequence of frequencies, got {seed!r}')
    else:
        targets = np.atleast_1d(np.asarray(seed, float))
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError('seed targets must be a non-empty 1-D sequence of frequencies')
    if matching not in _MATCHING:
        raise ValueError(f'matching must be one of {_MATCHING}, got {matching!r}')
    new_path = families or matching == 'identity' or targets is not None or seed != 'lowest'
    if families and fs is None:
        raise ValueError('families=True needs fs (the sampling rate) for the cycle criterion')
    if new_path and matching == 'legacy':
        raise ValueError("the 1.1.1 matching (matching='legacy') is available only with "
                         "seed='lowest' and families=False; pass matching='identity'")
    if max_cost is not None and matching != 'identity':
        raise ValueError("max_cost needs matching='identity'")

    rank_info = None
    if rank == 'auto' or new_path:
        rank, rank_info = _rank_rule(disp, vel, n_windows, length, step, center, rank,
                                     rank_rtol, rank_gap)
    if not new_path:
        tr = _track_v111(disp, vel, acc, fs=(1.0 if fs is None else fs), n_windows=n_windows,
                         length=length, step=step, n_track=n_track, M=M, center=center,
                         rank=rank, operator=operator, time_scale=time_scale, w_freq=w_freq,
                         ref_station=ref_station)
        if rank_info is not None:
            tr.update(rank_info)
        return tr
    tr = _track_112(disp, vel, acc, fs, n_windows, length, step, n_track, M, center, rank,
                    operator, time_scale, w_freq, ref_station, seed, targets, families,
                    max_cost, n_zero, min_cycles, w_freq_zero)
    tr.update(rank_info)
    return tr


def windowed(disp, vel, acc, fs, n_windows=None, length=None, step=None, n_track=None,
             M=None, center=False, rank='auto', seed='energy', families=True,
             matching='identity', max_cost=0.5, n_zero=1, **kw):
    """Windowed free-decay identification with the 1.1.2 protocol (new in 1.1.2).

    This is `track` with every 1.1.2 remedy switched on:
        center=False      single-equilibrium free decay; the static offset is kept
                          in the data and reported as the zero-frequency family
        rank='auto'       min over windows of the numerical rank (rtol 1e-12)
        seed='energy'     tracks seeded by modal energy in the first window
        families=True     harmonic and zero-frequency poles tracked separately
        matching='identity', max_cost=0.5
                          identity kept by shape and frequency continuity; gaps
                          are NaN, never relabeled
    Any of them can be overridden; other keywords go to `track`. `fs` is
    required. The 1.1.1 windowed path is `track(..., center=False)` with its
    defaults, which is unchanged.
    """
    return track(disp, vel, acc, fs=fs, n_windows=n_windows, length=length, step=step,
                 n_track=n_track, M=M, center=center, rank=rank, seed=seed,
                 families=families, matching=matching, max_cost=max_cost, n_zero=n_zero, **kw)


# --------------------------------------------------------------------- 1.1.1 routine
def _track_v111(disp, vel, acc, fs=1.0, n_windows=None, length=None, step=None,
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


# --------------------------------------------------------------------- 1.1.2 routine
def _rank_rule(disp, vel, n_windows, length, step, center, rank, rtol, gap):
    """Per-window numerical rank of [disp, vel]; rank='auto' -> min over windows."""
    disp, vel = np.asarray(disp, float), np.asarray(vel, float)
    ed = window_edges(disp.shape[0], n_windows, length, step)
    sv, rw = [], []
    for a, b in ed:
        Y1 = np.hstack([disp[a:b], vel[a:b]])
        if center:
            Y1 = Y1 - Y1.mean(0)
        s = np.linalg.svd(Y1, compute_uv=False)
        sv.append(s)
        rw.append(numerical_rank(s, rtol, gap))
    rw = np.array(rw, int)
    info = dict(rank_windows=rw, sv=np.array(sv))
    if rank == 'auto':
        r = int(rw.min())
        if r < 1:
            raise ValueError('a window has numerical rank 0')
        info.update(rank=r, rank_rule=dict(rule='min over windows', rtol=rtol, gap=gap))
        return r, info
    info.update(rank=rank, rank_rule=dict(rule=('all directions (1.1.1 default)' if rank is None
                                                else 'fixed'), rtol=rtol, gap=gap))
    return rank, info


def _candidates(r, n, Mm, Tw, families, min_cycles):
    """Split the poles of one window into harmonic and zero-frequency candidates."""
    lam = r.lam
    harm, zero = [], []
    for i, l in enumerate(lam):
        real = abs(l.imag) <= REAL_TOL*max(abs(l), 1e-300)
        up = l.imag > REAL_TOL*max(abs(l), 1e-300)
        if not (real or up):
            continue                                  # conjugate partner
        f = 1.0 if real else 2.0
        phi = r.Psi[:n, i]
        sh, mp = realify(phi)
        dcon = f*np.real(np.outer(r.U[:, i], phi))
        vcon = f*np.real(np.outer(r.U[:, i], r.Psi[n:2*n, i]))
        c = dict(i=i, lam=l, phi=phi, shape=sh, mpc=mp, u=2*np.abs(r.U[:, i]),
                 energy=float(np.mean(np.sum((vcon @ Mm)*vcon, 1))),
                 part=float(np.mean(np.sum((dcon @ Mm)*dcon, 1))))
        if families:
            if real or abs(l.imag)*Tw/(2*np.pi) < min_cycles:
                zero.append(c)
            else:
                harm.append(c)
        elif up:
            harm.append(c)
    harm.sort(key=lambda c: c['lam'].imag)
    zero.sort(key=lambda c: abs(c['lam']))
    return harm, zero


def _follow(cands, k_req, rule, targets, w_freq, max_cost):
    """Seed and follow one family with identity-preserving assignment.

    Returns (assign, cost, seed_window, seed_score): assign[w] maps track -> index
    into cands[w]; cost (nw, k) holds every proposed assignment cost.
    """
    nw = len(cands)
    sw = next((w for w in range(nw) if cands[w]), None)
    if sw is None:
        return [dict() for _ in range(nw)], np.full((nw, 0), np.nan), None, np.zeros(0)
    c0 = cands[sw]
    key = 'part' if rule in ('participation', 'zero') else 'energy'
    score = np.array([c[key] for c in c0])
    share = score/max(score.sum(), 1e-300)
    if targets is not None:
        cost0 = np.abs(np.abs(np.array([c['lam'].imag for c in c0]))[None, :] - targets[:, None])
        rows, cols = linear_sum_assignment(cost0)
        sel = list(cols[np.argsort(rows)])
    else:
        k = len(c0) if k_req is None else min(k_req, len(c0))
        if rule == 'lowest':
            sel = list(range(k))                        # cands are in ascending frequency
        elif rule == 'zero':
            sel = sorted(np.argsort(-score, kind='stable')[:k],
                         key=lambda j: abs(c0[j]['lam']))
        else:
            sel = sorted(np.argsort(-score, kind='stable')[:k],
                         key=lambda j: c0[j]['lam'].imag)
    k = len(sel)
    assign = [dict() for _ in range(nw)]
    assign[sw] = {j: int(ci) for j, ci in enumerate(sel)}
    ref = [c0[ci] for ci in sel]
    cost = np.full((nw, k), np.nan)
    for w in range(sw + 1, nw):
        cw = cands[w]
        if not cw or k == 0:
            continue
        C = np.array([[(1 - (c['shape'] @ p['shape'])**2)
                       + w_freq*abs(np.log(abs(c['lam'])/abs(p['lam']))) for c in cw]
                      for p in ref])
        rows, cols = linear_sum_assignment(C)
        for j, ci in zip(rows, cols):
            cost[w, j] = C[j, ci]
            if max_cost is not None and C[j, ci] > max_cost:
                continue                                 # rejected: gap, reference kept
            assign[w][int(j)] = int(ci)
            ref[j] = cw[ci]
    return assign, cost, sw, share[sel] if len(sel) else np.zeros(0)


def _fill(cands, assign, k, n, dXs, M, ref_station, harmonic):
    nw = len(cands)
    keys = ('wn', 'zeta', 'wd', 'mpc', 'amp', 'energy', 'mac_prev')
    out = {key: np.full((nw, k), np.nan) for key in keys}
    out['lam'] = np.full((nw, k), np.nan + 0j)
    out['shape'] = np.full((nw, k, n), np.nan)
    out['complex_shape'] = np.full((nw, k, n), np.nan + 0j)
    out['gap'] = np.ones((nw, k), bool)
    last = [None]*k
    for w in range(nw):
        for j, ci in assign[w].items():
            c = cands[w][ci]
            lam = c['lam']
            out['gap'][w, j] = False
            out['lam'][w, j] = lam
            out['wn'][w, j] = abs(lam)
            out['wd'][w, j] = abs(lam.imag)
            out['zeta'][w, j] = -lam.real/abs(lam)
            out['shape'][w, j] = c['shape']
            out['complex_shape'][w, j] = c['phi']
            out['mpc'][w, j] = c['mpc']
            if last[j] is not None:
                out['mac_prev'][w, j] = (c['shape'] @ last[j])**2
            last[j] = c['shape']
            if not harmonic:
                out['amp'][w, j] = np.sqrt(c['part'])
                continue
            if M is not None:
                vm = c['shape']/np.sqrt(c['shape'] @ M @ c['shape'])
                qa = np.sqrt(2)*(dXs[w] @ (M @ vm)).std()
                out['amp'][w, j] = qa
                out['energy'][w, j] = 0.5*(abs(lam)*qa)**2
            else:
                out['amp'][w, j] = np.sqrt(np.mean(c['u']**2))*abs(c['phi'][ref_station])
    return out


def _track_112(disp, vel, acc, fs, n_windows, length, step, n_track, M, center, rank,
               operator, time_scale, w_freq, ref_station, seed, targets, families,
               max_cost, n_zero, min_cycles, w_freq_zero):
    disp, vel, acc = (np.asarray(a, float) for a in (disp, vel, acc))
    m, n = disp.shape
    fs_t = 1.0 if fs is None else float(fs)
    ed = window_edges(m, n_windows, length, step)
    nw = len(ed)
    Mm = np.eye(n) if M is None else np.asarray(M, float)
    res = np.full(nw, np.nan)
    harm, zero, dXs, Tw = [], [], [], (ed[:, 1] - ed[:, 0])/fs_t
    for w, (a, b) in enumerate(ed):
        s = slice(a, b)
        r = ncvd(disp[s], vel[s], acc[s], center=center, rank=rank,
                 operator=operator, time_scale=time_scale)
        res[w] = r.res
        h, z = _candidates(r, n, Mm, Tw[w], families, min_cycles)
        harm.append(h)
        zero.append(z)
        dXs.append(disp[s] - disp[s].mean(0) if center else disp[s])
    rule = seed if targets is None else 'targets'
    assign, cost, sw, score = _follow(harm, n_track, rule, targets, w_freq, max_cost)
    k = cost.shape[1]
    out = _fill(harm, assign, k, n, dXs, M, ref_station, True)
    out.update(res=res, edges=ed, t=ed.mean(1)/fs_t, cost=cost,
               n_harmonic=np.array([len(h) for h in harm]), seed=rule, seed_window=sw,
               seed_score=score, matching='identity', max_cost=max_cost)
    if families:
        za, zc, zsw, zscore = _follow(zero, n_zero, 'zero', None, w_freq_zero, max_cost)
        kz = zc.shape[1]
        zo = _fill(zero, za, kz, n, dXs, M, ref_station, False)
        for key in ('mpc', 'energy'):
            zo.pop(key)
        zo.update(cost=zc, seed_window=zsw, seed_score=zscore,
                  n_candidates=np.array([len(z) for z in zero]))
        out['zero'] = TrackResult(zo)
        out['family_rule'] = dict(min_cycles=float(min_cycles), real_tol=REAL_TOL,
                                  window_duration=Tw, w_freq_zero=w_freq_zero)
    else:
        out['zero'] = None
        out['family_rule'] = None
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
