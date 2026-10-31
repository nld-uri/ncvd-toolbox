"""NCVD under an arbitrary linear operator, and the snapshot (dual) form.

The canonical NCVD of a pair (Y1, Y2) is the eigendecomposition of the
least-squares operator map B = Y1^+ Y2 (core.py documents the equivalence).
Nothing in that statement refers to d/dt, so the same decomposition applies to
any linear operator D acting on the snapshot index with Y2 = D[Y1]:

    D = d/dt on [q, qdot]   canonical CVD        (poles are continuous)
    D = unit time shift     DMD / Koopman form   (poles are discrete)
    D = d^2/dx^2 on space   wavenumber form      (characteristic values are -k^2)

THE ONE REQUIREMENT (state completeness)
----------------------------------------
The operator is free; the data are not. The row space of Y1 must be invariant
under D, i.e. Y2 = S Y1 must have a solution. Displacement snapshots of a
second-order system are NOT a state: augment with velocity (CVD form) or with
delays (DMD form). `state_completeness` returns the least-squares floor, which
is the measurable signature of a missing coordinate.

WHERE THIS SITS IN THE LITERATURE
---------------------------------
With D a unit time shift this is exact DMD (Tu et al. 2014); with D = d/dt it
approximates the Koopman GENERATOR (Klus et al. 2020), the same continuous-time
operator fitted by operator inference (Peherstorfer & Willcox 2016) and returned
by the linear core of SINDy (Brunton et al. 2016). The decomposition is
therefore NOT a contrast with DMD -- see core.py for the two contrasts that are
defensible and the one term ("continuous-time DMD") not to use.

WHY THE DERIVATIVE FORM IS PREFERRED FOR SPARSE DATA
----------------------------------------------------
The shift operator inherits the Nyquist limit because it infers the rate from
successive samples; the derivative operator carries the rate in the data and is
alias-free. Measured on a 4-DOF chain: the shift form is exact to 8e-13 up to
dt = 0.81 dt_Nyquist and fails (0.21 -> 0.98) beyond it, while the derivative
form holds at ~3e-15 out to dt = 8.1 dt_Nyquist. Use `aliasing_margin` to check.

SNAPSHOT (DUAL) FORM
--------------------
nonzero eig(Y1^+ Y2) = eig(Y2 Y1^+). When there are fewer snapshots than
stations (m < N) the m x m dual is the right object; `ncvd_snapshot` implements
a reduced form valid for any m, N and rank, and never forms the N x N balanced
quotient. It needs m >= 2K snapshots for K oscillatory modes in the temporal
form, but only m >= K in the spatial form, which also needs no uniform sampling
and no time ordering at all.
"""
import warnings
import numpy as np

from .core import NCVDResult, _sort_order

__all__ = ['ncvd_snapshot', 'from_state', 'from_delays', 'from_spatial',
           'state_completeness', 'aliasing_margin']


def ncvd_snapshot(Y1, Y2, rank=None, weights=None, rcond=1e-12, sort=True, forcing=None):
    """Canonical NCVD of (Y1, Y2) in reduced form; valid for any m, N and rank.

    Equivalent to `core.ncvd_pair` with operator='ls' on full-column-rank data,
    but computed in an orthonormal basis of the retained row space of Y1, so it
    is well defined when Y1 is rank deficient and cheap when m < N. The
    balanced quotient is never formed.

    Parameters
    ----------
    Y1, Y2 : (m, N) arrays with Y2 = D[Y1] for a linear operator D.
    rank : int or None.  None keeps the numerical rank of Y1 (the safe default
        here; see core.ncvd_pair, whose rank=None keeps ALL directions).
    weights : (N,) positive or None.  SPATIAL (per-column) inner-product weights
        used only to choose the retained subspace (quadrature weights, mass
        weighting, channel trust).  When Y2 = S Y1 holds exactly the result is
        independent of the weights; they matter only where the fit is imperfect.
        NOTE the name clash: `core.ncvd_pair(weights=...)` takes per-ROW
        (per-record) weights instead. The two are different objects.
    forcing : (m, p) array or None.  Known input record F. Its contribution to
        Y2 ~ Y1 B + F G is projected out before the operator is estimated
        (Frisch-Waugh), and the input gains G are returned. Pass the input in
        whatever units are available; no mass matrix is needed. Under the shift
        operator this same structure is DMD with control (Proctor, Brunton &
        Kutz 2016) -- the 'same algebra, different operator' caution applies to
        the forced case too.
        CAVEAT that is not about the algebra: a forced stationary test helps the
        within-window identification but BREAKS the robustness-matrix
        construction, which needs the same mode at DIFFERENT amplitudes. One
        amplitude distribution means every window returns the same shape and the
        matrix collapses to rank one. Forced NCVD gives poles and shapes, not
        the amplitude-dependent programme.

    Estimator variants (TLS, bias-compensated LS) are deliberately NOT offered
    here: this routine IS the canonical least-squares definition. Use
    `core.ncvd_pair(operator=...)` for those, and note that it silently drops
    `structured_split` when `rank` is set with `operator='tls'`.

    Returns
    -------
    NCVDResult with lam, Phi (N, r), Psi = conj(Phi), E (N, r) the extractor
    (U = Y1 E, Phi = E^-H), U (m, r), R (m, N), G (p, N) input gains or None,
    res, floor, exact1, condE, rank.
    """
    Y1 = np.asarray(Y1); Y2 = np.asarray(Y2)
    if Y1.ndim != 2 or Y1.shape != Y2.shape:
        raise ValueError(f'Y1 and Y2 must be 2-D with the same shape, got {Y1.shape} and {Y2.shape}')
    F = None
    if forcing is not None:
        F = np.asarray(forcing, float).reshape(Y1.shape[0], -1)
    sw = None if weights is None else np.sqrt(np.asarray(weights, float))
    if sw is not None and (sw <= 0).any():
        raise ValueError('weights must be strictly positive')
    Yw = Y1 if sw is None else Y1*sw
    s = np.linalg.svd(Yw, compute_uv=False)
    nrank = int(np.sum(s > rcond*max(s[0], 1e-300)))
    r = nrank if rank is None else int(min(rank, nrank))
    if r < 1:
        raise ValueError('Y1 has numerical rank 0')
    Bq = np.linalg.svd(Yw, full_matrices=False)[2][:r].conj().T
    if sw is not None:
        Bq = np.linalg.qr(Bq/sw[:, None])[0]      # weights pick the subspace only
    A, C = Y1 @ Bq, Y2 @ Bq                       # UNPROJECTED reduced pair
    # Known input (Frisch-Waugh): project F out of both blocks before estimating
    # the operator, exactly as core.ncvd_pair does. Two things must not be lost:
    #   (1) U is built on the UNPROJECTED pair (U = Y1 E). Building it on the
    #       projected pair would return the forcing-removed modal coordinates
    #       instead of the actual response.
    #   (2) R carries the -F G term. Without it the input contribution reappears
    #       as apparent nonlinearity and poisons the residual readouts in
    #       `diagnostics` -- the very thing those readouts exist to measure.
    if F is None:
        Ap, Cp = A, C
    else:
        Pf = np.linalg.pinv(F)
        Ap, Cp = A - F @ (Pf @ A), C - F @ (Pf @ C)
    Sr = np.linalg.pinv(Ap) @ Cp                  # r x r reduced operator map
    lam, W = np.linalg.eig(Sr)
    if sort:
        o = _sort_order(lam)
        lam, W = lam[o], W[:, o]
    E = Bq @ W                                    # B E = E Lam
    Phi = np.linalg.pinv(E).conj().T              # Phi^H E = I
    U = A @ W                                     # (1) unprojected: U = Y1 E
    fit = (U*lam) @ Phi.conj().T
    G = None
    if F is not None:
        G = np.linalg.lstsq(F, Y2 - fit, rcond=None)[0]
        fit = fit + F @ G                         # (2)
    R = Y2 - fit
    nY = max(np.linalg.norm(Y2), 1e-300)
    Dsn = A if F is None else np.hstack([A, F])   # floor must include the input
    return NCVDResult(
        lam=lam, Phi=Phi, Psi=Phi.conj(), E=E, U=U, R=R, s=s, rank=r, V=None, T=None,
        S2=Sr, share=np.ones(r), B=None, G=G, condT=float(np.linalg.cond(W)),
        condV=float(np.linalg.cond(E)), condE=float(np.linalg.cond(E)),
        res=float(np.linalg.norm(R)/nY),
        floor=float(np.linalg.norm(Y2 - Dsn @ np.linalg.pinv(Dsn) @ Y2)/nY),
        exact1=float(np.linalg.norm(Y1 - U @ Phi.conj().T)/max(np.linalg.norm(Y1), 1e-300)),
        resid=float(np.linalg.norm(Y2 - A @ Sr @ np.linalg.pinv(Bq))/nY),
        time_scale=1.0, operator='ls', readout='eig')


def from_state(disp, vel, acc, **kw):
    """Derivative form: D = d/dt on the state [q, qdot]. Alias-free."""
    disp, vel, acc = (np.atleast_2d(np.asarray(a, float)) for a in (disp, vel, acc))
    return ncvd_snapshot(np.hstack([disp, vel]), np.hstack([vel, acc]), **kw)


def from_delays(snapshots, n_delays=2, dt=None, **kw):
    """Shift (DMD/Koopman) form on a delay-embedded state.

    n_delays = 1 is the plain shift map; use >= 2 when only displacement is
    measured, since displacement alone is not a state. If dt is given the
    result carries 'lam_c' = log(lam)/dt, the continuous poles. Subject to the
    Nyquist limit -- check with `aliasing_margin`.
    """
    S = np.atleast_2d(np.asarray(snapshots, float))
    d = int(n_delays)
    if d < 1 or len(S) <= d:
        raise ValueError('need n_delays >= 1 and more snapshots than delays')
    X = np.hstack([S[d-1-j: len(S)-1-j] for j in range(d)])
    Y = np.hstack([S[d-j:   len(S)-j]   for j in range(d)])
    out = ncvd_snapshot(X, Y, **kw)
    if dt is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            out['lam_c'] = np.log(out['lam'].astype(complex))/float(dt)
        out['dt'] = float(dt)
    return out


def from_spatial(field, d_field, **kw):
    """Spatial-operator form: D is a spatial derivative applied to each snapshot.

    `field` and `d_field` are (m, N) with rows = snapshots; the pair is
    TRANSPOSED internally so the operator acts along space. Characteristic
    values are the spatial operator's eigenvalues (-k^2 for d2/dx2).

    Needs no state augmentation, no uniform sampling and no time ordering --
    permuting the snapshots leaves the result unchanged -- and only m >= K
    snapshots for K spatial modes. Pair the wavenumbers with poles from
    `from_state` on the SAME snapshots to get a dispersion relation.
    """
    F = np.atleast_2d(np.asarray(field, float))
    D = np.atleast_2d(np.asarray(d_field, float))
    return ncvd_snapshot(F.T, D.T, **kw)


def state_completeness(Y1, Y2):
    """Relative least-squares floor ||(I - Y1 Y1^+) Y2|| / ||Y2||.

    This is the smallest residual ANY member of the NCVD family can reach
    (core.py, optimality). A floor near machine precision means the row space
    of Y1 is invariant under the operator; a large floor on clean data means a
    missing coordinate, not a bad decomposition. Displacement-only snapshots of
    a second-order system give ~0.8; the state [q, qdot] gives ~1e-14.
    """
    Y1 = np.asarray(Y1); Y2 = np.asarray(Y2)
    fit = Y1 @ np.linalg.lstsq(Y1, Y2, rcond=None)[0]
    return float(np.linalg.norm(Y2 - fit)/max(np.linalg.norm(Y2), 1e-300))


def aliasing_margin(dt, lam, warn=True):
    """dt / dt_Nyquist for the fastest pole. Values > 1 alias the shift form.

    The derivative form is unaffected; this check applies to `from_delays` and
    to any discrete-time readout.
    """
    w = np.max(np.abs(np.asarray(lam).imag))
    if w <= 0:
        return 0.0
    ratio = float(dt*w/np.pi)
    if warn and ratio > 1.0:
        warnings.warn(f'dt is {ratio:.2f} x the Nyquist limit of the fastest pole; '
                      'the shift/delay form will alias. Use the derivative form '
                      '(from_state) or refine.varpro.', RuntimeWarning, stacklevel=2)
    return ratio
