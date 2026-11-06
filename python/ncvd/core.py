"""Canonical Nonlinear Characteristic Value Decomposition (NCVD).

THE DECOMPOSITION (canonical definition, 27 Oct 2026)
-----------------------------------------------------
For a state record X1 = [disp, vel] and its rate X2 = [vel, acc],

    (1)  X1 = U Phi^H ,     X2 = U Lam Phi^H + R ,     Lam = diag(lam) ,

with U the temporal basis (characteristic coordinates), Phi the characteristic
mode shapes, lam the characteristic values and R everything the diagonal form
cannot represent (noise, nonlinearity, forcing that was not supplied).

THE CANONICAL DECOMPOSITION IS (1) TOGETHER WITH (2):

    (2)  M = X1^+ X2 ,  M E = E Lam ,  Phi = E^-H ,  U = X1 E ,
         R = X2 - U Lam Phi^H

i.e. the eigendecomposition of the least-squares operator map, with NO balanced
quotient. E is the extractor (Phi^H E = I); Psi = conj(Phi) are the physical
eigenvector shapes. Proposition: (2) is the unique minimiser of ||R|| over all
(Phi invertible, Lam diagonal), because M = Phi^-H Lam Phi^H ranges over every
diagonalisable matrix, so min ||R|| is the unconstrained least-squares floor.

WHICH ROUTINE COMPUTES WHICH
----------------------------
`operators.ncvd_snapshot` implements (2) DIRECTLY and is the reference
implementation of the definition; it is also the only one valid when X1 is rank
deficient or has fewer rows than columns.

`ncvd_pair` / `ncvd` below still compute the SAME result by the historical route
through the balanced quotient Q, kept for backward compatibility and to
reproduce the published diagonal readout. The two agree to 1.1e-14 in the poles
and to MAC 1.000000000000 in the shapes. Q is scaffolding: see EQUIVALENCES.

RELATION TO DMD, KOOPMAN GENERATORS AND OPERATOR INFERENCE
----------------------------------------------------------
Form (2) places the method exactly. With D a unit time shift it IS exact DMD
(Tu et al., J. Comput. Dyn. 1:391, 2014, doi 10.3934/jcd.2014.1.391), defined
there as the eigendecomposition of the least-squares one-step map. With
D = d/dt it approximates the Koopman GENERATOR, the object estimated from state
and derivative data by Klus et al. (Physica D 406:132416, 2020,
doi 10.1016/j.physd.2020.132416); the same continuous-time operator is fitted by
operator inference (Peherstorfer & Willcox, CMAME 306:196, 2016,
doi 10.1016/j.cma.2016.03.025) and returned by the linear core of SINDy
(Brunton et al., PNAS 113:3932, 2016, doi 10.1073/pnas.1517384113).
"continuous-time DMD" is NOT an established term; do not use it.

DO NOT claim the decomposition itself as a contrast with DMD -- it is DMD's
algebra under a different operator. Two contrasts ARE defensible, and both
follow from the operator rather than the algebra:
  (a) the derivative form is ALIAS-FREE (3e-15 at dt = 8.1x Nyquist) where the
      shift form fails past Nyquist (0.21 -> 0.98), and tolerates an 82x spread
      in dt at 2.8e-14;
  (b) for D = d/dt on [q, qdot], R is the residual of Newton's second law and
      carries FORCE units, which licenses the localisation/order/class/memory
      readouts in `diagnostics`; for the shift operator R is a one-step
      prediction error with no physical reading.
The price is that rates are needed: where they come from numerical
differentiation, differentiation error competes with the aliasing error (a)
avoids, and the advantage must be argued case by case.

ALGORITHM (one path; every option below enters at a single step)
---------------------------------------------------------------
1.  Optional preprocessing: centring, row weights, rank-r whitening of X1,
    projection of a supplied forcing record out of both blocks, and a
    rescaling of the time unit.
2.  Balanced quotient  Q = (K21 K11^-1 + K22 K12^-1 + K11 K21^-1 + K12 K22^-1)/4,
    with Kij = Yi^H Yj, and its eigenvectors V.  For exact linear data
    Q = (A + A^-1)/2, so V are eigenvectors of A.
3.  Xi = Y1 V^-H  (so that Sigma_1 = I) and Sigma_2 = V^H B V^-H, where B is the
    regression operator Y2 ~ Y1 B ('ls': B = Y1^+ Y2; 'tls': total least squares).
4.  Sigma_2 = T Lam T^-1.  Rotate:  U = Xi T,  Phi = V T^-H.

Step 4 makes both identities exact and pairs lam[i] with Phi[:, i] by
construction, and reproduces (2) exactly via E = V^-H T.  The earlier GCVD readout used V itself as the shapes and the
diagonal of Sigma_2 as the poles; that is exact only when V diagonalises
Sigma_2, which fails whenever the balanced map lam -> (lam + 1/lam)/2 (a
Joukowski map) folds two poles onto nearly the same point.  This happens for
lightly damped poles with |lam| close to 1 in the chosen time unit, or for
pairs with lam_j lam_k close to 1.  `time_scale='auto'` moves every pole
outside the unit circle, where the Joukowski map is one-to-one, so that V is
already nearly diagonalising and T is well conditioned.

EQUIVALENCES (proved in the manual, verified in tests)
------------------------------------------------------
With E = V^-H T:   lam = eig(B),   B E = E Lam,   U = Y1 E,   Phi = E^-H.
Hence the characteristic values and the column spaces of U and Phi do not
depend on Q; Q fixes the normalisation of the columns and the conditioning of
the intermediate steps.  With operator='ls' the characteristic values equal
those of the least-squares state-rate operator -- which is why (2) is now the
definition and this route is only one way to evaluate it.

MEASURED, so the trade-off is not guessed: substituting random invertible
matrices for V moved the poles by at most 1.9e-10 and the shapes by
1-MAC 4.4e-16. On the folding case `time_scale='auto'` is needed only by the
superseded DIAGONAL readout (wrong there by 1.0); the canonical readout is
already at a 5.6e-15 pole error with condT 1.069 and no rescaling. Under (2)
there is no second stage to condition at all -- the only conditioning left is
cond(E), which is intrinsic to the data. The one case where Q genuinely
re-enters is RANK DEFICIENCY, where the pseudo-inverse identity above fails and
the answer becomes an artefact of whichever Q was formed; use
`operators.ncvd_snapshot` or pass `rank`.

Original implementation: D. Chelidze (ncvd.py).  This module extends it with
operator choice, forcing, row weights, time scaling and sorting while
reproducing it exactly for the default arguments (see tests/test_core.py).
"""
import warnings

import numpy as np

__all__ = ['ncvd', 'ncvd_pair', 'decompose', 'NCVDResult', 'DEFAULT_RANK',
           'balanced_quotient', 'regression_operator', 'auto_time_scale', 'noise_variance',
           'numerical_rank']

DEFAULT_RANK = None
RANK_RTOL = 1e-12          # relative singular-value threshold of rank='auto' (1.1.2)


def numerical_rank(s, rtol=RANK_RTOL, gap=None):
    """Numerical rank of a matrix from its singular values (added in 1.1.2).

    The declared criterion used by `rank='auto'` in `ncvd_pair`, `ncvd` and
    `tracking.track`:

        r = #{ i : s_i > rtol * s_1 }                         (relative threshold)

    and, if `gap` is given, r is further cut at the FIRST index i < r with
    s_i / s_{i+1} > gap                                       (gap criterion)

    Parameters
    ----------
    s : singular values in descending order (as returned by numpy.linalg.svd).
    rtol : relative threshold, default 1e-12 (the threshold of the 1.1.1 rank
        warning and of `operators.ncvd_snapshot`).
    gap : None (threshold only, the default) or a ratio > 1, e.g. 1e3 for a
        three-decade drop.

    Returns int (0 for an all-zero matrix).
    """
    s = np.asarray(s, float).ravel()
    if s.size == 0 or not s[0] > 0:
        return 0
    r = int(np.sum(s > rtol*s[0]))
    if gap is not None and r > 1:
        if not gap > 1:
            raise ValueError('gap must be a ratio > 1')
        with np.errstate(divide='ignore'):
            ratio = s[:r - 1]/s[1:r]
        big = np.where(ratio > gap)[0]
        if big.size:
            r = int(big[0]) + 1
    return r


class NCVDResult(dict):
    """Dictionary of NCVD outputs with attribute access (rec.lam == rec['lam'])."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())

    def __repr__(self):
        k = len(self['lam'])
        return (f"NCVDResult(k={k}, operator={self.get('operator')!r}, "
                f"res={self['res']:.3e}, condT={self['condT']:.3e}, "
                f"time_scale={self.get('time_scale', 1.0):.4g})")


# --------------------------------------------------------------------------- helpers
def _rdiv(A, B):
    """A B^-1, solved rather than inverted."""
    return np.linalg.solve(B.T, A.T).T


def balanced_quotient(Y1, Y2):
    """Balanced matrix Q, symmetric in the four second-moment blocks."""
    K11, K12 = Y1.conj().T @ Y1, Y1.conj().T @ Y2
    K21, K22 = Y2.conj().T @ Y1, Y2.conj().T @ Y2
    return 0.25*(_rdiv(K21, K11) + _rdiv(K22, K12) + _rdiv(K11, K21) + _rdiv(K12, K22))


def noise_variance(X, fs=1.0, band=(0.40, 0.49)):
    """Per-column white-noise variance from the PSD plateau near Nyquist.

    Valid when the sampling rate is well above the highest mode (the band
    (0.40, 0.49) fs must contain noise only). Returns variances (p,).
    """
    from scipy.signal import welch
    X = np.asarray(X, float)
    f, P = welch(X - X.mean(0), fs=fs, nperseg=min(len(X), 4096), axis=0, detrend='linear')
    mk = (f >= band[0]*fs) & (f <= band[1]*fs)
    if not mk.any():
        mk = f >= 0.8*f[-1]
    return np.median(P[mk], axis=0)*(fs/2)


def regression_operator(Y1, Y2, operator='ls', structured_split=None, noise_var=None):
    """Operator B with Y2 ~ Y1 B.

    operator : 'ls'  ordinary least squares, B = Y1^+ Y2 (forward regression;
                     unbiased for noise in Y2 and for forcing uncorrelated with
                     the present state, biased by noise in Y1).
               'tls' total least squares on column-normalised data; treats
                     noise in Y1 and Y2 symmetrically (measurement noise only;
                     not for unmeasured forcing, which it mistakes for noise).
               'bcls' bias-compensated least squares: B = (K11 - m N)^-1 K12,
                     with N = diag(noise_var) the white sensor-noise variances
                     of the Y1 columns. Removes the errors-in-variables bias of
                     'ls'; requires structured_split (noise in the rate block
                     is then independent of the regressors).
    structured_split : int or None
        If given (= n, the number of stations) the record is assumed to be
        Y1 = [d, v], Y2 = [v, a]. Only the acceleration block is regressed, so
        the kinematic identity v = v is kept exactly and the shared velocity
        noise is not counted twice by TLS.
    """
    operator = operator.lower()
    if operator == 'ls':
        return np.linalg.lstsq(Y1, Y2, rcond=None)[0]
    if operator == 'bcls':
        if structured_split is None or noise_var is None:
            raise ValueError("'bcls' needs a structured [d, v]/[v, a] record and noise_var")
        n = structured_split
        m = Y1.shape[0]
        K11 = Y1.conj().T @ Y1 - m*np.diag(np.asarray(noise_var, float))
        Ba = np.linalg.solve(K11, Y1.conj().T @ Y2[:, n:])
        top = np.zeros((Y1.shape[1], n), dtype=Ba.dtype)
        top[n:, :] = np.eye(n)
        return np.hstack([top, Ba])
    if operator != 'tls':
        raise ValueError(f"unknown operator {operator!r}; use 'ls', 'tls' or 'bcls'")
    if structured_split is not None:
        n = structured_split
        # regress a on [d, v] by TLS, keep the kinematic block exact
        Ba = _tls(Y1, Y2[:, n:])
        top = np.zeros((Y1.shape[1], n), dtype=Ba.dtype)
        top[n:, :] = np.eye(n)                 # v = [d, v] @ [0; I]
        return np.hstack([top, Ba])
    return _tls(Y1, Y2)


def _tls(A, B):
    """Total least squares solution of A X ~ B with column equilibration."""
    sa = np.sqrt(np.mean(np.abs(A)**2, 0)); sa[sa == 0] = 1
    sb = np.sqrt(np.mean(np.abs(B)**2, 0)); sb[sb == 0] = 1
    Z = np.hstack([A/sa, B/sb])
    p = A.shape[1]
    _, _, Wh = np.linalg.svd(Z, full_matrices=False)
    W = Wh.conj().T
    W12, W22 = W[:p, p:], W[p:, p:]
    Xs = -_rdiv(W12, W22)                     # scaled solution
    return (Xs/sa[:, None])*sb[None, :]


def auto_time_scale(Y1, Y2, margin=2.0, floor=1e-8):
    """Time-unit factor c such that every pole of c^-1 B has modulus >= margin.

    The rates are divided by c (Y2 -> Y2/c), i.e. time is measured in units of
    1/c.  Poles with modulus below `floor` relative to the largest (rigid-body
    or static modes) are ignored when choosing c.
    """
    lam = np.linalg.eigvals(np.linalg.lstsq(Y1, Y2, rcond=None)[0])
    a = np.abs(lam)
    a = a[a > floor*max(a.max(), 1e-300)]
    if a.size == 0:
        return 1.0
    return float(a.min()/margin)


def _sort_order(lam):
    """Ascending |Im lam|, positive imaginary part first in each pair, then real poles."""
    keys = [(abs(l.imag) < 1e-12*max(abs(l), 1e-300), round(abs(l.imag), 12),
             -np.sign(l.imag), abs(l)) for l in lam]
    return sorted(range(len(lam)), key=lambda i: keys[i])


# --------------------------------------------------------------------------- main
def ncvd_pair(Y1, Y2, rank=None, center=False, operator='ls', forcing=None,
              time_scale=None, weights=None, sort=True, structured_split=None,
              legacy_balanced=False, readout='eig', noise_var=None,
              rank_rtol=RANK_RTOL, rank_gap=None):
    """NCVD of a generic matrix pair (Y1, Y2) with Y2 ~ Y1 B.

    Use this for shift-map data (Y2 = Y1 delayed one sample), Hankel matrices,
    or any pair; use `ncvd` for a [disp, vel] / [vel, acc] state record.
    Rows are time instants, columns are coordinates.

    Parameters
    ----------
    rank : int, None or 'auto'
        Whiten Y1 by its SVD and keep `rank` directions. Does not change lam
        when Y1 has full column rank; repairs conditioning when the spectrum
        spans decades. Shapes are mapped back to the original coordinates.
        None (the default, unchanged since 1.0) keeps EVERY direction, which
        makes the result depend on the balanced quotient when Y1 is rank
        deficient (a RuntimeWarning says so). 'auto' (opt-in, 1.1.2) keeps the
        numerical rank of the preprocessed Y1 by `numerical_rank(s,
        rank_rtol, rank_gap)` and returns it as `rank`, with the rule as
        `rank_rule`.
    rank_rtol, rank_gap : criterion of rank='auto' (see `numerical_rank`);
        ignored otherwise.
    center : bool
        Remove each column mean (use for multi-equilibrium systems).
    operator : 'ls' or 'tls'
        Regression operator used for Sigma_2 (see `regression_operator`).
    forcing : (m, p) array or None
        Known input record F. Its contribution Y2 ~ Y1 B + F G is projected out
        before the operator is estimated (Frisch-Waugh). Pass the forcing in
        whatever units you have; no mass matrix is needed.
    time_scale : None, 'auto' or float c
        Measure time in units of 1/c while forming Q (rates divided by c).
        lam is returned in the ORIGINAL time unit. 'auto' places every pole
        outside the unit circle, where the balanced (Joukowski) map is
        one-to-one.
    weights : (m,) array or None
        Non-negative row weights (e.g. per-record weights in an ensemble).
    sort : bool
        Order modes by ascending |Im lam|, conjugate pairs adjacent.
    legacy_balanced : bool
        Superseded readout with the balanced-matrix shapes V and the
        one-to-one assignment heuristic. Cost-of-change comparisons only.
    readout : 'eig' (canonical) or 'diag' (published GCVD: lam_i = Sigma_2[i,i]).
        'diag' is provided for benchmarking only.

    Returns
    -------
    NCVDResult with
        lam   (k,)    characteristic values, lam[i] paired with Phi[:, i]
        Phi   (n2, k) characteristic shapes (regression shapes, X1 = U Phi^H)
        Psi   (n2, k) conj(Phi): the eigenvector shapes, A Psi[:, i] = lam[i] Psi[:, i]
                      for a linear record; x(t) = sum_i U[t, i] Psi[:, i]
        U     (m, k)  temporal basis (characteristic coordinates)
        R     (m, k)  residual of the second identity (whitened coords if rank)
        V, T, S2, s, share, condT, condV, resid, res, exact1, time_scale,
        operator, B (regression operator in the working coordinates),
        G (forcing gains or None)
    """
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)
    if Y1.ndim != 2 or Y2.ndim != 2 or Y1.shape[0] != Y2.shape[0]:
        raise ValueError('Y1 and Y2 must have the same number of rows')
    if center:
        Y1 = Y1 - Y1.mean(0)
        Y2 = Y2 - Y2.mean(0)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, float)).reshape(-1, 1)
        Y1, Y2 = Y1*w, Y2*w
    F = None
    if forcing is not None:
        F = np.asarray(forcing).reshape(Y1.shape[0], -1)
        if center:
            F = F - F.mean(0)
        if weights is not None:
            F = F*w

    rank_rule = None
    if isinstance(rank, str):
        if rank != 'auto':
            raise ValueError(f"rank must be None, an int or 'auto', got {rank!r}")
        rank = numerical_rank(np.linalg.svd(Y1, compute_uv=False), rank_rtol, rank_gap)
        if rank < 1:
            raise ValueError('Y1 has numerical rank 0')
        rank_rule = dict(rule='numerical rank', rtol=rank_rtol, gap=rank_gap)
    s_all = np.linalg.svd(Y1, compute_uv=False)
    if rank is None:
        nr = int(np.sum(s_all > 1e-12*max(s_all[0], 1e-300)))
        if nr < Y1.shape[1]:
            warnings.warn(
                f'Y1 has numerical rank {nr} < {Y1.shape[1]} columns. With rank=None the '
                'pseudo-inverse makes the result depend on the balanced quotient Q, which '
                'otherwise cancels exactly: on a rank-2-of-8 record, substituting a different '
                '(equally valid) Q moved the poles by 48 and destroyed the shapes. Pass '
                f'rank={nr} (or use operators.ncvd_snapshot, whose default is the numerical '
                'rank) to make the decomposition well defined.', RuntimeWarning, stacklevel=3)
        Z1, Z2, back = Y1, Y2, None
    else:
        _, s_all, Wh = np.linalg.svd(Y1, full_matrices=False)
        Wr, sr = Wh[:rank].conj().T, s_all[:rank]
        Z1, Z2 = Y1 @ (Wr/sr), Y2 @ (Wr/sr)
        back = Wr*sr[None, :]
        if structured_split is not None and operator == 'tls':
            structured_split = None            # structure lost after whitening

    # forcing: project F out of both blocks before estimating the operator
    if F is not None:
        Pf = np.linalg.pinv(F)
        P1, P2 = Z1 - F @ (Pf @ Z1), Z2 - F @ (Pf @ Z2)
    else:
        P1, P2 = Z1, Z2

    # time unit used for the balanced quotient
    if time_scale is None:
        c = 1.0
    elif isinstance(time_scale, str):
        if time_scale != 'auto':
            raise ValueError("time_scale must be None, 'auto' or a number")
        c = auto_time_scale(P1, P2)
    else:
        c = float(time_scale)

    resid = float(np.linalg.norm(P2 - P1 @ np.linalg.lstsq(P1, P2, rcond=None)[0])
                  / max(np.linalg.norm(P2), 1e-300))

    Q = balanced_quotient(P1, P2/c)
    _, V = np.linalg.eig(Q)
    VmH = np.linalg.inv(V).conj().T
    Xi = Z1 @ VmH                                        # Sigma_1 = I
    nv = None
    if operator == 'bcls':
        if rank is not None or F is not None or weights is not None:
            raise ValueError("'bcls' is available for unweighted, unwhitened, unforced records")
        nv = noise_var
    B = regression_operator(P1, P2, operator, structured_split, nv)
    S2 = V.conj().T @ B @ VmH                            # = pinv(Xi_P) P2 V^-H for 'ls'
    G = None
    if F is not None:
        G = np.linalg.lstsq(F, Z2 - Z1 @ B, rcond=None)[0]

    Emat = None
    if readout == 'diag':
        lam = np.diag(S2).copy()
        T = np.eye(len(lam), dtype=complex)
        W, U = V, Xi
        share = np.ones(len(lam))
    elif legacy_balanced:
        lam0, T = np.linalg.eig(S2)
        lam, share = _assign_legacy(lam0, T)
        W, U = V, Xi
    else:
        lam, T = np.linalg.eig(S2)                       # Sigma_2 = T Lam T^-1
        W, U = V @ np.linalg.inv(T).conj().T, Xi @ T     # Phi = V T^-H, U = Xi T
        share = np.ones(len(lam))
        Emat = VmH @ T                                   # extractor: U = Y1 E, Phi = E^-H

    if sort:
        o = _sort_order(lam)
        lam, W, U, share = lam[o], W[:, o], U[:, o], share[o]
        if readout != 'diag' and not legacy_balanced:
            T = T[:, o]
            if Emat is not None:
                Emat = Emat[:, o]
    Phi = W if back is None else back @ W
    fit = (U*lam) @ W.conj().T
    if G is not None:
        fit = fit + F @ G
    R = Z2 - fit
    out = NCVDResult(
        lam=lam, Phi=Phi, Psi=Phi.conj(), E=Emat, U=U, R=R, V=V, T=T, S2=S2, s=s_all,
        share=share, B=B, G=G,
        condT=float(np.linalg.cond(T)), condV=float(np.linalg.cond(V)), resid=resid,
        res=float(np.linalg.norm(R)/max(np.linalg.norm(Z2), 1e-300)),
        exact1=float(np.linalg.norm(Z1 - U @ W.conj().T)/max(np.linalg.norm(Z1), 1e-300)),
        time_scale=c, operator=operator, readout=('legacy' if legacy_balanced else readout))
    if rank_rule is not None:                            # opt-in path only (1.1.2)
        out['rank'] = int(rank)
        out['rank_rule'] = rank_rule
    return out


def ncvd(disp, vel, acc, rank=None, center=False, legacy_balanced=False,
         operator='ls', forcing=None, time_scale=None, weights=None, sort=True,
         structured=True, readout='eig', noise_var=None, fs=None,
         rank_rtol=RANK_RTOL, rank_gap=None):
    """The canonical NCVD of a state record.

    Parameters
    ----------
    disp, vel, acc : (m, n) arrays
        Displacement, velocity and acceleration at n stations over m snapshots.
    rank, center, legacy_balanced :
        As in the original toolbox (see `ncvd_pair`). rank='auto' (opt-in,
        1.1.2) keeps the numerical rank of [disp, vel] by the criterion
        `numerical_rank(s, rank_rtol, rank_gap)`.
    operator : 'ls' (default, canonical), 'tls' or 'bcls'
    noise_var : 'auto', (2n,) array of [disp, vel] noise variances, or None.
        Used by operator='bcls'. 'auto' estimates them from the spectral
        plateau near Nyquist and needs fs.
    forcing : (m, p) array or None   known excitation record (EMA)
    time_scale : None, 'auto' or float   time unit used for the balanced quotient
    weights : (m,) row weights or None
    sort : bool   order modes by ascending |Im lam| (default True; False keeps
        the raw eig order of the original code)
    structured : bool   keep the kinematic block exact in TLS

    Returns NCVDResult (dict) with lam, Phi, U, R, V, T, S2, s, condT, res,
    exact1, resid, share, ...; see `ncvd_pair`.  Phi has 2n rows: the first n
    are the displacement part, the last n the velocity part (= lam * first n
    for an exact linear record).
    """
    disp, vel, acc = (np.atleast_2d(np.asarray(a)) for a in (disp, vel, acc))
    if disp.shape != vel.shape or disp.shape != acc.shape:
        raise ValueError('disp, vel and acc must have the same shape (m, n)')
    n = disp.shape[1]
    if isinstance(noise_var, str):
        if fs is None:
            raise ValueError("noise_var='auto' needs fs")
        noise_var = noise_variance(np.hstack([disp, vel]), fs)
    X1 = np.hstack([disp, vel])
    X2 = np.hstack([vel, acc])
    return ncvd_pair(X1, X2, rank=rank, center=center, operator=operator,
                     forcing=forcing, time_scale=time_scale, weights=weights,
                     sort=sort, structured_split=(n if structured else None),
                     legacy_balanced=legacy_balanced, readout=readout, noise_var=noise_var,
                     rank_rtol=rank_rtol, rank_gap=rank_gap)


decompose = ncvd


def _assign_legacy(lam, T):
    """Superseded one-to-one pole-to-shape assignment. Cost-of-change use only."""
    from scipy.optimize import linear_sum_assignment
    P = np.abs(T)**2
    P = P/P.sum(0, keepdims=True)
    rows, cols = linear_sum_assignment(-P)
    out = np.empty_like(lam)
    share = np.empty(len(lam))
    out[rows] = lam[cols]
    share[rows] = P[rows, cols]
    return out, share
