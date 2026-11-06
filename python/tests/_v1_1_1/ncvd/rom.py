"""Reduced-order-model bases from tracked characteristic shapes.

Ported from the CVD/NNM workflow of the Nonlinear Dynamics Laboratory
(Stein & Chelidze, Nonlinear Dynamics 113 (2025) 25-45, for the windowed
GCVD-ROM idea). Each tracked resonance gives a family of shapes over the
swept energy range (a resonance robustness matrix). The functions below
extract the near-linear shape and its amplitude correction, measure whether
the corrections leave a retained subspace (escape), size the subspace by
closure, and project the equations of motion.

Note: these diagnostics are computed from shapes. Values obtained with the
canonical NCVD shapes differ from values obtained with the superseded
balanced-matrix shapes and must not be mixed.
"""
import numpy as np

__all__ = ['robustness_basis', 'deflate', 'escape', 'closure', 'project',
           'residual_augmentation', 'augment']


def robustness_basis(shapes, rank=None, tol=1e-3):
    """Orthonormal basis of one resonance's shape family and its normalised
    singular values. shapes : (n_windows, n)."""
    U, sv, _ = np.linalg.svd(np.asarray(shapes, float).T, full_matrices=False)
    sv = sv/sv[0]
    r = rank if rank is not None else max(1, int(np.sum(sv > tol)))
    return U[:, :r], sv


def deflate(shapes, ref=None):
    """Near-linear shape v0, leading amplitude correction, and its relative size.

    shapes : (n_windows, n), LAST row = lowest-amplitude window (default reference).
    Returns (v0, corr, mag); mag = leading residual singular value / leading
    singular value of the family.
    """
    S = np.asarray(shapes, float).T
    sv_family = np.linalg.svd(S, compute_uv=False)
    v = np.asarray(S[:, -1] if ref is None else ref, float)
    v0 = v/max(np.linalg.norm(v), 1e-300)
    Rr = S - np.outer(v0, v0 @ S)
    U, sv, _ = np.linalg.svd(Rr, full_matrices=False)
    c = U[:, 0]
    if c[int(np.argmax(np.abs(c)))] < 0:
        c = -c
    return v0, c, float(sv[0]/max(sv_family[0], 1e-300))


def escape(V_kept, corrections):
    """Norm of the part of each unit correction outside span(V_kept)."""
    Q, _ = np.linalg.qr(np.asarray(V_kept, float))
    C = np.asarray(corrections, float)
    C = C.reshape(-1, 1) if C.ndim == 1 else C
    return np.linalg.norm(C - Q @ (Q.T @ C), axis=0)


def closure(V0, corrections, k=1, tol=0.05, p_max=None):
    """Smallest P such that the first k corrections escape span(V0[:, :P]) by
    less than tol. Returns None if not met within p_max. Do not pose with k = P."""
    V = np.asarray(V0, float)
    hi = V.shape[1] if p_max is None else min(p_max, V.shape[1])
    C = np.asarray(corrections, float)[:, :k]
    for P in range(max(k, 1), hi + 1):
        if escape(V[:, :P], C).max() < tol:
            return P
    return None


def residual_augmentation(R, K, M=None, n_space=None, n_dir=1):
    """Basis directions implied by the NCVD residual: a residual-informed static
    correction.

    The residual of the canonical NCVD is the nonlinear force field projected
    off the data span, so its leading spatial direction IS the direction of the
    nonlinear force (recovered to MAC 1.000000 on the reference systems). The
    displacement response that force drives is the static correction K^-1 w, and
    that is the direction a reduced basis is missing -- not the force direction
    itself.

    Parameters
    ----------
    R : (m, n) or (m, 2n) residual from the decomposition.
    K : (n, n) stiffness. M : (n, n) mass or None (maps the acceleration-space
        residual direction back to a force direction, w = M d).
    n_space : n, when R spans the full [kinematic, dynamic] state.
    n_dir : number of directions to return (leading singular directions of R).

    Returns
    -------
    (directions (n, n_dir) orthonormal, shares (n_dir,) singular-value shares).

    WHEN IT HELPS (measured, 10-DOF chain, in-band release, equal basis size).
    Spending the extra basis vector on this correction rather than on the next
    linear mode reduced the response error by 1.4x to 3.2x wherever the
    nonlinear force reached ~0.3 of the linear restoring force, and was
    break-even (0.55x to 0.98x) where it was weaker, i.e. where every basis
    works. The gain tracked the strength of the nonlinearity, NOT the escape of
    the correction from the retained band.
    """
    R = np.asarray(R).real
    Rd = R if n_space is None else R[:, n_space:]
    _, sv, Wh = np.linalg.svd(Rd, full_matrices=False)
    n_dir = int(max(1, min(n_dir, Wh.shape[0])))
    D = Wh[:n_dir].real.T                                   # (n, n_dir)
    if M is not None:
        D = np.asarray(M, float) @ D                        # force direction w = M d
    S = np.linalg.solve(np.asarray(K, float), D)            # static correction K^-1 w
    Qd, _ = np.linalg.qr(S)
    shares = sv[:n_dir]**2/max(np.sum(sv**2), 1e-300)
    return Qd[:, :n_dir], shares


def augment(V, extra):
    """Orthonormal basis spanning [V, extra], `extra` appended after V."""
    V = np.asarray(V, float); E = np.asarray(extra, float)
    V = V.reshape(-1, 1) if V.ndim == 1 else V
    E = E.reshape(-1, 1) if E.ndim == 1 else E
    return np.linalg.qr(np.column_stack([V, E]))[0]


def project(M, K, fnl, P, C=None):
    """Galerkin projection on a constant basis P (n x r), q = P eta.

    Returns dict(Mr, Kr, Cr, fnl_r, Pinv); fnl_r(eta) = P^+ fnl(P eta).

    NOTE: P should be ORTHONORMAL (as returned by `robustness_basis`, `augment`
    or `residual_augmentation`). For orthonormal P, P^+ = P^T and the reduced
    matrices inherit the symmetry of M, K and C. For a non-orthonormal P,
    P^+ M P is NOT symmetric and the reduced model loses the energy structure
    of the original; a warning is issued.
    """
    P = np.asarray(P, float)
    P = P.reshape(-1, 1) if P.ndim == 1 else P
    if P.shape[1] and not np.allclose(P.T @ P, np.eye(P.shape[1]), atol=1e-8):
        import warnings
        warnings.warn('basis P is not orthonormal; the reduced matrices will not be '
                      'symmetric and the ROM loses the energy structure. '
                      'Orthonormalise with rom.augment or numpy.linalg.qr.',
                      RuntimeWarning, stacklevel=2)
    Pinv = np.linalg.pinv(P)
    out = dict(Mr=Pinv @ M @ P, Kr=Pinv @ K @ P, Pinv=Pinv,
               fnl_r=(lambda eta: Pinv @ fnl(P @ eta)))
    out['Cr'] = None if C is None else Pinv @ C @ P
    return out
