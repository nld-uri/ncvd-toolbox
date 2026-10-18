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

__all__ = ['robustness_basis', 'deflate', 'escape', 'closure', 'project']


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


def project(M, K, fnl, P, C=None):
    """Galerkin projection on a constant basis P (n x r), q = P eta.

    Returns dict(Mr, Kr, Cr, fnl_r, Pinv); fnl_r(eta) = P^+ fnl(P eta).
    """
    P = np.asarray(P, float)
    P = P.reshape(-1, 1) if P.ndim == 1 else P
    Pinv = np.linalg.pinv(P)
    out = dict(Mr=Pinv @ M @ P, Kr=Pinv @ K @ P, Pinv=Pinv,
               fnl_r=(lambda eta: Pinv @ fnl(P @ eta)))
    out['Cr'] = None if C is None else Pinv @ C @ P
    return out
