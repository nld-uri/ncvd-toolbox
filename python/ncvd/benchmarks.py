"""Reference methods for benchmarking: published GCVD readout, DMD, ITD.

These are provided to reproduce comparisons, not to produce results.
"""
import numpy as np
from .core import ncvd_pair, balanced_quotient, _rdiv

__all__ = ['gcvd', 'dmd', 'itd', 'shift_pair']


def gcvd(disp, vel, acc, quotient='balanced', center=False):
    """GCVD as published (Li, Stein & Chelidze 2025, Eqs. 11-14 with the quotient of Eq. 16, 10 or 43).

    quotient : 'balanced' (Eq. 16), 'forward' (Eq. 10, R' = K21 K11^-1) or
               'forcing' (Eq. 43, (K11 K21^-1 + K21 K11^-1)/2).
    Readout: shapes V (eigenvectors of the quotient), characteristic values
    r_i = s2_ii / s1_i (diagonal of Sigma_2).  Returns dict(lam, Phi, U, S2).
    """
    Y1 = np.hstack([disp, vel]); Y2 = np.hstack([vel, acc])
    if center:
        Y1 = Y1 - Y1.mean(0); Y2 = Y2 - Y2.mean(0)
    K11, K12 = Y1.T @ Y1, Y1.T @ Y2
    K21, K22 = Y2.T @ Y1, Y2.T @ Y2
    if quotient == 'balanced':
        R = balanced_quotient(Y1, Y2)
    elif quotient == 'forward':
        R = _rdiv(K21, K11)
    elif quotient == 'forcing':
        R = 0.5*(_rdiv(K11, K21) + _rdiv(K21, K11))
    else:
        raise ValueError(quotient)
    _, V = np.linalg.eig(R)
    W = np.linalg.inv(V).conj().T             # adjoint modes, V = W^-H
    P1, P2 = Y1 @ W, Y2 @ W
    s1 = np.linalg.norm(P1, axis=0)
    U = P1/s1
    S2 = np.diag(s1) @ np.linalg.pinv(P1) @ P2
    lam = np.diag(S2)/s1
    return dict(lam=lam, Phi=V, U=U, S2=S2)


def shift_pair(X, lag=1):
    """(X[:-lag], X[lag:]) for discrete-time methods."""
    return X[:-lag], X[lag:]


def dmd(X, dt, rank=None):
    """Exact DMD of snapshot rows X (m, p). Returns dict(lam, Phi, mu).

    lam = log(mu)/dt are continuous-time poles, Phi the exact DMD modes.
    """
    X1, X2 = X[:-1].T, X[1:].T                # columns = snapshots
    Ur, s, Vh = np.linalg.svd(X1, full_matrices=False)
    r = len(s) if rank is None else rank
    Ur, s, Vr = Ur[:, :r], s[:r], Vh[:r].conj().T
    At = Ur.conj().T @ X2 @ Vr / s
    mu, Wt = np.linalg.eig(At)
    Phi = X2 @ Vr / s @ Wt
    return dict(lam=np.log(mu.astype(complex))/dt, Phi=Phi, mu=mu)


def itd(X, dt, rank=None):
    """Ibrahim time-domain method (Eqs. 27-30): two-block Hankel matrices.

    H1 = [x_k; x_{k+1}],  H2 = [x_{k+2}; x_{k+3}];  H2 = Phi_i Theta^2 Xi.
    Returns dict(lam, Phi) with Phi the top block of the ITD shapes.
    """
    m, p = X.shape
    H1 = np.hstack([X[0:m-3], X[1:m-2]])      # rows = time
    H2 = np.hstack([X[2:m-1], X[3:m]])
    if rank is not None:
        _, _, Wh = np.linalg.svd(H1, full_matrices=False)
        Wr = Wh[:rank].T
        H1r, H2r = H1 @ Wr, H2 @ Wr
    else:
        Wr, H1r, H2r = None, H1, H2
    B = np.linalg.lstsq(H1r, H2r, rcond=None)[0]   # H2 = H1 B
    mu2, E = np.linalg.eig(B)
    Phi = np.linalg.inv(E).conj().T                 # H1 = (H1 E) E^-1, shapes E^-H
    if Wr is not None:
        Phi = Wr @ Phi
    lam = np.log(mu2.astype(complex))/(2*dt)
    return dict(lam=lam, Phi=Phi[:p])
