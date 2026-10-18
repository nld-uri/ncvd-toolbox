"""Test systems: generally damped chains, Duffing chains, simulation and noise."""
import numpy as np

__all__ = ['chain', 'paper_chain10', 'paper_chain3', 'state_matrix', 'true_modes',
           'simulate_linear', 'simulate_forced_linear', 'simulate_nonlinear',
           'duffing_chain', 'add_noise']


def chain(n, m=1.0, k=1.0, c=0.0, c_star=None, k_ends=('fixed', 'fixed'), c_ends=None):
    """Mass matrix, damping and stiffness of an n-mass chain.

    m, k, c may be scalars or arrays (k and c: n+1 elements for fixed-fixed,
    element i joins mass i-1 and i, with the first and last joining the walls).
    c_star : dict {i: value} of extra grounded dampers at mass i (0-based).
    k_ends : 'fixed' or 'free' for the left and right ends.
    """
    m = np.broadcast_to(np.asarray(m, float), (n,))
    k = np.broadcast_to(np.asarray(k, float), (n + 1,)).copy()
    c = np.broadcast_to(np.asarray(c, float), (n + 1,)).copy()
    if k_ends[0] == 'free':
        k[0] = 0.0; c[0] = 0.0
    if k_ends[1] == 'free':
        k[-1] = 0.0; c[-1] = 0.0

    def assemble(e):
        Kx = np.zeros((n, n))
        for i in range(n):
            Kx[i, i] = e[i] + e[i + 1]
            if i < n - 1:
                Kx[i, i + 1] = Kx[i + 1, i] = -e[i + 1]
        return Kx
    M, K, C = np.diag(m), assemble(k), assemble(c)
    if c_star:
        for i, val in c_star.items():
            C[i, i] += val
    return M, C, K


def paper_chain10():
    """10-DOF chain of Li, Stein & Chelidze (2025), Appendix C.1 (Eqs. C.28-C.30).

    M = I, K tridiagonal (8, -4), C tridiagonal with diagonal
    (0.9, 0.9, 0.6, ..., 0.6) and off-diagonal -0.3: c_i = 0.3, k_i = 4, with two
    extra dampers c* = 0.3 at the first two masses.
    """
    n = 10
    M = np.eye(n)
    K = 8*np.eye(n) - 4*(np.eye(n, k=1) + np.eye(n, k=-1))
    C = np.diag([0.9, 0.9] + [0.6]*8) - 0.3*(np.eye(n, k=1) + np.eye(n, k=-1))
    return M, C, K


def paper_chain3():
    """3-DOF chain with widely spaced modes, Appendix C.2 (Eqs. C.31-C.33)."""
    M = np.eye(3)
    C = np.array([[7.07, -0.01, 0.0], [-0.01, 0.42, -0.0055], [0.0, -0.0055, 15.81]])
    K = np.array([[5000., -9., 0.], [-9., 18., -9.], [0., -9., 25000.]])
    return M, C, K


def state_matrix(M, C, K):
    """First-order matrix A of x' = A x, x = [q; q']."""
    n = M.shape[0]
    Mi = np.linalg.inv(M)
    return np.block([[np.zeros((n, n)), np.eye(n)], [-Mi @ K, -Mi @ C]])


def true_modes(M, C, K):
    """Exact poles (positive Im, ascending) and their complex displacement shapes."""
    A = state_matrix(M, C, K)
    lam, E = np.linalg.eig(A)
    idx = np.where(lam.imag > 0)[0]
    idx = idx[np.argsort(lam.imag[idx])]
    n = M.shape[0]
    return lam[idx], E[:n, idx]


def simulate_linear(M, C, K, x0, t):
    """Exact free response by modal superposition. Returns (disp, vel, acc)."""
    A = state_matrix(M, C, K)
    lam, E = np.linalg.eig(A)
    c = np.linalg.solve(E, np.asarray(x0, float))
    X = np.real((np.exp(np.outer(t, lam))*c) @ E.T)
    Xd = X @ A.T
    n = M.shape[0]
    return X[:, :n], X[:, n:], Xd[:, n:]


def simulate_forced_linear(M, C, K, f, t, x0=None):
    """Response to a sampled force f (m, n), zero-order hold, exact discretisation."""
    from scipy.linalg import expm
    n = M.shape[0]
    A = state_matrix(M, C, K)
    Bm = np.vstack([np.zeros((n, n)), np.linalg.inv(M)])
    dt = t[1] - t[0]
    Z = expm(np.block([[A, Bm], [np.zeros((n, 2*n + n))]])*dt)
    Ad, Bd = Z[:2*n, :2*n], Z[:2*n, 2*n:]
    x = np.zeros(2*n) if x0 is None else np.asarray(x0, float)
    X = np.empty((len(t), 2*n))
    for i in range(len(t)):
        X[i] = x
        x = Ad @ x + Bd @ f[i]
    acc = X @ A.T[:, n:] + f @ np.linalg.inv(M).T
    return X[:, :n], X[:, n:], acc


def duffing_chain(n=3, m=1.0, k=1.0, c=0.01, k3=None, ends=('fixed', 'fixed')):
    """Chain with cubic springs. k3 : dict {element index: coefficient} or scalar for
    a cubic spring on every element. Returns (M, C, K, fnl) with fnl(q) the
    nonlinear restoring force (added to K q)."""
    M, C, K = chain(n, m, k, c, k_ends=ends)
    ne = n + 1
    if k3 is None:
        k3 = {0: 1.0}
    if np.isscalar(k3):
        k3 = {e: float(k3) for e in range(ne)}
    left = ends[0] == 'fixed'
    right = ends[1] == 'fixed'

    def fnl(q):
        q = np.asarray(q, float)
        f = np.zeros_like(q)
        for e, a in k3.items():
            if e == 0:
                if not left:
                    continue
                s = q[..., 0]
                f[..., 0] += a*s**3
            elif e == n:
                if not right:
                    continue
                s = -q[..., n - 1]
                f[..., n - 1] -= a*s**3
            else:
                s = q[..., e] - q[..., e - 1]          # elongation of element e
                f[..., e] += a*s**3
                f[..., e - 1] -= a*s**3
        return f
    return M, C, K, fnl


def simulate_nonlinear(M, C, K, fnl, x0, t, rtol=1e-10, atol=1e-12, method='DOP853'):
    """Free response of M q'' + C q' + K q + fnl(q) = 0. Returns (disp, vel, acc)."""
    from scipy.integrate import solve_ivp
    n = M.shape[0]
    Mi = np.linalg.inv(M)

    def rhs(_, x):
        q, v = x[:n], x[n:]
        return np.concatenate([v, -Mi @ (C @ v + K @ q + fnl(q))])
    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(x0, float), t_eval=t,
                    rtol=rtol, atol=atol, method=method)
    q, v = sol.y[:n].T, sol.y[n:].T
    a = -(v @ C.T + q @ K.T + fnl(q)) @ Mi.T
    return q, v, a


def add_noise(x, snr, rng=None, db=False, per_channel=False):
    """Additive white Gaussian noise at signal-to-noise power ratio `snr`.

    Follows Li, Stein & Chelidze (2025): y = x + (Ps/SNR)^(1/2) n, n ~ N(0,1),
    with Ps the mean power of the record (per channel if per_channel=True).
    snr is a linear power ratio unless db=True. snr = inf returns x unchanged.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    if np.isinf(snr):
        return x.copy()
    r = 10**(snr/10) if db else snr
    Ps = np.mean(x**2, axis=0) if per_channel else np.mean(x**2)
    return x + np.sqrt(Ps/r)*rng.standard_normal(x.shape)
