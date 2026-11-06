"""Synthetic free-decay records for the 1.1.2 regression tests.

Every record is built here from the toolbox's own simulators or in closed form,
so the tests depend on no file outside the repository. Each one reproduces one
defect of the 1.1.1 tracker (see test_release_112.py and the 1.1.2 changelog).

A  slowfast()        post-transient quadratic-cubic 3-DOF slow-fast decay. The
                     slaves carry a decaying static offset; with center=False it
                     appears as a slow pole (0.077 rad/s, zeta 0.06, 0.6 cycles per
                     window) below the master (0.99 rad/s).
B  switching()       exact piecewise-linear 3-DOF record: mode 2 oscillates in
                     windows 0-1 and 4-5 and is two real poles in windows 2-3, so
                     the number of oscillatory pairs drops from 3 to 2 and back.
B2 switching(swap=True)  as B, but in windows 2-3 mode 2 is replaced by an
                     unrelated oscillatory pole (1.1 rad/s, MAC 0.043 with mode 2),
                     so the pair count stays 3 while a mode disappears.
C  duffing_single()  symmetric cubic 3-DOF chain released in mode 1: the
                     antisymmetric mode never participates (rank 4 of 6 in every
                     window); one tracked mode.
D  rank_deficient()  linear 4-DOF chain released exactly in mode 1: rank 2 of 8.
"""
import numpy as np

from ncvd import systems as sy

FS = 20.0
DT = 1.0/FS


def slowfast(w2=3.0, w3=5.0, b2=0.5, b3=0.8, k3=0.2, z1=0.005, z2=0.02, a1=1.0,
             T=400.0, t_cut=100.0):
    """Record A. Master q1 (omega 1) drives two slaves through the potential
    b_i q1^2 q_i; cubic stiffness k3 on the master. Returns
    (M, d, v, a, info) with info['offset_shape'] the static slaving direction."""
    M = np.eye(3)
    K = np.diag([1.0, w2**2, w3**2])
    C = np.diag([2*z1, 2*z2*w2, 2*z2*w3])

    def fnl(q):
        q = np.asarray(q, float)
        f = np.zeros_like(q)
        f[..., 0] = 2*b2*q[..., 0]*q[..., 1] + 2*b3*q[..., 0]*q[..., 2] + k3*q[..., 0]**3
        f[..., 1] = b2*q[..., 0]**2
        f[..., 2] = b3*q[..., 0]**2
        return f
    t = np.arange(0.0, T, DT)
    x0 = np.r_[a1, -b2*a1**2/(2*w2**2), -b3*a1**2/(2*w3**2), 0.0, 0.0, 0.0]
    d, v, a = sy.simulate_nonlinear(M, C, K, fnl, x0, t)
    keep = t >= t_cut
    off = np.array([0.0, b2/w2**2, b3/w3**2])
    return M, d[keep], v[keep], a[keep], dict(offset_shape=off/np.linalg.norm(off),
                                               w_master=1.0)


def switching(L=1000, swap=False):
    """Record B (swap=False) or B2 (swap=True); 6 windows of L samples.
    Returns (d, v, a, info) with the true poles and shapes."""
    n, nw = 3, 6
    K = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    w2, Ph = np.linalg.eigh(K)
    wn = np.sqrt(w2)
    lam = np.array([-0.01*wn[0] + 1j*wn[0], -0.012*wn[1] + 1j*wn[1], -0.01*wn[2] + 1j*wn[2]])
    sig = (-0.15, -0.45)
    lam_new = -0.011 + 1.1j
    shape_new = Ph[:, 0] - Ph[:, 2] + 0.3*Ph[:, 1]        # MAC 0.043 with mode 2; a shape
    shape_new = shape_new/np.linalg.norm(shape_new)       # inside span(modes 1, 3) would make
                                                          # the state rank deficient
    rng = np.random.default_rng(3)
    tt = np.arange(L)*DT
    D, V, A = [], [], []
    for w in range(nw):
        dw, vw, aw = np.zeros((L, n)), np.zeros((L, n)), np.zeros((L, n))
        for k in range(3):
            c = [1.0, 0.6, 0.4][k]*np.exp(1j*rng.uniform(0, 2*np.pi))
            if k == 1 and w in (2, 3):
                if swap:
                    z = c*np.exp(lam_new*tt)
                    dw += 2*np.outer(z, shape_new).real
                    vw += 2*np.outer(lam_new*z, shape_new).real
                    aw += 2*np.outer(lam_new**2*z, shape_new).real
                else:
                    for s, b in zip(sig, (0.5, -0.3)):
                        e = b*np.exp(s*tt)
                        dw += np.outer(e, Ph[:, 1])
                        vw += np.outer(s*e, Ph[:, 1])
                        aw += np.outer(s*s*e, Ph[:, 1])
                continue
            z = c*np.exp(lam[k]*tt)
            dw += 2*np.outer(z, Ph[:, k]).real
            vw += 2*np.outer(lam[k]*z, Ph[:, k]).real
            aw += 2*np.outer(lam[k]**2*z, Ph[:, k]).real
        D.append(dw); V.append(vw); A.append(aw)
    return (np.vstack(D), np.vstack(V), np.vstack(A),
            dict(lam=lam, shapes=Ph, real_poles=sig, lam_new=lam_new, shape_new=shape_new))


def duffing_single(T=400.0):
    """Record C. Returns (M, d, v, a)."""
    M, C, K, fnl = sy.duffing_chain(3, k=1.0, c=0.005, k3=0.5)
    lam0, E0 = sy.true_modes(M, C, K)
    sh = np.real(E0[:, 0]*np.exp(-1j*np.angle(E0[np.argmax(np.abs(E0[:, 0])), 0])))
    x0 = np.r_[1.2*sh/np.abs(sh).max(), np.zeros(3)]
    t = np.arange(0.0, T, DT)
    d, v, a = sy.simulate_nonlinear(M, C, K, fnl, x0, t)
    return M, d, v, a


def rank_deficient(T=200.0):
    """Record D. Returns (M, d, v, a, lam1) with lam1 the only pole in the record."""
    M, C, K = sy.chain(4, 1.0, 1.0, 0.02)
    lam, E = sy.true_modes(M, C, K)
    up = np.where(lam.imag > 0)[0]
    i1 = up[np.argmin(lam.imag[up])]
    x0 = np.r_[np.real(E[:, i1]), np.real(lam[i1]*E[:, i1])]
    x0 = x0/np.abs(x0[:4]).max()
    t = np.arange(0.0, T, DT)
    d, v, a = sy.simulate_linear(M, C, K, x0, t)
    return M, d, v, a, lam[i1]
