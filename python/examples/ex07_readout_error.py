"""Example 7. When is the published (diagonal) GCVD readout exact?

The published readout takes lam_i = Sigma_2[i, i] in the basis V of the
balanced quotient Q. It is exact when V diagonalises Sigma_2, i.e. when V are
eigenvectors of the regression operator B. For exact linear data
Q = (A + A^-1)/2 shares A's eigenvectors, so the readout is exact unless two
eigenvalues of Q coincide: mu = (lam + 1/lam)/2 folds lam and 1/lam together
(Joukowski map), so a pole with |lam| = 1 in the chosen time unit collides with
its conjugate. Panel (a): near that tie the off-diagonal part grows like
eps/|c - 1| (c = time-unit factor).

On nonlinear (or ensemble-averaged) data V no longer diagonalises Sigma_2.
Panel (b): sliding windows on a Duffing chain. The diagonal readout's pole
error is second order in the off-diagonal ratio; the shapes differ to first
order.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)


def offd(S):
    D = np.diag(np.diag(S)); return np.linalg.norm(S - D)/np.linalg.norm(D)


# (a) exact Joukowski tie: 4-DOF chain with omega = 1 exactly
n = 4
K = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1); K[-1, -1] = 1
C = 0.002*np.eye(n) + 0.001*K
t = np.arange(0, 200, 0.05)
d, v, acc = sy.simulate_linear(np.eye(n), C, K, np.random.default_rng(1).standard_normal(2*n), t)
deltas = np.r_[0, np.geomspace(1e-9, 1e-1, 33)]
off_a, err_a = [], []
lam = np.linalg.eigvals(sy.state_matrix(np.eye(n), C, K))
for dl in deltas:
    c = 1 + dl
    r = nc.ncvd(d, v/c, acc/c**2, readout='diag')
    off_a.append(offd(r.S2))
    err_a.append(max(np.min(np.abs(lam/c - x)) for x in r.lam)/np.abs(lam/c).min())
print('exact tie (c = 1): off-diagonal %.2e, published pole error %.2e' % (off_a[0], err_a[0]))

# (b) nonlinear windows
M, C, K, fnl = sy.duffing_chain(3, k=1.0, c=0.0, k3={0: 1.0}); C = 0.003*M
lam0, E0 = sy.true_modes(M, C, K)
sh = [nc.realify(E0[:, i])[0] for i in range(3)]; sh = [x/np.abs(x).max() for x in sh]
fs = 20.0; t = np.arange(0, 1500, 1/fs)
O, Pe, Se = [], [], []
for amp in [1.0, 2.0, 3.0]:
    d, v, acc = sy.simulate_nonlinear(M, C, K, fnl, np.r_[amp*sh[0] + 0.6*sh[1] + 0.5*sh[2], np.zeros(3)], t)
    for s0 in range(0, len(t) - 800, 1200):
        s = slice(s0, s0 + 800)
        r = nc.ncvd(d[s], v[s], acc[s], center=True)
        g = nc.ncvd(d[s], v[s], acc[s], center=True, readout='diag')
        ic, ig = nc.physical_modes(r.lam), nc.physical_modes(g.lam)
        j = [ig[np.argmin(np.abs(g.lam[ig] - r.lam[i]))] for i in ic]
        O.append(offd(r.S2))
        Pe.append(np.max(np.abs(g.lam[j] - r.lam[ic])/np.abs(r.lam[ic])))
        Se.append(max(1 - nc.mac(g.Phi[:3, jj].conj(), r.Psi[:3, ii]) for ii, jj in zip(ic, j)))
O, Pe, Se = map(np.array, (O, Pe, Se))
p2 = np.polyfit(np.log(O), np.log(Pe), 1)[0]
p1 = np.polyfit(np.log(O), np.log(np.sqrt(Se)), 1)[0]
print(f'nonlinear windows: off-diagonal {O.min():.1e}..{O.max():.1e}; '
      f'pole-difference slope {p2:.2f}; sqrt(1-MAC) slope {p1:.2f}')

fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7))
ax[0].loglog(deltas[1:], off_a[1:], 'o-', ms=2.5, color=pl._c(0), label=r'$\|\mathrm{off}\,\Sigma_2\|/\|\mathrm{diag}\,\Sigma_2\|$')
ax[0].loglog(deltas[1:], np.maximum(err_a[1:], 1e-17), 's-', ms=2.5, color=pl._c(1), label='published pole error')
ax[0].loglog(deltas[1:], 4e-17/deltas[1:], '--', color=pl._ref(), lw=0.8, label=r'$\propto |c-1|^{-1}$')
ax[0].set_xlabel(r'$|c-1|$ (pole modulus $|\lambda|/c$ near 1)'); ax[0].legend(fontsize=7)
ax[0].set_title(rf'at $c=1$: off-diagonal {off_a[0]:.2f}, pole error {err_a[0]:.1f}', fontsize=8)
ax[1].loglog(O, Pe, 'o', ms=3, color=pl._c(1), label=r'pole difference $|\lambda^{\mathrm{diag}}-\lambda|/|\lambda|$')
ax[1].loglog(O, np.sqrt(Se), '^', ms=3, color=pl._c(0), label=r'shape difference $\sqrt{1-\mathrm{MAC}}$')
xx = np.geomspace(O.min(), O.max(), 10)
ax[1].loglog(xx, xx**2*Pe.max()/O.max()**2, '--', color=pl._ref(), lw=0.8, label='slope 2')
ax[1].loglog(xx, xx*np.sqrt(Se).max()/O.max(), ':', color=pl._ref(), lw=0.8, label='slope 1')
ax[1].set_xlabel(r'off-diagonal ratio of $\Sigma_2$'); ax[1].legend(fontsize=6.5, loc='upper left')
fig.tight_layout()
fin(fig, a.out, 'ex07_readout_error.pdf')
