"""Example 1. Exact NCVD of a generally damped 10-DOF chain (paper Appendix C.1).

Shows the two identities, the poles against the truth, and complex shapes.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K = sy.paper_chain10()
t = np.arange(0, 60, 0.05)
d, v, acc = sy.simulate_linear(M, C, K, np.random.default_rng(0).standard_normal(20), t)
r = nc.ncvd(d, v, acc)
print(r)
lam_t, E_t = sy.true_modes(M, C, K)
idx = nc.physical_modes(r.lam)
print('pole error   ', np.max(np.abs(r.lam[idx] - lam_t)))
print('identity 1   ', r.exact1, '  identity 2', r.res)
mp = nc.modal_parameters(r.lam[idx])
for i in range(10):
    print(f'mode {i+1:2d}  wn {mp["wn"][i]:.5f}  zeta {mp["zeta"][i]:.5f}  '
          f'MAC {nc.mac(r.Psi[:10, idx[i]], E_t[:, i]):.12f}')

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
pl.plot_poles(ax[0], [r.lam], labels=['NCVD'], true=np.r_[lam_t, lam_t.conj()])
for j, i in enumerate([0, 1, 3]):
    est = nc.normalize_shape(r.Psi[:10, idx[i]])
    tru = nc.normalize_shape(E_t[:, i])
    z = np.r_[0, tru]
    ax[1].plot(z.real, z.imag, '-', color=pl._ref(), lw=2.4)
    z = np.r_[0, est]
    ax[1].plot(z.real, z.imag, 'o', ms=3.2, color=pl._c(j), label=rf'mode {i+1}')
ax[1].set_xlabel(r'$\mathrm{Re}\,\psi$'); ax[1].set_ylabel(r'$\mathrm{Im}\,\psi$')
ax[1].legend(loc='lower left')
fig.tight_layout()
fin(fig, a.out, 'ex01_linear_chain.pdf')
