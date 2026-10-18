"""Example 5. Frequency-energy dependence of a Duffing chain from one free decay.

Three masses, unit springs, a cubic spring (k3 = 1) grounding the first mass,
mass-proportional damping 0.003 M. Sliding-window NCVD tracks all three
resonances; every pole is validated against the measured response frequency
(band_frequency), and the window residual ||R||/||X2|| measures how much
nonlinearity each window engages.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K, fnl = sy.duffing_chain(3, k=1.0, c=0.0, k3={0: 1.0})
C = 0.003*M
lam0, E0 = sy.true_modes(M, C, K)
sh = [nc.realify(E0[:, i])[0] for i in range(3)]
sh = [x/np.abs(x).max() for x in sh]
x0 = np.r_[1.5*sh[0] + 0.6*sh[1] + 0.5*sh[2], np.zeros(3)]
fs = 20.0
t = np.arange(0, 1500, 1/fs)
d, v, acc = sy.simulate_nonlinear(M, C, K, fnl, x0, t)
tr = nc.track(d, v, acc, fs=fs, length=int(40*fs), step=int(20*fs), n_track=3, M=M)
# quality gate: compare mode-1 pole with the measured response frequency in each window
f_meas = np.array([nc.band_frequency(d[s:e, 0], 1/fs, 0.6, 1.1) for s, e in tr.edges])
print('mode-1 pole vs measured frequency, max relative difference:',
      np.nanmax(np.abs(tr.wd[:, 0] - f_meas)/f_meas))
print('linear limits', np.abs(lam0.imag), ' last window', tr.wd[-1])

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.5))
for j in range(3):
    ax[0].semilogx(tr.energy[:, j], tr.wd[:, j]/abs(lam0[j].imag), '-o', ms=2.5, color=pl._c(j),
                   label=rf'mode {j+1}')
ax[0].semilogx(tr.energy[:, 0], f_meas/abs(lam0[0].imag), 'x', ms=4, color='k', label='measured, mode 1')
ax[0].axhline(1, color=pl._ref(), ls='--', lw=0.8)
ax[0].set_xlabel(r'modal energy'); ax[0].set_ylabel(r'$\omega_d/\omega_{d,\mathrm{lin}}$')
ax[0].legend(fontsize=7)
ax[1].semilogy(tr.t, tr.res, '-o', ms=2.5, color=pl._c(0))
ax[1].set_xlabel(r'$t$ (s)'); ax[1].set_ylabel(r'$\|\mathbf{R}\|/\|\mathbf{X}_2\|$')
for j in range(3):
    ax[2].plot(tr.t, 1 - tr.mac_prev[:, j], '-', color=pl._c(j))
ax[2].set_yscale('log'); ax[2].set_xlabel(r'$t$ (s)'); ax[2].set_ylabel(r'$1-\mathrm{MAC}$ to previous window')
fig.tight_layout()
fin(fig, a.out, 'ex05_duffing_backbone.pdf')
