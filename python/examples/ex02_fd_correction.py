"""Example 2. Displacement-only data: FIR derivatives and exact pole correction.

Rates from any FIR stencil D give poles d(exp(lam dt)) instead of lam;
correct_poles inverts the stencil's symbol (generalising Eq. 48 of the paper).
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K = sy.paper_chain3()                         # 4.24, 70.7, 158 rad/s, zeta 0.05
lam_t, _ = sy.true_modes(M, C, K)
methods = ['forward', 'central', 'central4', 'savgol']
labels = ['forward', 'central', 'five-point', 'Savitzky--Golay (7, 3)']
fsv = np.geomspace(60, 3000, 18)
raw = np.zeros((len(methods), len(fsv))); cor = np.zeros_like(raw)
for j, fs in enumerate(fsv):
    t = np.arange(0, 3.0, 1/fs)
    q, *_ = sy.simulate_linear(M, C, K, np.r_[1, 1, 1, 0, 0, 0], t)
    for i, m in enumerate(methods):
        d, v, acc, f, _ = nc.state_from_displacement(q, fs, m)
        lam = nc.ncvd(d, v, acc).lam
        lam = lam[nc.physical_modes(lam)]
        lc = nc.correct_poles(lam, f, fs)
        raw[i, j] = max(np.min(np.abs(lam - l))/abs(l) for l in lam_t)
        cor[i, j] = max(np.min(np.abs(lc - l))/abs(l) for l in lam_t)
wmax = np.abs(lam_t.imag).max()
for i, m in enumerate(methods):
    b = nc.deriv_filter(m).band()
    ok = wmax/fsv < b
    print(f'{m:9s} band omega*dt < {b:.3f};  max corrected error inside band: {cor[i, ok].max():.1e}')
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for i in range(len(methods)):
    ax.loglog(fsv, raw[i], '-', color=pl._c(i), label=labels[i])
    ax.loglog(fsv, np.maximum(cor[i], 1e-16), ':o', ms=2.5, color=pl._c(i))
for i, m in enumerate(methods):
    ax.axvline(wmax/nc.deriv_filter(m).band(), color=pl._c(i), ls='--', lw=0.7)
ax.set_xlabel(r'sampling rate $f_s$ (Hz)')
ax.set_ylabel(r'$\max_i |\hat\lambda_i-\lambda_i|/|\lambda_i|$')
ax.text(0.03, 0.40, r'dotted: corrected; dashed: invertible band', transform=ax.transAxes, fontsize=8)
ax.legend(fontsize=8, loc='center right')
fig.tight_layout()
fin(fig, a.out, 'ex02_fd_correction.pdf')
