"""Example 4. Ensemble NCVD of trials with different sampling rates, lengths and noise.

The 16 trials of Table 5 of the paper (forcing amplitude, SNR in dB, length,
sampling rate). Stacking state-rate rows is the least-squares fusion of all
trials; the error falls as trials are added.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K = sy.paper_chain10(); n = 10
lam_t, E_t = sy.true_modes(M, C, K)
Af = [9, 10, 5, 10, 8, 5, 6, 8, 10, 10, 5, 10, 10, 7, 9, 5]
SNR = [64, 67, 39, 67, 57, 38, 45, 54, 69, 69, 40, 69, 69, 52, 63, 40]
Te = [254, 277, 81, 279, 208, 74, 119, 187, 290, 292, 89, 293, 290, 171, 250, 85]
FS = [111, 100, 200, 100, 125, 200, 166, 125, 100, 100, 200, 100, 100, 142, 111, 200]
if a.quick:
    Te = [min(x, 80) for x in Te]
rng = np.random.default_rng(3)
recs = []
for j in range(16):
    t = np.arange(0, Te[j], 1/FS[j])
    d, v, acc = sy.simulate_forced_linear(M, C, K, Af[j]*rng.standard_normal((len(t), n)), t)
    recs.append(tuple(sy.add_noise(z, SNR[j], rng, db=True) for z in (d, v, acc)))


def errs(r):
    idx = nc.physical_modes(r.lam)
    j = nc.match_modes(r.lam[idx], r.Psi[:n, idx], lam_t, E_t)
    le, Pe = r.lam[idx][j], r.Psi[:n, idx][:, j]
    zt = -lam_t.real/np.abs(lam_t)
    return (np.mean(np.abs(np.abs(le) - np.abs(lam_t))/np.abs(lam_t)),
            np.mean(np.abs(-le.real/np.abs(le) - zt)/zt),
            np.mean([nc.relative_error(Pe[:, i], E_t[:, i]) for i in range(n)]))


Ns = [1, 2, 4, 8, 16]
E = {w: np.array([errs(nc.ncvd_ensemble(recs[:N], weights=w)) for N in Ns]) for w in ['equal', 'samples']}
for w, e in E.items():
    print(w, np.round(e, 4))
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.4))
for q, lab in enumerate([r'$e_{\omega_n}$', r'$e_\zeta$', r'$e_\psi$']):
    for j, (w, e) in enumerate(E.items()):
        ax[q].loglog(Ns, e[:, q], '-o', ms=3, color=pl._c(j), label=f'weights = {w}')
    ax[q].set_xlabel('number of trials'); ax[q].set_ylabel(lab)
    ax[q].set_xticks(Ns); ax[q].set_xticklabels([str(x) for x in Ns])
ax[0].legend(fontsize=7)
fig.tight_layout()
fin(fig, a.out, 'ex04_ensemble.pdf')
