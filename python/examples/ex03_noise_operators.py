"""Example 3. Sensor noise: least squares, bias-compensated LS and TLS.

Operational setting of the paper (Sec. 6.2): the 10-DOF chain driven by
unmeasured white forcing at every mass, sampled at 200 Hz for 150 s, with
independent white noise on every displacement, velocity and acceleration
channel at SNR (dB). The canonical NCVD poles are least-squares poles; noise in
the regressors biases them. 'bcls' removes that bias using noise variances
estimated from the spectral plateau near Nyquist. 'tls' assumes all misfit is
measurement noise and fails under unmeasured forcing.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K = sy.paper_chain10(); n = 10
lam_t, E_t = sy.true_modes(M, C, K)
fs, T = 200.0, 150.0
snrs = [np.inf, 40, 30, 25, 20, 15]
n_real = 3 if a.quick else 10
rng = np.random.default_rng(0)


def med_err(lam):
    idx = nc.physical_modes(lam); le = lam[idx]
    j = [np.argmin(np.abs(le - l)) for l in lam_t]; le = le[j]
    ef = np.abs(np.abs(le) - np.abs(lam_t))/np.abs(lam_t)
    zt = -lam_t.real/np.abs(lam_t)
    ez = np.abs(-le.real/np.abs(le) - zt)/zt
    return np.median(ef), np.median(ez)


ops = {'LS (canonical)': dict(), 'BCLS, estimated noise': dict(operator='bcls', noise_var='auto', fs=fs),
       'TLS': dict(operator='tls')}
res = {k: np.zeros((len(snrs), n_real, 2)) for k in ops}
for i, s in enumerate(snrs):
    for r in range(n_real):
        t = np.arange(0, T, 1/fs)
        f = 10*rng.standard_normal((len(t), n))
        d, v, acc = sy.simulate_forced_linear(M, C, K, f, t)
        dn, vn, an = (sy.add_noise(z, s, rng, db=True, per_channel=True) for z in (d, v, acc))
        for k, kw in ops.items():
            res[k][i, r] = med_err(nc.ncvd(dn, vn, an, **kw).lam)
xs = np.array([50 if np.isinf(s) else s for s in snrs])
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
for j, (k, R) in enumerate(res.items()):
    for q in range(2):
        m, sd = R[:, :, q].mean(1), R[:, :, q].std(1)
        ax[q].errorbar(xs, m, sd, fmt='-o', ms=3, capsize=2, color=pl._c(j), label=k)
        print(k, ['freq', 'damping'][q], np.round(m, 4))
for q, lab in enumerate([r'median $|\hat\omega_n-\omega_n|/\omega_n$', r'median $|\hat\zeta-\zeta|/\zeta$']):
    ax[q].set_yscale('log'); ax[q].set_xlabel('SNR (dB)'); ax[q].set_ylabel(lab)
    ax[q].set_xticks(xs); ax[q].set_xticklabels([r'$\infty$' if np.isinf(s) else f'{s:g}' for s in snrs])
    ax[q].invert_xaxis()
ax[0].legend(fontsize=7.5)
fig.tight_layout()
fin(fig, a.out, 'ex03_noise_operators.pdf')
