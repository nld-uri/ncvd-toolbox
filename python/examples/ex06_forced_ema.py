"""Example 6. Measured forcing (EMA): exact poles from forced response.

A 4-DOF chain driven at three of its four masses by measured random forces. Passing the
force record projects it out of both blocks before the operator is estimated;
no mass matrix is needed. Without it, the forced part of the rate data leaks
into the poles.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import ncvd as nc
from ncvd import systems as sy, plotting as pl

a = setup(__doc__)
M, C, K = sy.chain(4, 1, 2, 0.05)
lam_t, _ = sy.true_modes(M, C, K)
t = np.arange(0, 60, 0.02)
rng = np.random.default_rng(1)
f = rng.standard_normal((len(t), 4))*np.r_[1, 0.5, 0.0, 2.0]       # mass 3 not driven
d, v, acc = sy.simulate_forced_linear(M, C, K, f, t)
r_f = nc.ncvd(d, v, acc, forcing=f[:, [0, 1, 3]])
r_0 = nc.ncvd(d, v, acc)
for name, r in [('with forcing', r_f), ('without', r_0)]:
    le = r.lam[nc.physical_modes(r.lam)]
    print(f'{name:13s} pole error {np.max(np.abs(le - lam_t)):.2e}  res {r.res:.1e}')
fig, ax = plt.subplots(figsize=(3.4, 2.8))
pl.plot_poles(ax, [r_0.lam, r_f.lam], labels=['forcing ignored', 'forcing supplied'],
              true=np.r_[lam_t, lam_t.conj()])
fig.tight_layout()
fin(fig, a.out, 'ex06_forced_ema.pdf')
