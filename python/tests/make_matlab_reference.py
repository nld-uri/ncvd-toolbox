"""Write matlab/tests/reference.mat: inputs and Python outputs for cross-checking."""
import os, sys
import numpy as np
from scipy.io import savemat
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
import ncvd as nc
from ncvd import systems as sy

out = {}
M, C, K = sy.paper_chain10()
t = np.arange(0, 60, 0.05)
d, v, a = sy.simulate_linear(M, C, K, np.random.default_rng(0).standard_normal(20), t)
r = nc.ncvd(d, v, a)
out.update(c10_d=d, c10_v=v, c10_a=a, c10_lam=r.lam, c10_res=r.res, c10_Psi=r.Psi)
rng = np.random.default_rng(1)
dn, vn, an = (sy.add_noise(z, 30, rng, db=True, per_channel=True) for z in (d, v, a))
out.update(c10n_d=dn, c10n_v=vn, c10n_a=an)
for op in ['ls', 'tls']:
    out[f'c10n_lam_{op}'] = nc.ncvd(dn, vn, an, operator=op).lam
out['c10n_lam_bcls'] = nc.ncvd(dn, vn, an, operator='bcls', noise_var='auto', fs=20.0).lam
out['c10n_noisevar'] = nc.noise_variance(np.hstack([dn, vn]), 20.0)
out['c10n_lam_diag'] = nc.ncvd(dn, vn, an, readout='diag').lam
out['c10n_lam_rank'] = nc.ncvd(dn, vn, an, rank=16).lam
# FD correction
M3, C3, K3 = sy.chain(3, 1, 1, 0.05)
q, *_ = sy.simulate_linear(M3, C3, K3, np.random.default_rng(4).standard_normal(6), np.arange(0, 80, 0.1))
out['fd_q'] = q
for m in ['forward', 'central', 'central4', 'savgol']:
    dd, vv, aa, f, _ = nc.state_from_displacement(q, 10.0, m)
    lam = nc.ncvd(dd, vv, aa).lam
    out[f'fd_{m}_raw'] = lam
    out[f'fd_{m}_cor'] = nc.correct_poles(lam, f, 10.0)
    out[f'fd_{m}_band'] = f.band()
# Duffing tracking
Md, Cd, Kd, fnl = sy.duffing_chain(3, k=1.0, c=0.0, k3={0: 1.0}); Cd = 0.003*Md
t2 = np.arange(0, 600, 0.05)
lam0, E0 = sy.true_modes(Md, Cd, Kd)
sh = [nc.realify(E0[:, i])[0] for i in range(3)]; sh = [x/np.abs(x).max() for x in sh]
x0 = np.r_[1.5*sh[0] + 0.6*sh[1] + 0.5*sh[2], np.zeros(3)]
qd, vd, ad = sy.simulate_nonlinear(Md, Cd, Kd, fnl, x0, t2)
tr = nc.track(qd, vd, ad, fs=20, length=800, step=400, n_track=3, M=Md)
out.update(duf_d=qd, duf_v=vd, duf_a=ad, duf_wd=tr.wd, duf_res=tr.res, duf_x0=x0, duf_energy=tr.energy)
savemat(os.path.join(here, '..', '..', 'matlab', 'tests', 'reference.mat'), out, do_compression=True)
print('wrote reference.mat with', len(out), 'arrays')
