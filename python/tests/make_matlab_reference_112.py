"""Write matlab/tests/reference_112.mat: the synthetic records of _records.py
and the Python outputs of the 1.1.2 opt-in calls, for the MATLAB cross-checks."""
import os
import sys
import numpy as np
from scipy.io import savemat

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, '..'), here]
import ncvd as nc                     # noqa: E402
from ncvd import rom                  # noqa: E402
import _records as R                  # noqa: E402

out = {}
MA, dA, vA, aA, infoA = R.slowfast()
out.update(A_M=MA, A_d=dA, A_v=vA, A_a=aA, A_offset=infoA['offset_shape'])
w = nc.windowed(dA, vA, aA, R.FS, n_windows=6, n_track=1, M=MA)
out.update(A_windowed_wd=w.wd, A_windowed_zero_lam=w.zero.lam, A_windowed_rank=w.rank)
t = nc.track(dA, vA, aA, fs=R.FS, n_windows=6, n_track=1, M=MA, center=False, seed=[2.0],
             families=True, matching='identity')
out['A_target_wd'] = t.wd
dB, vB, aB, infoB = R.switching()
out.update(B_d=dB, B_v=vB, B_a=aB, B_lam=infoB['lam'])
t = nc.track(dB, vB, aB, fs=R.FS, n_windows=6, n_track=3, matching='identity', families=True, n_zero=2)
out.update(B_identity_wd=t.wd, B_identity_gap=t.gap.astype(float), B_zero_lam=t.zero.lam)
dB2, vB2, aB2, infoB2 = R.switching(swap=True)
out.update(B2_d=dB2, B2_v=vB2, B2_a=aB2)
t = nc.track(dB2, vB2, aB2, fs=R.FS, n_windows=6, n_track=3, matching='identity', max_cost=0.5)
out.update(B2_gate_wd=t.wd, B2_gate_cost=t.cost)
MC, dC, vC, aC = R.duffing_single()
out.update(C_M=MC, C_d=dC, C_v=vC, C_a=aC)
t = nc.track(dC, vC, aC, fs=R.FS, n_windows=10, n_track=1, M=MC, rank='auto')
_, _, mag = rom.deflate(t.shape[:, 0, :])
P, sv = rom.family_closure(t.shape[:, 0, :], tol=0.5*mag)
out.update(C_rankauto_wd=t.wd, C_rankauto_rank=t.rank, C_family_P=P, C_family_sv=sv, C_family_tol=0.5*mag)
MD, dD, vD, aD, lam1 = R.rank_deficient()
out.update(D_M=MD, D_d=dD, D_v=vD, D_a=aD, D_lam1=lam1)
savemat(os.path.join(here, '..', '..', 'matlab', 'tests', 'reference_112.mat'), out, do_compression=True)
print('wrote reference_112.mat with', len(out), 'arrays')
