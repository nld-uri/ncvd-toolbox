"""Replication: does the canonical readout change the results of
Li, Stein & Chelidze, MSSP 222 (2025) 111769?

For each numerical experiment we compute the published readout (diagonal of
Sigma_2 in the basis of the quotient used in the paper's public code,
github.com/lhwx1224/CVD_GCVD) and the canonical NCVD readout (eigenvalues of
Sigma_2), on the same data.

  Sec. 6.2  single record, random forcing, quotient 'forced' (Eq. 43)
  Sec. 6.3  16-trial ensemble (Table 5), averaged 'forced' quotient (Eq. 59)
  Sec. 6.4  3-DOF free decay (App. C.2), balanced quotient (Eq. 16)

Setups follow the public scripts (dt = 0.005 s, 150 s, forcing amplitude 10,
analytic rates, awgn-style noise with SNR in dB on Y and dY). Writes
paper_replication.json next to this file.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ncvd as nc
from ncvd import systems as sy
from ncvd.core import _rdiv, balanced_quotient

M, C, K = sy.paper_chain10(); n = 10
lam_t, _ = sy.true_modes(M, C, K)


def sim(dt, tend, amp, rng):
    t = np.arange(0, tend + dt/2, dt)
    d, v, a = sy.simulate_forced_linear(M, C, K, amp*rng.standard_normal((len(t), n)), t)
    return np.hstack([d, v]), np.hstack([v, a])


def awgn_db(X, snr_db, rng):
    return X if np.isinf(snr_db) else sy.add_noise(X, snr_db, rng, db=True)


def readouts(Y, dY, Cq):
    """Published code: [~, L, W] = eig(C) (left eigenvectors), P = Y W,
    S2 = S1 pinv(P1) P2, R = S2/S1; poles = diag(R). Canonical: eig(R)."""
    _, W = np.linalg.eig(Cq.conj().T)
    P1, P2 = Y @ W, dY @ W
    s1 = np.linalg.norm(P1, axis=0)
    R = (np.diag(s1) @ np.linalg.pinv(P1) @ P2)/s1[None, :]
    dg, ev = np.diag(R), np.linalg.eigvals(R)
    off = np.linalg.norm(R - np.diag(dg))/np.linalg.norm(dg)
    return dg, ev, off


def errs(lam, ref):
    idx = nc.physical_modes(lam); le = lam[idx]
    le = le[[np.argmin(np.abs(le - l)) for l in ref]]
    zt = -ref.real/np.abs(ref)
    return (float(np.mean(np.abs(np.abs(le) - np.abs(ref))/np.abs(ref))),
            float(np.mean(np.abs(-le.real/np.abs(le) - zt)/zt)))


def diff(dg, ev):
    return float(max(np.min(np.abs(ev - x))/abs(x) for x in dg))


out = {}
rng = np.random.default_rng(1)
rows = []
for snr in [np.inf, 54.5, 44.5, 34.5]:
    for rep in range(3):
        Y, dY = sim(0.005, 150, 10, rng)
        Y, dY = awgn_db(Y, snr, rng), awgn_db(dY, snr, rng)
        K11, K21 = Y.T @ Y, dY.T @ Y
        dg, ev, off = readouts(Y, dY, 0.5*(_rdiv(K11, K21) + _rdiv(K21, K11)))
        rows.append(dict(snr_db=str(snr), offdiag=float(off), max_rel_pole_diff=diff(dg, ev),
                         err_published=errs(dg, lam_t), err_canonical=errs(ev, lam_t)))
out['sec6_2'] = rows

Af = [9, 10, 5, 10, 8, 5, 6, 8, 10, 10, 5, 10, 10, 7, 9, 5]
SNR = [64, 67, 39, 67, 57, 38, 45, 54, 69, 69, 40, 69, 69, 52, 63, 40]
Te = [254, 277, 81, 279, 208, 74, 119, 187, 290, 292, 89, 293, 290, 171, 250, 85]
FS = [111, 100, 200, 100, 125, 200, 166, 125, 100, 100, 200, 100, 100, 142, 111, 200]
data = []
for j in range(16):
    Y, dY = sim(1/FS[j], Te[j], Af[j], np.random.default_rng(100 + j))
    data.append((awgn_db(Y, SNR[j], rng), awgn_db(dY, SNR[j], rng)))
rows = []
for N in [1, 2, 4, 8, 16]:
    Cq = np.zeros((2*n, 2*n))
    for Y, dY in data[:N]:
        K11, K21 = Y.T @ Y, dY.T @ Y
        Cq += 0.5*(_rdiv(K11, K21) + _rdiv(K21, K11))
    Y = np.vstack([x[0] for x in data[:N]]); dY = np.vstack([x[1] for x in data[:N]])
    dg, ev, off = readouts(Y, dY, Cq/N)
    rows.append(dict(N=N, offdiag=float(off), max_rel_pole_diff=diff(dg, ev),
                     err_published=errs(dg, lam_t), err_canonical=errs(ev, lam_t)))
out['sec6_3'] = rows

M3, C3, K3 = sy.paper_chain3()
l3, _ = sy.true_modes(M3, C3, K3)
rows = []
for dt in [0.001, 0.01, 0.05, 0.2]:
    t = np.arange(0, 1.48, dt)
    d, v, a = sy.simulate_linear(M3, C3, K3, np.r_[1., 1, 1, 0, 0, 0], t)
    Y, dY = np.hstack([d, v]), np.hstack([v, a])
    if len(t) < 7:
        continue
    dg, ev, off = readouts(Y, dY, balanced_quotient(Y, dY))
    rows.append(dict(dt=dt, offdiag=float(off), max_rel_pole_diff=diff(dg, ev),
                     err_published=errs(dg, l3), err_canonical=errs(ev, l3)))
out['sec6_4'] = rows

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_replication.json'), 'w') as f:
    json.dump(out, f, indent=1)
for k, rows in out.items():
    print(k)
    for r in rows:
        print('  ', {kk: (np.round(vv, 6).tolist() if not isinstance(vv, str) else vv) for kk, vv in r.items()})
