"""Unit tests for the NCVD toolbox.  Run:  python -m pytest -q"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import ncvd as nc
from ncvd import systems as sy, benchmarks as bm, rom
import _reference_ncvd as ref


def nearest_err(est, true):
    est, true = np.asarray(est), np.asarray(true)
    return max(np.min(np.abs(e - true)) for e in est)


@pytest.fixture(scope='module')
def chain10():
    M, C, K = sy.paper_chain10()
    t = np.arange(0, 60, 0.05)
    x0 = np.random.default_rng(0).standard_normal(20)
    d, v, a = sy.simulate_linear(M, C, K, x0, t)
    lam = np.linalg.eigvals(sy.state_matrix(M, C, K))
    return M, C, K, t, d, v, a, lam


@pytest.fixture(scope='module')
def light4():
    """Lightly damped 4-DOF chain with |lam_2| = 1: balanced map folds."""
    n = 4
    K = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1); K[-1, -1] = 1
    C = 0.002*np.eye(n) + 0.001*K
    M = np.eye(n)
    t = np.arange(0, 200, 0.05)
    x0 = np.random.default_rng(1).standard_normal(2*n)
    d, v, a = sy.simulate_linear(M, C, K, x0, t)
    return M, C, K, d, v, a, np.linalg.eigvals(sy.state_matrix(M, C, K))


def offdiag(S):
    D = np.diag(np.diag(S))
    return np.linalg.norm(S - D)/np.linalg.norm(D)


# ---------------------------------------------------------------- reproduction
@pytest.mark.parametrize('rank', [None, 12])
@pytest.mark.parametrize('legacy', [False, True])
def test_reproduces_reference(chain10, rank, legacy):
    *_, d, v, a, _ = chain10
    r0 = ref.ncvd(d, v, a, rank=rank, legacy_balanced=legacy)
    r1 = nc.ncvd(d, v, a, rank=rank, legacy_balanced=legacy, sort=False)
    assert np.allclose(r0['lam'], r1.lam, atol=1e-10)
    assert np.allclose(r0['Phi'], r1.Phi, atol=1e-8)
    assert np.allclose(r0['U'], r1.U, atol=1e-8)
    assert abs(r0['res'] - r1.res) < 1e-10


# ---------------------------------------------------------------- exactness
def test_exact_identities_and_poles(chain10):
    *_, d, v, a, lam = chain10
    r = nc.ncvd(d, v, a)
    assert r.exact1 < 1e-12 and r.res < 1e-12
    assert nearest_err(r.lam, lam) < 1e-10
    X1 = np.hstack([d, v]); X2 = np.hstack([v, a])
    assert np.allclose(X1, r.U @ r.Phi.conj().T, atol=1e-10)
    assert np.allclose(X2, (r.U*r.lam) @ r.Phi.conj().T, atol=1e-9)


def test_shapes_match_true(chain10):
    M, C, K, *_ = chain10
    *_, d, v, a, lam = chain10
    r = nc.ncvd(d, v, a)
    lt, Et = sy.true_modes(M, C, K)
    for l, e in zip(lt, Et.T):
        i = np.argmin(np.abs(r.lam - l))
        assert nc.mac(r.Psi[:10, i], e) > 1 - 1e-10
        assert np.allclose(r.Psi[10:, i], l*r.Psi[:10, i], atol=1e-8*np.abs(r.Psi[:, i]).max())


def test_equivalence_theorem(chain10):
    *_, d, v, a, _ = chain10
    r = nc.ncvd(d, v, a, sort=False)
    Y1 = np.hstack([d, v]); Y2 = np.hstack([v, a])
    B = np.linalg.lstsq(Y1, Y2, rcond=None)[0]
    E = np.linalg.inv(r.V).conj().T @ r.T
    assert np.allclose(B @ E, E*r.lam, atol=1e-9*np.abs(B).max())
    assert np.allclose(r.U, Y1 @ E, atol=1e-9)
    assert np.allclose(r.Phi, np.linalg.inv(E).conj().T, atol=1e-9)


def test_rank_whitening_keeps_poles(chain10):
    *_, d, v, a, lam = chain10
    assert nearest_err(nc.ncvd(d, v, a, rank=20).lam, lam) < 1e-9


# ---------------------------------------------------------------- Joukowski fold
def test_fold_breaks_legacy_not_canonical(light4):
    M, C, K, d, v, a, lam = light4
    r = nc.ncvd(d, v, a)                 # time unit puts |lam_2| = 1
    assert offdiag(r.S2) > 1e-2          # V does not diagonalise Sigma_2
    assert r.res < 1e-12 and nearest_err(r.lam, lam) < 1e-10
    g = bm.gcvd(d, v, a)
    assert nearest_err(g['lam'], lam) > 1e-3   # published diagonal readout is wrong


def test_auto_time_scale_unfolds(light4):
    M, C, K, d, v, a, lam = light4
    r = nc.ncvd(d, v, a, time_scale='auto')
    assert offdiag(r.S2) < 1e-8 and r.condT < 1 + 1e-6
    assert nearest_err(r.lam, lam) < 1e-10
    assert np.all(np.abs(lam/r.time_scale) >= 2 - 1e-9)


def test_time_scale_invariance(chain10):
    *_, d, v, a, lam = chain10
    for c in [0.1, 1.0, 7.0, 'auto']:
        assert nearest_err(nc.ncvd(d, v, a, time_scale=c).lam, lam) < 1e-10


# ---------------------------------------------------------------- operators
def test_tls_equals_ls_without_noise(chain10):
    *_, d, v, a, lam = chain10
    assert nearest_err(nc.ncvd(d, v, a, operator='tls').lam, lam) < 1e-8


# ---------------------------------------------------------------- kinematics
@pytest.mark.parametrize('method', ['forward', 'backward', 'central', 'central4', 'savgol'])
def test_fd_correction(method):
    M, C, K = sy.chain(3, 1, 1, 0.05)
    fs = 10
    t = np.arange(0, 80, 1/fs)
    d, *_ = sy.simulate_linear(M, C, K, np.random.default_rng(4).standard_normal(6), t)
    lam = np.linalg.eigvals(sy.state_matrix(M, C, K))
    dd, vv, aa, f, _ = nc.state_from_displacement(d, fs, method)
    r = nc.ncvd(dd, vv, aa)
    assert nearest_err(nc.correct_poles(r.lam, f, fs), lam) < 1e-10


# ---------------------------------------------------------------- forcing
def test_known_forcing():
    M, C, K = sy.chain(4, 1, 2, 0.05)
    t = np.arange(0, 100, 0.02)
    f = np.random.default_rng(3).standard_normal((len(t), 4))
    d, v, a = sy.simulate_forced_linear(M, C, K, f, t)
    lam = np.linalg.eigvals(sy.state_matrix(M, C, K))
    r_f = nc.ncvd(d, v, a, forcing=f)
    r_0 = nc.ncvd(d, v, a)
    assert nearest_err(r_f.lam, lam) < 1e-8 and r_f.res < 1e-10
    assert nearest_err(r_0.lam, lam) > 1e-3


# ---------------------------------------------------------------- ensemble
def test_ensemble_mixed_sampling():
    M, C, K = sy.paper_chain10()
    lam = np.linalg.eigvals(sy.state_matrix(M, C, K))
    rng = np.random.default_rng(5)
    recs = []
    for fs, T in [(100, 30), (125, 20), (200, 10)]:
        t = np.arange(0, T, 1/fs)
        recs.append(sy.simulate_linear(M, C, K, rng.standard_normal(20), t))
    r = nc.ncvd_ensemble(recs)
    assert nearest_err(r.lam, lam) < 1e-9
    assert len(r.blocks) == 3


# ---------------------------------------------------------------- benchmarks
def test_benchmarks_clean(chain10):
    *_, t, d, v, a, lam = chain10
    dt = t[1] - t[0]
    X = np.hstack([d, v])
    assert nearest_err(bm.dmd(X, dt)['lam'], lam) < 1e-8
    assert nearest_err(bm.itd(X, dt, rank=20)['lam'], lam) < 1e-6


# ---------------------------------------------------------------- tracking
def test_tracking_linear_constant(chain10):
    *_, d, v, a, lam = chain10
    tr = nc.track(d, v, a, fs=20, n_windows=6, n_track=3)
    assert np.nanmax(tr.res) < 1e-10
    assert np.allclose(tr.wn, tr.wn[0], rtol=1e-8)


def test_tracking_duffing_hardening():
    M, C, K, fnl = sy.duffing_chain(3, k=1.0, c=0.005, k3=0.5)
    t = np.arange(0, 400, 0.05)
    lam0, E0 = sy.true_modes(M, C, K)
    x0 = np.r_[1.2*np.real(E0[:, 0])/np.abs(E0[:, 0]).max(), np.zeros(3)] \
        + np.r_[0.05*np.real(E0[:, 1]), 0.05*np.real(E0[:, 2]), np.zeros(1)][:6]
    d, v, a = sy.simulate_nonlinear(M, C, K, fnl, x0, t)
    tr = nc.track(d, v, a, fs=20, n_windows=10, n_track=1, M=M)
    w = tr.wd[:, 0]
    assert w[0] > w[-1] > 0.99*abs(lam0[0].imag)   # hardening: frequency falls with decay
    assert tr.res[0] > tr.res[-1]                   # nonlinearity measure decays


# ---------------------------------------------------------------- modal helpers
def test_realify_and_mac():
    rng = np.random.default_rng(2)
    v = rng.standard_normal(6)
    w, m = nc.realify(v*np.exp(0.7j))
    assert m > 1 - 1e-12 and nc.mac(w, v) > 1 - 1e-12
    mm, _ = nc.mac_phase_matched(v*np.exp(0.3j), v + 0j)
    assert mm > 1 - 1e-12


def test_rom_helpers():
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(8); v0 /= np.linalg.norm(v0)
    c = rng.standard_normal(8); c -= v0*(v0 @ c); c /= np.linalg.norm(c)
    fam = np.array([v0 + s*c for s in np.linspace(0.3, 0, 12)])
    fam /= np.linalg.norm(fam, axis=1, keepdims=True)
    V0, corr, mag = rom.deflate(fam)
    assert abs(V0 @ v0) > 1 - 1e-12 and abs(corr @ c) > 1 - 1e-9 and mag > 0
    assert rom.escape(np.c_[v0, c], corr)[0] < 1e-9


# ---------------------------------------------------------------- readout & noise operators
def test_forced_quotient_readout_exact_on_noisy_forced_data():
    """Proposition 1(1): the 'forced' quotient diagonalises Sigma_2 for any data."""
    M, C, K = sy.paper_chain10()
    rng = np.random.default_rng(7)
    t = np.arange(0, 60, 0.01)
    d, v, a = sy.simulate_forced_linear(M, C, K, rng.standard_normal((len(t), 10)), t)
    d, v, a = (sy.add_noise(z, 30, rng, db=True) for z in (d, v, a))
    g = bm.gcvd(d, v, a, quotient='forcing')
    assert offdiag(g['S2']) < 1e-9
    r = nc.ncvd(d, v, a)
    assert nearest_err(g['lam'], r.lam) < 1e-9


def test_bcls_reduces_noise_bias():
    M, C, K = sy.paper_chain10()
    lam = sy.true_modes(M, C, K)[0]
    rng = np.random.default_rng(11)
    fs = 200.0
    t = np.arange(0, 150, 1/fs)
    d, v, a = sy.simulate_forced_linear(M, C, K, 10*rng.standard_normal((len(t), 10)), t)
    d, v, a = (sy.add_noise(z, 15, rng, db=True, per_channel=True) for z in (d, v, a))

    def ferr(r):
        le = r.lam[nc.physical_modes(r.lam)]
        return np.median(np.abs(np.abs(le) - np.abs(lam))/np.abs(lam))
    assert ferr(nc.ncvd(d, v, a, operator='bcls', noise_var='auto', fs=fs)) < 0.5*ferr(nc.ncvd(d, v, a))


def test_filter_bands():
    assert abs(nc.deriv_filter('forward').band() - np.pi) < 1e-12
    assert abs(nc.deriv_filter('central').band() - np.pi/2) < 1e-3
    assert nc.deriv_filter('savgol', window=7, polyorder=3).band() < np.pi/2


def test_diag_readout_option_matches_benchmark(chain10):
    *_, d, v, a, _ = chain10
    r = nc.ncvd(d, v, a, readout='diag', sort=False)
    g = bm.gcvd(d, v, a)
    assert nearest_err(r.lam, g['lam']) < 1e-9
