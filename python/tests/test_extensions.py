"""Tests for the operator/refinement/diagnostics extensions (toolbox 1.1)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ncvd as nc
from ncvd import operators as op, refine as rf, diagnostics as dg


# --------------------------------------------------------------- fixtures
def chain(n=4, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([1.0, 1.3, 0.8, 1.1])[:n]
    M = np.diag(m)
    k = np.array([1200., 900., 1500., 700., 1000.])[:n+1]
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k[i] + k[i+1]
        if i+1 < n:
            K[i, i+1] = K[i+1, i] = -k[i+1]
    C = 1.2e-4*K + 0.02*M
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    lam, Th = np.linalg.eig(A)
    t = np.linspace(0, 4.0, 4000)
    c0 = np.linalg.solve(Th, np.concatenate([rng.standard_normal(n)*0.01, np.zeros(n)]))
    X = (Th @ (np.exp(np.outer(lam, t))*c0[:, None])).T.real
    return t, X[:, :n], X[:, n:], (X @ A.T)[:, n:], lam, M, K


def field(m=24, N=200, K_=4, seed=1, uneven=True):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, N)
    kw = np.arange(1, K_+1)*np.pi
    PHI = np.sin(np.outer(xs, kw))
    wn = 2.0*kw**2
    lam = -0.005*wn + 1j*wn
    amp = rng.standard_normal(K_) + 1j*rng.standard_normal(K_)
    t = np.sort(rng.random(m))*2.0 if uneven else np.linspace(0, 2.0, m)

    def U(o=0, dx=0):
        E = np.exp(np.outer(t, lam))*(lam**o)[None, :]*amp[None, :]
        return 2*np.real(E) @ (PHI*((-kw**2)**(dx//2))[None, :]).T
    return t, U, kw, lam


def _match(ref, cand):
    from scipy.optimize import linear_sum_assignment
    ref = np.atleast_1d(ref); cand = np.atleast_1d(cand)
    r, c = linear_sum_assignment(np.abs(ref[:, None] - cand[None, :]))
    return cand[c]


# --------------------------------------------------------------- operators
def test_snapshot_form_matches_core_on_full_rank():
    _, d, v, a, lam, _, _ = chain()
    r0 = nc.ncvd(d, v, a)
    r1 = op.from_state(d, v, a)
    # compare by nearest-pole assignment: sort_complex tie-breaks conjugate pairs
    # with bitwise-equal real parts inconsistently between the two orderings
    assert np.max(np.abs(_match(r0.lam, r1.lam) - r0.lam)) < 1e-9
    for i in range(r1.Phi.shape[1]):
        j = int(np.argmin(np.abs(r0.lam - r1.lam[i])))
        mac = (abs(np.vdot(r0.Phi[:, j], r1.Phi[:, i]))**2
               / (np.linalg.norm(r0.Phi[:, j])**2*np.linalg.norm(r1.Phi[:, i])**2))
        assert mac > 1 - 1e-10
    assert r1.res < 1e-10 and r1.exact1 < 1e-10


def test_state_completeness_detects_missing_coordinate():
    _, d, v, a, _, _, _ = chain()
    assert op.state_completeness(np.hstack([d, v]), np.hstack([v, a])) < 1e-10
    assert op.state_completeness(d, v) > 0.1          # displacement alone is not a state


def test_delay_form_recovers_continuous_poles():
    t, d, v, a, lam, _, _ = chain()
    dt = t[1] - t[0]
    out = op.from_delays(np.hstack([d, v]), n_delays=1, dt=dt)
    ref = np.sort_complex(lam[lam.imag > 0])
    got = _match(ref, out['lam_c'])
    assert np.max(np.abs(got - ref)/np.abs(ref)) < 1e-8


def test_aliasing_margin_warns_past_nyquist():
    _, _, _, _, lam, _, _ = chain()
    assert op.aliasing_margin(1e-4, lam, warn=False) < 1.0
    with pytest.warns(RuntimeWarning, match='Nyquist'):
        op.aliasing_margin(1.0, lam)


def test_spatial_form_needs_no_time_order():
    _, U, kw, _ = field()
    r = op.from_spatial(U(), U(dx=2))
    k = np.sort(np.sqrt(np.clip(-r.lam.real, 0, None)))[-len(kw):]
    assert np.max(np.abs(k - kw)/kw) < 1e-8
    p = np.random.default_rng(0).permutation(U().shape[0])
    r2 = op.from_spatial(U()[p], U(dx=2)[p])
    k2 = np.sort(np.sqrt(np.clip(-r2.lam.real, 0, None)))[-len(kw):]
    assert np.allclose(k, k2, atol=1e-9)


def test_spatial_form_needs_only_K_snapshots():
    _, U, kw, _ = field(m=len(np.arange(4)))      # m = K = 4
    r = op.from_spatial(U(), U(dx=2))
    k = np.sort(np.sqrt(np.clip(-r.lam.real, 0, None)))[-len(kw):]
    assert np.max(np.abs(k - kw)/kw) < 1e-8


def test_snapshot_weights_do_not_move_exact_poles():
    _, U, _, lam = field()
    Y1 = np.hstack([U(), U(1)]); Y2 = np.hstack([U(1), U(2)])
    ref = np.concatenate([lam, lam.conj()])
    base = _match(ref, op.ncvd_snapshot(Y1, Y2).lam)
    N = U().shape[1]
    w = np.concatenate([np.ones(N), np.full(N, 1e-4)])
    got = _match(ref, op.ncvd_snapshot(Y1, Y2, weights=w).lam)
    assert np.max(np.abs(got - ref)/np.abs(ref)) < 1e-8
    assert np.max(np.abs(base - ref)/np.abs(ref)) < 1e-8


def test_rank_deficient_record_warns():
    _, d, v, a, _, _, _ = chain()
    d2 = np.hstack([d, d])                        # duplicated stations -> rank deficient
    # the warning fires first; on an EXACTLY singular record the balanced quotient
    # then fails outright, which is the behaviour the warning is there to pre-empt
    with pytest.warns(RuntimeWarning, match='numerical rank'):
        with pytest.raises(np.linalg.LinAlgError):
            nc.ncvd(d2, np.hstack([v, v]), np.hstack([a, a]))
    # the snapshot form handles the same record without complaint
    r = op.from_state(d2, np.hstack([v, v]), np.hstack([a, a]))
    assert r['rank'] == 2*d.shape[1]
    assert r['res'] < 1e-9


# --------------------------------------------------------------- refinement
def test_varpro_beats_canonical_under_noise():
    t, d, v, a, lam, _, _ = chain()
    rng = np.random.default_rng(5)
    snr = 40.0
    dn = d + rng.standard_normal(d.shape)*np.std(d)*10**(-snr/20)
    vn = v + rng.standard_normal(v.shape)*np.std(v)*10**(-snr/20)
    an = a + rng.standard_normal(a.shape)*np.std(a)*10**(-snr/20)
    ref = lam[lam.imag > 0]
    ref = ref[np.argsort(ref.imag)]
    rec = nc.ncvd(dn, vn, an)
    z = lambda L: -L.real/np.abs(L)
    e0 = np.mean(np.abs(z(_match(ref, rec.lam)) - z(ref))/z(ref))
    out = rf.refine_poles(rec, dn, t)
    e1 = np.mean(np.abs(z(_match(ref, out['lam'])) - z(ref))/z(ref))
    assert e1 < e0, (e0, e1)
    assert out['success']


# --------------------------------------------------------------- diagnostics
def test_residual_blocks_and_direction_on_localised_nonlinearity():
    from scipy.integrate import solve_ivp
    n = 4
    _, _, _, _, _, M, K = chain(n)
    C = 1.2e-4*K + 0.02*M
    Minv = np.linalg.inv(M)
    w = np.array([0, 0, 1.0, 0]); k3 = 2.0e4

    def rhs(tt, x):
        q, vv = x[:n], x[n:]
        return np.concatenate([vv, -Minv @ (K @ q + C @ vv + w*k3*q[2]**3)])
    T = 2.0
    te = np.linspace(0, T, 6000)
    x0 = np.concatenate([np.array([.05, .04, .055, .045]), np.zeros(n)])
    X = solve_ivp(rhs, (0, T), x0, t_eval=te, rtol=1e-11, atol=1e-13,
                  method='DOP853').y.T
    d, v = X[:, :n], X[:, n:]
    a = np.array([rhs(0, x) for x in X])[:, n:]
    Y1, Y2 = np.hstack([d, v]), np.hstack([v, a])
    lib = np.stack([d[:, i]**3 for i in range(n)] + [v[:, i]**3 for i in range(n)], 1)
    grp = {'displacement': list(range(n)), 'velocity': list(range(n, 2*n))}
    rep = dg.residual_report(Y1, Y2, library=lib, lib_groups=grp, n_space=n)
    assert rep['kinematic_block'] < 1e-10            # structurally zero
    assert rep['dynamic_block'] > 1e-4
    assert rep['rank1_share'] > 0.999
    assert rep['power'] < 1e-10                      # R carries no net power
    true_dir = Minv @ w; true_dir /= np.linalg.norm(true_dir)
    assert (rep['direction'] @ true_dir)**2 > 0.999
    assert rep['explained']['displacement'] > 0.99
    assert rep['explained']['velocity'] < 0.2


def test_amplitude_exponent_reads_the_order():
    a = np.logspace(-3, -1, 8)
    assert abs(dg.amplitude_exponent(a, 7.0*a**3)[0] - 3.0) < 1e-8
    assert abs(dg.amplitude_exponent(a, 2.0*a**2)[0] - 2.0) < 1e-8


def test_channel_slopes_separates_conservative_from_dissipative():
    a = np.logspace(-3, -1.5, 12)
    cons = dg.channel_slopes(a, np.full_like(a, 0.01), 10.0 + 300.0*a**2,
                             zeta_lin=0.01, freq_lin=10.0)
    assert cons['verdict'] == 'conservative'
    assert abs(cons['freq_slope'] - 2.0) < 1e-6
    diss = dg.channel_slopes(a, 0.01 + 5.0*a, np.full_like(a, 10.0),
                            zeta_lin=0.01, freq_lin=10.0)
    assert diss['verdict'] == 'dissipative'
    assert abs(diss['zeta_slope'] - 1.0) < 0.05


def test_window_cycles_warns_when_too_short():
    assert dg.window_cycles(40000, 4000.0, 10, 3.0, warn=False) == pytest.approx(3.0)
    with pytest.warns(RuntimeWarning, match='cycles'):
        dg.window_cycles(40000, 4000.0, 120, 3.0)


# --------------------------------------------------------------- rom
def test_residual_augmentation_is_orthonormal_and_static():
    rng = np.random.default_rng(2)
    n = 6
    K = np.diag(np.arange(1, n+1, dtype=float))*1e3
    M = np.eye(n)
    w = np.zeros(n); w[3] = 1.0
    R = np.outer(rng.standard_normal(200), w)             # rank-1 residual along w
    D, sh = nc.rom.residual_augmentation(R, K, M=M, n_dir=1)
    assert D.shape == (n, 1)
    assert abs(np.linalg.norm(D[:, 0]) - 1) < 1e-12
    want = np.linalg.solve(K, w); want /= np.linalg.norm(want)
    assert abs(abs(D[:, 0] @ want) - 1) < 1e-8
    assert sh[0] > 0.999


def test_project_warns_on_non_orthonormal_basis():
    n = 4
    M = np.eye(n); K = np.eye(n)*10.0
    P = np.ones((n, 2)); P[:, 1] = np.arange(n)
    with pytest.warns(RuntimeWarning, match='orthonormal'):
        nc.rom.project(M, K, lambda q: np.zeros(n), P)


# --------------------------------------------------------------- forced input
def test_snapshot_forced_recovers_poles():
    """Mirrors test_known_forcing for the CANONICAL direct route."""
    from ncvd import systems as sy
    M, C, K = sy.chain(4, 1, 2, 0.05)
    t = np.arange(0, 100, 0.02)
    f = np.random.default_rng(3).standard_normal((len(t), 4))
    d, v, a = sy.simulate_forced_linear(M, C, K, f, t)
    lam = np.linalg.eigvals(sy.state_matrix(M, C, K))
    rf = op.from_state(d, v, a, forcing=f)
    r0 = op.from_state(d, v, a)
    e_f = np.max(np.abs(_match(lam, rf.lam) - lam))
    e_0 = np.max(np.abs(_match(lam, r0.lam) - lam))
    assert e_f < 1e-8, e_f
    assert rf.res < 1e-10, rf.res
    assert e_0 > 1e-3, e_0                      # ignoring the input is wrong
    assert rf.G is not None and rf.G.shape[0] == f.shape[1]


def test_snapshot_forced_U_is_on_the_unprojected_response():
    """U must be Y1 E, not the forcing-removed coordinates."""
    from ncvd import systems as sy
    M, C, K = sy.chain(4, 1, 2, 0.05)
    t = np.arange(0, 60, 0.02)
    f = np.random.default_rng(5).standard_normal((len(t), 4))
    d, v, a = sy.simulate_forced_linear(M, C, K, f, t)
    rf = op.from_state(d, v, a, forcing=f)
    Y1 = np.hstack([d, v])
    assert np.allclose(rf.U, Y1 @ rf.E, atol=1e-8*max(1.0, np.abs(rf.U).max()))
    # and the first identity still holds on the ACTUAL response
    assert rf.exact1 < 1e-10


def test_snapshot_forced_residual_carries_the_input_term():
    """Without the -F G term the input reappears as apparent nonlinearity and
    the diagnostics would report a spurious nonlinear force."""
    from ncvd import systems as sy
    M, C, K = sy.chain(4, 1, 2, 0.05)
    t = np.arange(0, 60, 0.02)
    f = np.random.default_rng(7).standard_normal((len(t), 4))
    d, v, a = sy.simulate_forced_linear(M, C, K, f, t)
    rf = op.from_state(d, v, a, forcing=f)
    Y1, Y2 = np.hstack([d, v]), np.hstack([v, a])
    R_with = np.asarray(rf.R).real
    R_without = Y2 - (np.asarray(rf.U)*rf.lam) @ np.asarray(rf.Phi).conj().T
    assert np.linalg.norm(R_with) < 1e-8*np.linalg.norm(Y2)
    assert np.linalg.norm(R_without.real) > 1e-3*np.linalg.norm(Y2)
    # the diagnostic that would be poisoned: residual direction is meaningless
    # noise when the input term is kept, and rank-1 along the input when dropped
    rep = dg.residual_report(Y1, Y2, rec=rf, n_space=4)
    assert rep['dynamic_block'] < 1e-7, rep['dynamic_block']


def test_snapshot_forcing_matches_core_route():
    from ncvd import systems as sy
    M, C, K = sy.chain(4, 1, 2, 0.05)
    t = np.arange(0, 60, 0.02)
    f = np.random.default_rng(11).standard_normal((len(t), 4))
    d, v, a = sy.simulate_forced_linear(M, C, K, f, t)
    rs = op.from_state(d, v, a, forcing=f)
    rc = nc.ncvd(d, v, a, forcing=f)
    assert np.max(np.abs(_match(rc.lam, rs.lam) - rc.lam)) < 1e-8
