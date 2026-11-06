"""Regression tests for release 1.1.2.  Run:  python -m pytest -q

Three groups.

1. DEFAULTS ARE 1.1.1.  Every default call of track, ncvd, ncvd_pair, closure,
   robustness_basis, deflate and escape is run on the synthetic records under
   the frozen v1.1.1 package (tests/_v1_1_1, checked against the tag's git
   blobs) and under this version, and must agree to 1e-12 relative.  The
   witness tests pin the four 1.1.1 defects on the default path, so a silent
   change of a default fails here.
2. OPT-IN FIXES.  The same records with the new keywords (rank='auto',
   seed='energy', families=True, matching='identity', max_cost, closure
   strict=True) and the new functions (numerical_rank, windowed,
   family_closure).  All of these fail on 1.1.1.
3. INTEGRITY.  The vendored copy is the tag; the robustness-matrix and
   reduction-basis functions are byte-identical to 1.1.1.
"""
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import ncvd as nc                                   # noqa: E402
import ncvd.core as ncore                           # noqa: E402
from ncvd import rom                                # noqa: E402
import _records as R                                # noqa: E402

V111_DIR = os.path.join(HERE, '_v1_1_1', 'ncvd')
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RTOL = 1e-12
TW = 1000/R.FS                                      # window duration of the 6-window layouts


def _load_v111():
    if 'ncvd_v111' in sys.modules:
        return sys.modules['ncvd_v111']
    spec = importlib.util.spec_from_file_location(
        'ncvd_v111', os.path.join(V111_DIR, '__init__.py'),
        submodule_search_locations=[V111_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ncvd_v111'] = mod
    spec.loader.exec_module(mod)
    return mod


def _blob(path):
    data = open(path, 'rb').read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return f(*a, **k)


def _alt_quotient(Y1, Y2):
    """One term of the balanced quotient. Equally valid on full-rank linear data
    (every invertible V gives the same poles), different from the default Q."""
    return ncore._rdiv(Y2.conj().T @ Y1, Y1.conj().T @ Y1)


# ------------------------------------------------------------------ records
@pytest.fixture(scope='module')
def v111():
    return _load_v111()


@pytest.fixture(scope='module')
def recA():
    return R.slowfast()


@pytest.fixture(scope='module')
def recB():
    return R.switching()


@pytest.fixture(scope='module')
def recB2():
    return R.switching(swap=True)


@pytest.fixture(scope='module')
def recC():
    return R.duffing_single()


@pytest.fixture(scope='module')
def recD():
    return R.rank_deficient()


# ------------------------------------------------------------------ comparison
def assert_same(a, b, where='result'):
    """Recursive agreement of two outputs to RTOL relative (per array)."""
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), where
        assert set(a) == set(b), f'{where}: keys differ {set(a) ^ set(b)}'
        for k in a:
            assert_same(a[k], b[k], f'{where}.{k}')
        return
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{where}[{i}]')
        return
    if a is None or b is None or isinstance(a, (str, bool)):
        assert a == b, f'{where}: {a!r} != {b!r}'
        return
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape, f'{where}: shape {x.shape} != {y.shape}'
    if x.size == 0:
        return
    nx, ny = np.isnan(x), np.isnan(y)
    assert np.array_equal(nx, ny), f'{where}: NaN pattern differs'
    fin = ~nx
    if not fin.any():
        return
    scale = max(float(np.max(np.abs(y[fin]))), 1e-300)
    err = float(np.max(np.abs(x[fin] - y[fin])))
    assert err <= RTOL*scale, f'{where}: max diff {err:.3e} > {RTOL:g} x {scale:.3e}'


def _shapes_C(mod, recC):
    M, d, v, a = recC
    tr = _quiet(mod.track, d, v, a, fs=R.FS, n_windows=10, n_track=1, M=M)
    return tr['shape'][:, 0, :]


# Each entry: id -> function(module, records) returning the output of DEFAULT calls
# (only arguments that existed in 1.1.1 are passed).
DEFAULT_CALLS = {
    'track_A_n1_M_uncentered': lambda m, r: m.track(*r['A'][1:4], fs=R.FS, n_windows=6, n_track=1,
                                                    M=r['A'][0], center=False),
    'track_A_all': lambda m, r: m.track(*r['A'][1:4], fs=R.FS, n_windows=6),
    'track_B_n3': lambda m, r: m.track(*r['B'][:3], fs=R.FS, n_windows=6, n_track=3),
    'track_B_overlap': lambda m, r: m.track(*r['B'][:3], fs=R.FS, length=1500, step=750),
    'track_B2_n3': lambda m, r: m.track(*r['B2'][:3], fs=R.FS, n_windows=6, n_track=3),
    'track_C_n1_M': lambda m, r: m.track(*r['C'][1:], fs=R.FS, n_windows=10, n_track=1, M=r['C'][0]),
    'track_C_rank4': lambda m, r: m.track(*r['C'][1:], fs=R.FS, n_windows=10, rank=4),
    'track_D_default': lambda m, r: m.track(*r['D'][1:4], fs=R.FS, n_windows=4),
    'ncvd_A': lambda m, r: m.ncvd(*r['A'][1:4]),
    'ncvd_A_center_auto': lambda m, r: m.ncvd(*r['A'][1:4], center=True, time_scale='auto'),
    'ncvd_pair_D': lambda m, r: m.ncvd_pair(np.hstack(r['D'][1:3]), np.hstack(r['D'][2:4])),
    'ncvd_pair_D_rank2': lambda m, r: m.ncvd_pair(np.hstack(r['D'][1:3]), np.hstack(r['D'][2:4]), rank=2),
    'ncvd_pair_B': lambda m, r: m.ncvd_pair(np.hstack(r['B'][:2]), np.hstack(r['B'][1:3])),
    'closure_one_mode': lambda m, r: m.rom.closure(*[x[:, None] for x in m.rom.deflate(_shapes_C(m, r['C']))[:2]], k=1),
    'closure_modal_basis': lambda m, r: m.rom.closure(np.linalg.eigh(2*np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1))[1],
                                                      m.rom.deflate(_shapes_C(m, r['C']))[1][:, None], k=1, tol=0.5),
    'robustness_basis': lambda m, r: m.rom.robustness_basis(_shapes_C(m, r['C'])),
    'robustness_basis_rank2': lambda m, r: m.rom.robustness_basis(_shapes_C(m, r['C']), rank=2),
    'deflate': lambda m, r: m.rom.deflate(_shapes_C(m, r['C'])),
    'deflate_ref': lambda m, r: m.rom.deflate(_shapes_C(m, r['C']), ref=np.ones(3)),
    'escape': lambda m, r: m.rom.escape(np.eye(3)[:, :2], m.rom.deflate(_shapes_C(m, r['C']))[1]),
}


@pytest.fixture(scope='module')
def records(recA, recB, recB2, recC, recD):
    return dict(A=recA, B=recB, B2=recB2, C=recC, D=recD)


# ================================================================== 1. defaults
@pytest.mark.parametrize('call', sorted(DEFAULT_CALLS))
def test_default_call_equals_v111(call, v111, records):
    f = DEFAULT_CALLS[call]
    assert_same(_quiet(f, nc, records), _quiet(f, v111, records), call)


def test_default_track_seeds_on_slow_offset_pole_A(recA):
    """Defect 1 on the default path: n_track=1 seeds on the lowest |Im lam| of
    window 0, a non-oscillatory static-offset pole, then jumps to 2 Omega."""
    M, d, v, a, info = recA
    tr = _quiet(nc.track, d, v, a, fs=R.FS, n_windows=6, n_track=1, M=M, center=False)
    assert tr.wd[0, 0]*TW/(2*np.pi) < 1                 # less than one cycle per window
    assert np.all(tr.wd[1:, 0] > 1.9)                   # 2 Omega harmonic, not the master


def test_default_track_relabels_when_pair_count_drops_B(recB):
    """Defect 2 on the default path: mode 3 is written into track 1 when mode 2
    stops oscillating, and mode 2 is never recovered."""
    d, v, a, info = recB
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3)
    w3 = info['lam'][2].imag
    assert np.allclose(tr.wd[2:, 1], w3, rtol=1e-8)
    assert np.all(np.isnan(tr.wd[2:, 2]))


def test_default_track_relabels_unrelated_pole_B2(recB2):
    d, v, a, info = recB2
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3)
    assert np.allclose(tr.wd[2:4, 1], info['lam_new'].imag, rtol=1e-8)


def test_default_closure_one_mode_returns_none_C(recC):
    """Defect 3 on the default path: a deflation correction is orthogonal to its
    own v0, so with one tracked mode the only candidate P = 1 never closes."""
    v0, c, _ = rom.deflate(_shapes_C(nc, recC))
    assert rom.closure(v0[:, None], c[:, None], k=1) is None


def test_default_rank_keeps_null_directions_D(recD, monkeypatch):
    """Defect 4 on the default path: rank=None keeps all 8 directions of a
    rank-2 record; 3 of the 4 'modes' tracked are null-space eigenvalues, and
    they change with the (arbitrary) balanced quotient."""
    M, d, v, a, lam1 = recD
    with pytest.warns(RuntimeWarning, match='numerical rank 2 < 8'):
        tr = nc.track(d, v, a, fs=R.FS, n_windows=4)
    assert tr.lam.shape[1] == 4
    assert np.sum(np.abs(tr.lam[0]) < 1e-10) == 3
    monkeypatch.setattr(ncore, 'balanced_quotient', _alt_quotient)
    tr2 = _quiet(nc.track, d, v, a, fs=R.FS, n_windows=4)
    null = np.abs(tr.lam[0]) < 1e-10
    assert np.max(np.abs(tr2.lam[0, null] - tr.lam[0, null])/np.abs(tr.lam[0, null])) > 0.1


# ================================================================== 2. opt-in fixes
def test_numerical_rank_rule():
    s = np.array([1.0, 0.5, 1e-6, 1e-7, 1e-14])
    assert nc.numerical_rank(s) == 4                    # relative threshold 1e-12
    assert nc.numerical_rank(s, rtol=1e-8) == 2
    assert nc.numerical_rank(s, gap=1e3) == 2           # first ratio s_i/s_{i+1} > 1e3
    assert nc.numerical_rank(s, gap=1e9) == 4
    assert nc.numerical_rank(np.zeros(3)) == 0


def test_ncvd_rank_auto_is_numerical_rank_D(recD):
    M, d, v, a, lam1 = recD
    with warnings.catch_warnings():
        warnings.simplefilter('error')                  # no rank warning on the opt-in path
        r = nc.ncvd(d, v, a, rank='auto')
    assert r.rank == 2 and r.rank_rule['rtol'] == 1e-12 and r.rank_rule['gap'] is None
    assert np.min(np.abs(r.lam - lam1)) < 1e-10
    with pytest.raises(ValueError):
        nc.ncvd(d, v, a, rank='numerical')


def test_track_rank_auto_D_is_Q_independent(recD, monkeypatch):
    M, d, v, a, lam1 = recD
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tr = nc.track(d, v, a, fs=R.FS, n_windows=4, rank='auto')
    assert tr.rank == 2 and list(tr.rank_windows) == [2, 2, 2, 2]
    assert tr.rank_rule == dict(rule='min over windows', rtol=1e-12, gap=None)
    assert tr.lam.shape[1] == 1
    assert np.max(np.abs(tr.lam[:, 0] - lam1)) < 1e-10
    monkeypatch.setattr(ncore, 'balanced_quotient', _alt_quotient)
    tr2 = nc.track(d, v, a, fs=R.FS, n_windows=4, rank='auto')
    assert_same(tr2.lam, tr.lam, 'lam under a different Q')
    assert np.min([nc.mac(x, y) for x, y in zip(tr2.shape[:, 0], tr.shape[:, 0])]) > 1 - 1e-12


def test_track_rank_auto_C_rank_four(recC):
    M, d, v, a = recC
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tr = nc.track(d, v, a, fs=R.FS, n_windows=10, n_track=1, M=M, rank='auto')
    assert tr.rank == 4 and np.all(tr.rank_windows == 4)
    assert np.all(tr.wd[:, 0] > 0.79)                   # the master, not a null pole


def test_track_rank_explicit_override(recC):
    M, d, v, a = recC
    tr = nc.track(d, v, a, fs=R.FS, n_windows=10, n_track=1, M=M, rank='auto', rank_rtol=1e-3)
    assert tr.rank <= 4 and tr.rank_rule['rtol'] == 1e-3
    tr4 = nc.track(d, v, a, fs=R.FS, n_windows=10, n_track=1, M=M, rank=4)
    tra = nc.track(d, v, a, fs=R.FS, n_windows=10, n_track=1, M=M, rank='auto')
    assert_same(tra.wd, tr4.wd, 'auto rank 4 vs rank=4')


def test_families_never_seed_zero_frequency_pole_A(recA):
    M, d, v, a, info = recA
    for seed in ('lowest', 'energy'):
        tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=1, M=M, center=False,
                      seed=seed, families=True, matching='identity', rank='auto')
        assert np.all(np.abs(tr.wd[:, 0] - info['w_master']) < 0.02), seed
        assert np.all(tr.wd[:, 0]*TW/(2*np.pi) >= 1)
        z = tr.zero
        assert z.lam.shape == (6, 1)
        assert abs(z.lam[0, 0].imag)*TW/(2*np.pi) < 1   # reported with its shape ...
        assert nc.mac(z.shape[0, 0], info['offset_shape']) > 0.9
        assert tr.family_rule['min_cycles'] == 1.0


def test_energy_seed_without_families_picks_master_A(recA):
    M, d, v, a, info = recA
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=1, M=M, center=False, seed='energy',
                  matching='identity')
    assert abs(tr.wd[0, 0] - info['w_master']) < 0.02
    assert tr.seed == 'energy' and tr.seed_window == 0
    assert 0.5 < tr.seed_score[0] <= 1.0                # share of the window-0 harmonic energy


def test_seed_targets_and_participation_A(recA):
    M, d, v, a, info = recA
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=1, M=M, center=False,
                  seed=[2.0], families=True, matching='identity')
    assert np.all(np.abs(tr.wd[:, 0] - 2.0) < 0.03)     # the 2 Omega harmonic, on request
    tp = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=2, M=M, center=False,
                  seed='participation', families=True, matching='identity')
    assert tp.seed_score.shape == (2,) and np.all(tp.seed_score > 0)
    assert np.all(np.diff(tp.wd[0]) > 0)                 # columns in ascending frequency


def test_identity_matching_carries_gaps_B(recB):
    d, v, a, info = recB
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3, matching='identity')
    w = info['lam'].imag
    assert np.allclose(tr.wd[:, 0], w[0], rtol=1e-8)
    assert np.allclose(tr.wd[[0, 1, 4, 5], 1], w[1], rtol=1e-8)    # mode 2 recovered
    assert np.all(np.isnan(tr.wd[[2, 3], 1])) and np.all(tr.gap[[2, 3], 1])
    assert np.allclose(tr.wd[:, 2], w[2], rtol=1e-8)
    assert not tr.gap[:, [0, 2]].any()
    assert np.all(np.isnan(tr.cost[0]))                              # seed window
    ok = ~tr.gap & (np.arange(6)[:, None] > 0)
    assert np.all(tr.cost[ok] < 1e-8)
    assert list(tr.n_harmonic) == [3, 3, 2, 2, 3, 3]


def test_zero_family_reports_real_poles_B(recB):
    d, v, a, info = recB
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3, matching='identity',
                  families=True, n_zero=2)
    z = tr.zero
    assert z.lam.shape == (6, 2) and tr.zero.seed_window == 2
    got = np.sort(z.lam[2:4].real, axis=1)
    assert np.allclose(got, np.sort(info['real_poles']), atol=1e-8)
    assert np.all(np.isnan(z.lam[[0, 1, 4, 5]])) and np.all(z.gap[[0, 1, 4, 5]])
    assert nc.mac(z.shape[2, 0], info['shapes'][:, 1]) > 1 - 1e-10
    assert np.allclose(tr.wd[[0, 1, 4, 5], 1], info['lam'][1].imag, rtol=1e-8)


def test_identity_matching_gate_B2(recB2):
    d, v, a, info = recB2
    tr = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3, matching='identity')
    assert np.allclose(tr.wd[2:4, 1], info['lam_new'].imag, rtol=1e-8)   # no gate: assigned
    assert np.all(tr.cost[2:4, 1] > 1.0)                                 # ... at a high cost
    tg = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=3, matching='identity', max_cost=0.5)
    assert np.all(np.isnan(tg.wd[2:4, 1])) and np.all(tg.gap[2:4, 1])
    assert np.all(tg.cost[2:4, 1] > 0.5)                                 # rejected cost kept
    assert np.allclose(tg.wd[[0, 1, 4, 5], 1], info['lam'][1].imag, rtol=1e-8)


def test_closure_strict_C(recC):
    v0, c, _ = rom.deflate(_shapes_C(nc, recC))
    with pytest.raises(ValueError, match='one tracked mode'):
        rom.closure(v0[:, None], c[:, None], k=1, strict=True)
    Phi = np.linalg.eigh(2*np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1))[1]
    P = rom.closure(Phi, c[:, None], k=1, tol=0.5)
    assert P is not None and rom.closure(Phi, c[:, None], k=1, tol=0.5, strict=True) == P
    with pytest.raises(ValueError, match='not met'):
        rom.closure(Phi, c[:, None], k=1, tol=1e-9, p_max=1, strict=True)
    with pytest.raises(ValueError, match='columns'):
        rom.closure(Phi, c[:, None], k=2, strict=True)


def test_family_closure_C(recC):
    sh = _shapes_C(nc, recC)
    _, _, mag = rom.deflate(sh)
    P, sv = rom.family_closure(sh, tol=0.5*mag)
    assert P >= 2 and abs(sv[0] - mag) < 1e-14
    assert rom.family_closure(sh, tol=2*mag)[0] == 1
    # defining property: the family lies within tol of span(v0, first P-1 directions)
    v0, _, _ = rom.deflate(sh)
    S = sh.T
    Rr = S - np.outer(v0, v0 @ S)
    U = np.linalg.svd(Rr, full_matrices=False)[0]
    B = np.column_stack([v0, U[:, :P - 1]])
    rest = S - B @ np.linalg.lstsq(B, S, rcond=None)[0]
    s1 = np.linalg.svd(S, compute_uv=False)[0]
    assert np.linalg.svd(rest, compute_uv=False)[0] < 0.5*mag*s1


def test_windowed_is_track_with_1_1_2_protocol(recA):
    M, d, v, a, info = recA
    w = nc.windowed(d, v, a, R.FS, n_windows=6, n_track=1, M=M)
    t = nc.track(d, v, a, fs=R.FS, n_windows=6, n_track=1, M=M, center=False, rank='auto',
                 seed='energy', matching='identity', families=True, max_cost=0.5)
    assert_same(w, t, 'windowed vs track')
    assert np.all(np.abs(w.wd[:, 0] - info['w_master']) < 0.02)


def test_families_need_sampling_rate(recA):
    M, d, v, a, info = recA
    with pytest.raises(ValueError, match='fs'):
        nc.track(d, v, a, n_windows=6, families=True)


def test_opt_in_values_reject_typos(recB):
    d, v, a, info = recB
    for kw in (dict(seed='largest'), dict(matching='hungarian'), dict(rank='full')):
        with pytest.raises(ValueError):
            nc.track(d, v, a, fs=R.FS, n_windows=6, **kw)


# ================================================================== 3. integrity
def test_vendored_copy_is_the_v111_tag():
    man = json.load(open(os.path.join(HERE, '_v1_1_1', 'MANIFEST.json')))
    files = sorted(f for f in os.listdir(V111_DIR) if f.endswith('.py'))
    assert files == sorted(man['python'])
    for f in files:
        assert _blob(os.path.join(V111_DIR, f)) == man['python'][f], f


def test_basis_functions_untouched(v111):
    """robustness_basis, deflate, escape (and project, augment,
    residual_augmentation) are byte-identical to 1.1.1 in Python and MATLAB."""
    for name in ('robustness_basis', 'deflate', 'escape', 'project', 'augment',
                 'residual_augmentation'):
        assert inspect.getsource(getattr(rom, name)) == inspect.getsource(getattr(v111.rom, name)), name
    man = json.load(open(os.path.join(HERE, '_v1_1_1', 'MANIFEST.json')))
    mdir = os.path.join(REPO, 'matlab', '+ncvd')
    for f, h in man['matlab_untouched'].items():
        assert _blob(os.path.join(mdir, f)) == h, f


def test_matlab_v111_copies_are_the_tag():
    """matlab/tests/v111/<name>_v111.m are the tag's files with only the
    function name changed, so the MATLAB suite can compare defaults live."""
    man = json.load(open(os.path.join(HERE, '_v1_1_1', 'MANIFEST.json')))
    vdir = os.path.join(REPO, 'matlab', 'tests', 'v111')
    for f, h in man['matlab_v111'].items():
        name = f[:-2]
        text = open(os.path.join(vdir, name + '_v111.m'), 'rb').read()
        head, rest = text.split(b'\n', 1)
        assert head.count(name.encode() + b'_v111(') == 1, f
        orig = head.replace(name.encode() + b'_v111(', name.encode() + b'(') + b'\n' + rest
        assert hashlib.sha1(b'blob %d\0' % len(orig) + orig).hexdigest() == h, f
