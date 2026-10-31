"""What the NCVD residual says about the nonlinearity.

R = Y2 - U Lam Phi^H = (I - Y1 Y1^+) Y2 is not waste. It is the part of the
operator image that the data span cannot represent, and for a nonlinear system
it is the nonlinear force field projected off that span: measured agreement
with (I - P) F_nl is 4.8e-14 on a 4-DOF chain. It reads

    WHERE  the leading right-singular direction of the dynamic block matches
           M^-1 w of the true element to MAC 1.000000 (grounded spring, link
           spring and dissipative element alike), with rank-1 share 1.000000;
    ORDER  ||R|| ~ a^p with p the order of the nonlinear force -- measured
           3.015 (cubic), 2.001 (quadratic), 0.999 (signal-proportional noise),
           -0.001 (fixed noise floor).  PERTURBATIVE: the exponent converges to
           p as the nonlinear force becomes a small fraction of the linear one;
    TYPE   projected-library attribution separates displacement-driven from
           velocity-driven from non-smooth laws (see `residual_report`).

TWO STRUCTURAL FACTS that bound what R can ever say
---------------------------------------------------
1.  For a [q, qdot] / [qdot, qddot] record the KINEMATIC block of R is ZERO BY
    CONSTRUCTION (~4e-15), because velocity is literally a column of the
    regressor. Only the dynamic block carries information, so report
    ||R_dyn|| / ||qddot||, not ||R|| / ||X2||, whose denominator mixes units.
2.  R is orthogonal to the data span, so <R, qdot> = 0 IDENTICALLY (1.1e-15).
    The residual carries NO NET POWER and cannot by itself tell a conservative
    nonlinearity from a dissipative one. That distinction lives in the two
    channels of lambda (see `channel_slopes`) and in library attribution.

WINDOW LENGTH
-------------
Windows shorter than about three cycles of the slowest tracked mode let the
state span ABSORB the superharmonics. On one cubic record, 0.13 cycles/window
gave relative residual 4.0e-7 and amplitude exponent 1.29, while 10
cycles/window gave 1.4e-2 and 3.01 on the SAME data. Pole tracking is
unaffected; only residual diagnostics need the long window. Check with
`window_cycles` before trusting any residual number.

"""
import warnings
import numpy as np

__all__ = ['residual_report', 'residual_direction', 'amplitude_exponent',
           'channel_slopes', 'window_cycles']


def residual_direction(R, n_space=None, weight=None):
    """Leading spatial direction of the residual and its rank-1 share.

    Returns (direction, rank1_share, singular_values). `weight` (the mass
    matrix) maps the acceleration-space direction back to a force direction:
    w ~ M @ direction.
    """
    R = np.asarray(R).real
    Rd = R if n_space is None else R[:, n_space:]
    U, sv, Wh = np.linalg.svd(Rd, full_matrices=False)
    d = Wh[0].real
    d = d/max(np.linalg.norm(d), 1e-300)
    if weight is not None:
        d = np.asarray(weight, float) @ d
        d = d/max(np.linalg.norm(d), 1e-300)
    return d, float(sv[0]**2/max(np.sum(sv**2), 1e-300)), sv


def residual_report(Y1, Y2, rec=None, library=None, lib_groups=None, n_space=None):
    """Full residual diagnostic.

    library : (m, J) candidate nonlinear regressors evaluated along the record.
    lib_groups : {name: [column indices]} -- explained fraction per group, which
        is what names the TYPE of the nonlinearity. A useful default grouping is
        displacement polynomials / velocity polynomials / sign(velocity): a
        cubic spring gives 1.000 / 0.008 / 0.000, quadratic drag and cubic
        damping give velocity 1.000, and Coulomb friction gives sign(v) 0.9998
        with the polynomials failing -- non-smoothness is itself diagnosable.
    n_space : number of physical stations when Y1 = [q, qdot].

    The library is projected off the data span before regression, which is what
    makes it well conditioned (848 against 5.9e5 on the reference case) and
    necessary, since R is orthogonal to that span by construction.
    """
    from .operators import ncvd_snapshot, state_completeness
    Y1 = np.asarray(Y1); Y2 = np.asarray(Y2)
    rec = rec if rec is not None else ncvd_snapshot(Y1, Y2)
    R = np.asarray(rec['R']).real
    Rd = R if n_space is None else R[:, n_space:]
    out = dict(res=float(rec['res']), floor=state_completeness(Y1, Y2))
    if n_space is not None:
        out['kinematic_block'] = float(np.linalg.norm(R[:, :n_space])
                                       / max(np.linalg.norm(Y2[:, :n_space]), 1e-300))
        out['dynamic_block'] = float(np.linalg.norm(Rd)
                                     / max(np.linalg.norm(Y2[:, n_space:]), 1e-300))
    d, share, sv = residual_direction(R, n_space)
    out['direction'], out['rank1_share'], out['singular_values'] = d, share, sv
    # <R, Y1> = 0 by the normal equations. The normalised ratio is only
    # meaningful when R is well above machine precision; on an exactly linear
    # record it is rounding noise on a machine-zero residual, so report None.
    out['power'] = (None if out['res'] < 1e-12 else
                    float(abs(np.sum(R*Y1[:, :R.shape[1]]))
                          / max(np.linalg.norm(R)*np.linalg.norm(Y1), 1e-300)))
    if library is not None:
        L = np.atleast_2d(np.asarray(library, float))
        Lp = L - Y1 @ np.linalg.lstsq(Y1, L, rcond=None)[0]
        def frac(cols):
            A = Lp[:, cols]
            fit = A @ np.linalg.lstsq(A, Rd, rcond=None)[0]
            return float(1 - np.linalg.norm(Rd - fit)**2/max(np.linalg.norm(Rd)**2, 1e-300))
        out['explained_all'] = frac(list(range(L.shape[1])))
        out['coeffs'] = np.linalg.lstsq(Lp, Rd, rcond=None)[0]
        out['cond_projected'] = float(np.linalg.cond(Lp))
        if lib_groups:
            out['explained'] = {k: frac(c) for k, c in lib_groups.items()}
    return out


def amplitude_exponent(amplitudes, residual_norms):
    """Slope of log||R|| against log(amplitude) = order of the nonlinear force.

    Use INDEPENDENT RELEASES at different amplitudes, not windows of one decay:
    for a dissipative nonlinearity the decay rate is itself amplitude dependent,
    which biases a single-record estimate. Measure the amplitude of the variable
    the force acts on (displacement for a spring, velocity for a damper).
    """
    a = np.asarray(amplitudes, float); r = np.asarray(residual_norms, float)
    m = (a > 0) & (r > 0)
    if m.sum() < 3:
        raise ValueError('need at least 3 positive (amplitude, residual) pairs')
    p = np.polyfit(np.log(a[m]), np.log(r[m]), 1)
    corr = float(np.corrcoef(np.log(a[m]), np.log(r[m]))[0, 1])
    return float(p[0]), corr


def channel_slopes(amplitude, zeta, freq, zeta_lin=None, freq_lin=None, tol=0.02):
    """Amplitude slopes of the damping and frequency channels of lambda.

    A CONSERVATIVE nonlinearity moves Im(lambda) and leaves Re(lambda) alone; a
    DISSIPATIVE one does the reverse, with a describing-function exponent that
    names the law. Measured against predictions on a 4-DOF chain:

        cubic stiffness q^3   damping flat (<0.15 %), frequency slope +2.000
        quadratic drag v|v|   damping +1.040   (predicted +1)
        cubic damping  v^3    damping +2.114   (predicted +2)
        Coulomb  sgn(v)       damping -0.867   (predicted -1)

    The discriminant is the RATIO of relative damping change to relative
    frequency change: about 2 for conservative, 1e3 - 1e4 for dissipative.
    Returns dict(zeta_slope, freq_slope, ratio, zeta_flat, freq_flat).
    """
    a = np.asarray(amplitude, float)
    z = np.asarray(zeta, float); f = np.asarray(freq, float)
    zl = float(np.median(z)) if zeta_lin is None else float(zeta_lin)
    fl = float(np.median(f)) if freq_lin is None else float(freq_lin)
    dz, df = z - zl, np.abs(f - fl)
    out = {}
    okz = dz > tol*abs(zl)
    out['zeta_slope'] = (float(np.polyfit(np.log(a[okz]), np.log(dz[okz]), 1)[0])
                         if okz.sum() > 3 else None)
    okf = df > 1e-4*abs(fl)
    out['freq_slope'] = (float(np.polyfit(np.log(a[okf]), np.log(df[okf]), 1)[0])
                         if okf.sum() > 3 else None)
    rz = float(np.max(np.abs(dz))/max(abs(zl), 1e-300))
    rf = float(np.max(df)/max(abs(fl), 1e-300))
    out['rel_zeta_change'], out['rel_freq_change'] = rz, rf
    out['ratio'] = rz/max(rf, 1e-300)
    out['zeta_flat'] = out['zeta_slope'] is None
    out['freq_flat'] = out['freq_slope'] is None
    out['verdict'] = ('conservative' if out['ratio'] < 10
                      else 'dissipative' if out['ratio'] > 100 else 'mixed')
    return out


def window_cycles(n_samples, fs, n_windows, f_slowest, min_cycles=3.0, warn=True):
    """Cycles of the slowest tracked mode per window; warns below min_cycles.

    Residual-based diagnostics (size, order, type, and residual-driven ROM
    basis augmentation) are invalid below about three cycles per window.
    """
    cyc = float(n_samples/fs/n_windows*f_slowest)
    if warn and cyc < min_cycles:
        warnings.warn(f'{cyc:.2f} cycles of the slowest tracked mode per window '
                      f'(< {min_cycles}); the state span will absorb the superharmonics and '
                      'the residual will be understated. Pole tracking is unaffected.',
                      RuntimeWarning, stacklevel=2)
    return cyc
