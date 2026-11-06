# NCVD toolbox changelog

## 1.1.2 -- 2026-11-06

**The published procedure is unchanged.** Every default call returns what 1.1.1 returns;
each remedy below is opt-in (a new keyword argument or a new function, off by default).
`python/tests/test_release_112.py` runs 20 default calls of `track`, `ncvd`, `ncvd_pair`,
`closure`, `robustness_basis`, `deflate` and `escape` on five synthetic records under this
version and under a byte-exact copy of the v1.1.1 package (`python/tests/_v1_1_1/`,
checked against the tag's git blobs) and requires agreement to 1e-12 relative. They agree
exactly, because the default path of `track` is the 1.1.1 source (`_track_v111`, checked
line for line). `matlab/tests/test_ncvd.m/testDefaultsMatchV111` does the same against the
tag's MATLAB functions kept as `matlab/tests/v111/*_v111.m`. Only the version strings
(`__version__`, `cite()`, `Contents.m`, `CITATION.cff`) change on a default call.

### Four weaknesses of the 1.1.1 tracker, kept on the default path, with opt-in remedies

Each is reproduced by a synthetic record in `python/tests/_records.py` and pinned by a
witness test on the default path.

1. **Seeding.** `track(n_track=k)` takes the k lowest-frequency poles with Im(lam) > 0 of
   window 0. With `center=False` the decaying static offset of a quadratic nonlinearity is
   such a pole. On record A (quadratic-cubic slow-fast decay) the default seeds on
   0.0385 rad/s (zeta 0.87, 0.31 cycles per window) and then follows the 2 Omega harmonic
   (2.009 rad/s), never the master (1.007 rad/s). On a rank-deficient window the lowest pole
   can be a null-space eigenvalue of modulus 1e-16.
   Remedies: `seed='energy'` (largest modal energy of the pole's own velocity
   contribution), `seed='participation'`, or a list of target frequencies; and
   `families=True` (needs `fs`), which tracks harmonic poles (at least `min_cycles` = 1
   damped cycle per window) separately from the zero-frequency family (real poles and slower
   pairs). The zero-frequency family is reported with its shapes in `tr.zero` and is never
   seeded as a harmonic master. On record A, `windowed()` tracks the master in all six
   windows (1.0074 to 1.0006 rad/s, within 0.12 % of the FFT peak of the master coordinate).
2. **Identity.** When a window has fewer pairs than tracks, the Python 1.1.1 tracker
   discards the assignment's row indices and writes the candidates into tracks 0, 1, ...
   On record B (exact piecewise-linear record, mode 2 real in windows 2-3) mode 3 is written
   into the mode-2 track and mode 2 is never recovered. MATLAB 1.1.1 keeps the row indices
   but stores a zero reference for the missing track, so that track cannot be recovered
   either. Remedy: `matching='identity'`. The assignment is against each track's last
   assigned window; a missing or rejected match is a gap (NaN, `tr.gap`); the proposed costs
   are returned in `tr.cost`; `max_cost` rejects poor matches (record B2: an unrelated pole
   that replaces mode 2 at cost 1.21 is rejected at `max_cost=0.5`).
3. **Closure with one tracked mode.** `closure(V0=[v0], corrections=[c], k=1)` returns
   `None`, because a deflation correction is orthogonal to the v0 of its own family, so the
   only candidate P = 1 can never close. Remedies: `closure(..., strict=True)` raises
   ValueError (MATLAB: `ncvd:closure`) stating the reason (degenerate posing, too few
   correction columns, or not met within `p_max`); the new `rom.family_closure(shapes, tol)`
   returns the closure of one shape family, P = 1 + #(relative deflation singular values
   >= tol).
4. **Rank.** `rank=None` keeps every direction of [disp, vel]. On record D (linear 4-DOF
   chain released in mode 1, rank 2 of 8) three of the four tracked "modes" are null-space
   eigenvalues, and their values change when the balanced quotient is replaced by another
   valid one. Remedy: `rank='auto'` in `track`, `ncvd` and `ncvd_pair`. It uses the declared
   rule `numerical_rank(s, rtol=1e-12, gap=None)` (relative threshold, optional first-gap
   cut), taking the minimum over windows in `track`. The rank used, the per-window ranks, the
   rule and the singular values are returned; an integer `rank` remains the explicit
   override. On record D the tracked pole equals the true pole to 8.9e-16 and does not change
   with the quotient.

### New functions

* `windowed(disp, vel, acc, fs, ...)` (MATLAB `ncvd.windowed`): `track` with every remedy
  on: `center=False, rank='auto', seed='energy', families=True, matching='identity',
  max_cost=0.5`. Earlier work assumed a `windowed` function that did not exist; the 1.1.1
  windowed path is `track(..., center=False)` with its defaults, which is unchanged.
* `numerical_rank` (MATLAB `ncvd.numericalRank`), `rom.family_closure` (MATLAB
  `ncvd.familyClosure`).

### Unchanged on purpose

* `rom.robustness_basis`, `rom.deflate`, `rom.escape`, `rom.project`, `rom.augment`,
  `rom.residual_augmentation` and their MATLAB counterparts are byte-identical to 1.1.1
  (tested). The toolbox has no function that builds a resonance robustness matrix S_m by
  frequency band: `robustness_basis(shapes, rank=None, tol=1e-3)` takes one resonance's
  shape family as given and returns its SVD basis and normalized singular values.
* Python and MATLAB 1.1.1 defaults differ when a window has fewer pairs than tracks
  (Python relabels; MATLAB leaves a NaN but loses the track's reference). Both are published
  behavior and are kept. The opt-in identity matching is the same in both languages.
* MATLAB `closure` raises an index error when `corrections` has fewer than k columns, while
  Python truncates silently. Kept; `strict=True` raises in both.

### Still open

* The two known issues of 1.1.1 (`systems.simulate_nonlinear` callback shape,
  `diagnostics.channel_slopes` signed mask) are not addressed in this release.
* Code docstrings carried over from earlier releases still contain British spellings; they
  are left verbatim because the tests check the 1.1.1 functions byte for byte.

### Tests

Python: 48 (1.1.1) -> 94, all passing (20 default-agreement calls, 5 defect witnesses,
16 opt-in tests, 5 integrity checks). MATLAB: 14 -> 23 test functions. The MATLAB changes
were parsed with MISS_HIT 0.9.44 (no syntax errors) but NOT RUN: MATLAB could not start in
the build environment. Run `runtests('matlab/tests/test_ncvd.m')` before publishing.

## 1.1.1 -- 2026-10-27

### Known issues (found after release, 29-31 Oct 2026; not fixed in this version)

* `systems.simulate_nonlinear` calls the nonlinear-force callback `fnl` with two
  different shapes: once per state vector `(n,)` inside the right-hand side, and once
  with the whole `(N, n)` record to build the returned acceleration block. A callback
  written only for a single state vector returns a wrong acceleration with no error.
  Write `fnl` shape-safe (index with `q[..., j]` and broadcast).
* `diagnostics.channel_slopes` masks on the SIGNED damping change
  (`dz > tol*|zeta_lin|`, `tol = 0.02`) and then requires more than three ladder rungs
  to clear it, so it returns `None` for damping channels that fall below the linear
  baseline and for partial ladders. Compute slopes on `|dz|` and `|df|` until this is
  fixed.

**Forced modal analysis works on the canonical path.** Making `X1^+ X2` the definition had
left `operators.ncvd_snapshot` -- the reference implementation of that definition -- without
a `forcing` argument, so the canonical routine raised TypeError on a capability the legacy
Q route has had all along. Ported, with the two things that are easy to lose and are now
covered by tests that fail if either is dropped:

* `U` is built on the UNPROJECTED pair (`U = Y1 E`), not on the input-removed one, so the
  modal coordinates are those of the actual response.
* `R` carries the `-F G` term. Without it the input contribution reappears as apparent
  nonlinearity and poisons every residual readout in `diagnostics`.

`floor` now includes the input in its design matrix. `G` is returned. Four tests added
(48 total): pole recovery < 1e-8 against > 1e-3 when the input is ignored, `U = Y1 E`,
the residual term, and agreement with `core.ncvd(forcing=...)`.

Also documented, not changed: `ncvd_snapshot(weights=)` are SPATIAL per-column weights
while `core.ncvd_pair(weights=)` are per-ROW weights -- same name, different objects; and
estimator variants (TLS, bias-compensated LS) remain in `core.ncvd_pair` by design, since
`ncvd_snapshot` *is* the canonical least-squares definition.

## 1.1.0 -- 2026-10-27

Adds the operator generalization, residual diagnostics and iterative refinement.
No existing API changed behavior; the 29 tests of 1.0.0 still pass unmodified
(44 pass in total).

### New: `ncvd.operators` -- NCVD under an arbitrary linear operator
The canonical NCVD is the eigendecomposition of the least-squares operator map
B = Y1^+ Y2, and nothing in that statement refers to d/dt. The same
decomposition therefore applies to any linear operator with Y2 = D[Y1].

* `ncvd_snapshot(Y1, Y2, rank=None, weights=None)` -- reduced form, valid for
  any m, N and rank, computed in an orthonormal basis of the retained row space.
  Never forms the N x N balanced quotient, so it is the right routine when
  there are fewer snapshots than stations. **Its `rank=None` keeps the
  NUMERICAL rank**, unlike `core.ncvd_pair`, whose `rank=None` keeps every
  direction.
* `from_state`, `from_delays`, `from_spatial` -- derivative (CVD), shift
  (DMD/Koopman) and spatial-derivative forms.
* `state_completeness(Y1, Y2)` -- the least-squares floor. The row space of Y1
  must be invariant under the operator; displacement-only snapshots of a
  second-order system give ~0.8, the state [q, qdot] gives ~1e-14. A large
  floor on clean data means a missing coordinate, not a bad decomposition.
* `aliasing_margin(dt, lam)` -- warns past Nyquist. The shift form is exact to
  8e-13 up to dt = 0.81 dt_Nyquist and fails (0.21 -> 0.98) beyond it; the
  derivative form holds at ~3e-15 out to dt = 8.1 dt_Nyquist.
* The spatial form needs no state augmentation, no uniform sampling and no time
  ordering (permuting snapshots leaves it unchanged), and only m >= K snapshots
  for K modes against m >= 2K for the temporal form.

### New: `ncvd.diagnostics` -- the residual is the nonlinearity
R = (I - Y1 Y1^+) Y2 is the nonlinear force field projected off the data span
(agreement with (I - P) F_nl: 4.8e-14).

* `residual_direction` / `residual_report` -- WHERE (leading direction matches
  M^-1 w to MAC 1.000000, rank-1 share 1.000000), and TYPE via projected-library
  attribution (cubic spring: displacement 1.000 / velocity 0.008; quadratic drag
  and cubic damping: velocity 1.000; Coulomb: sign(v) 0.9998 with polynomials
  failing, so non-smoothness is itself diagnosable).
* `amplitude_exponent` -- ORDER of the force: 3.015 cubic, 2.001 quadratic,
  0.999 signal-proportional noise, -0.001 fixed noise floor. Perturbative, and
  it needs independent releases, not windows of one decay.
* `channel_slopes` -- conservative and dissipative nonlinearities separate
  between the two channels of lambda. Cubic stiffness: damping flat, frequency
  slope +2.000. Quadratic drag +1.040, cubic damping +2.114, Coulomb -0.867
  (describing-function predictions +1, +2, -1). Discriminant ratio ~2
  conservative against 1e3-1e4 dissipative.
* `window_cycles` -- warns below ~3 cycles of the slowest tracked mode per
  window. On one cubic record, 0.13 cycles/window gave relative residual 4.0e-7
  and exponent 1.29 where 10 cycles/window gave 1.4e-2 and 3.01 on the SAME
  data. Pole tracking is unaffected; only residual diagnostics need the window.

Two structural facts now documented: the kinematic block of R is zero by
construction (velocity is a column of the regressor), so report
||R_dyn||/||qddot||, not ||R||/||X2||; and R is orthogonal to the data span, so
<R, qdot> = 0 identically -- the residual carries no net power and cannot by
itself separate conservative from dissipative.

### New: `ncvd.refine` -- iterative refinement under noise
The canonical estimator has no significant bias (signed damping bias at 45 dB:
-0.29 +/- 0.33 % against a 2.50 % median absolute error); its error is variance.
So TLS and forward-backward averaging buy little. `varpro` / `refine_poles`
(Gauss-Newton on the poles with the amplitudes eliminated, warm-started from the
canonical NCVD) cut the damping error ~40x at 45 dB and ~38x at 30 dB, and the
frequency error 175x at 30 dB, and handle non-uniform sampling natively.

### New in `ncvd.rom`
* `residual_augmentation(R, K, M=None, n_dir=1)` -- the residual-informed static
  correction K^-1 w, where w is the nonlinear force direction recovered from R.
  Spending the extra basis vector on it rather than on the next linear mode
  reduced the in-band response error by 1.4x-3.2x wherever the nonlinear force
  reached ~0.3 of the linear restoring force, and was break-even (0.55x-0.98x)
  below that. The gain tracked the STRENGTH of the nonlinearity, not the escape
  of the correction from the retained band.
* `augment(V, extra)` -- orthonormal union.
* `project` now warns when the basis is not orthonormal: P^+ M P is then not
  symmetric and the reduced model loses the energy structure.

### Changed in `ncvd.core`
* `ncvd_pair` warns when Y1 is numerically rank deficient and `rank=None`. The
  balanced quotient Q cancels exactly from the canonical output ONLY when Y1 has
  full column rank; on a rank-deficient record the pseudo-inverse lets it back
  in and the result is an artefact of whichever Q is formed (on a rank-2-of-8
  record, substituting an equally valid Q moved the poles by 48 and destroyed
  the shapes).
  **This fires on the toolbox's own 10-DOF paper chain**: a 666-sample tracking
  window spans only 13 of 20 state directions, and there `rank=None` is wrong by
  7.8e-2 in the worst pole and varies with Q, while rank truncation gives
  4.6e-3 and is Q-independent -- a 17x improvement from one argument.
* `NCVDResult` now carries `E`, the extractor (U = Y1 E, Phi = E^-H), on the
  canonical path. `Psi = conj(Phi)` remains the physical eigenvector shapes.
