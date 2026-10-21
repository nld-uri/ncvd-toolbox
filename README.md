# NCVD toolbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23549069.svg)](https://doi.org/10.5281/zenodo.23549069)

**Nonlinear Characteristic Value Decomposition** for Python and MATLAB, from the
[Nonlinear Dynamics Laboratory](https://web.uri.edu/nld/) at the University of Rhode Island.

For a state record with displacement, velocity and acceleration at *n* stations,

```
X1 = [q, q']  = U Phi^H                  (exact)
X2 = [q', q''] = U Lam Phi^H + R          (exact for clean linear data)
```

with `U` the temporal basis, `Phi` the characteristic shapes, `Lam` the characteristic
values (poles) and `R` what no linear model of the record explains.

The decomposition builds on the characteristic value decomposition (CVD/GCVD) of
Li, Stein & Chelidze (2025). It reads the values from the eigenvalues of the coupling
matrix and rotates the shapes accordingly, so both identities are exact and each pole is
paired with its shape by construction.

## What it does

- Exact decomposition of a state record or of any matrix pair, with whitening, centring,
  row weights and a time-unit option for the balanced quotient
- Rates from displacement with any FIR stencil (forward, central, five-point,
  Savitzky–Golay) and exact pole correction inside each stencil's band
- Sensor noise: least squares (default), bias-compensated least squares, total least squares
- Measured forcing (no mass matrix needed) and ensembles of records with different
  sampling rates and lengths
- Amplitude-dependent identification of nonlinear systems by sliding windows:
  backbone curves, frequency–energy dependence, window residual
- Reduced-order-model diagnostics: robustness bases, deflation, escape, closure, projection
- Benchmarks (published GCVD readout, DMD, ITD) and test systems

Overview page: https://web.uri.edu/nld/ncvd-toolbox/

## Install and run

```bash
git clone https://github.com/nld-uri/ncvd-toolbox.git
cd ncvd-toolbox/python && pip install -e .[plot,test]
python -m pytest -q                      # 28 tests
python examples/ex01_linear_chain.py
```

```python
import ncvd as nc
rec = nc.ncvd(disp, vel, acc)
mp  = nc.modal_parameters(rec.lam)       # wn, wd, zeta, fn, fd
idx = nc.physical_modes(rec.lam)         # one pole per conjugate pair
shapes = rec.Psi[:n, idx]                # eigenvector shapes (displacement part)
```

MATLAB R2022a+ (base MATLAB, no toolboxes):

```matlab
addpath('matlab')
rec = ncvd.decompose(disp, vel, acc);
mp  = ncvd.modalParameters(rec.lam);
results = runtests('matlab/tests/test_ncvd.m')   % 13 tests, incl. Python cross-checks
```

## Contents

| Folder | Contents |
|---|---|
| `python/` | `ncvd` package, seven examples, replication of the published experiments, tests |
| `matlab/` | `+ncvd` package with the same API, seven examples, tests |
| `docs/manual/NCVD_manual.pdf` | user manual: theory, geometry, practical guidance, API (LaTeX source alongside, XeLaTeX) |
| [Live slides](https://claude.ai/artifact/5XebWDvuxhZizJR5Ls6mmn) | HTML slide deck with animations computed by the toolbox and running in the browser |
| `docs/slides/NCVD_toolbox_slides.pdf` | PDF export of the live slides |
| `docs/slides/live/` | the five animations as standalone pages, and the code that builds them |
| `docs/slides/NCVD_toolbox.pptx` | PowerPoint version of the live slides: animations as MP4 movies, speaker notes (`live/lib/export_pptx.py` builds it) |
| `docs/animations/` | animations (MP4) and stills (`make_animations.py` rebuilds them) |

## How to cite

Please cite the software ([doi:10.5281/zenodo.23549069](https://doi.org/10.5281/zenodo.23549069)) and the method paper.
GitHub's **Cite this repository** button (from `CITATION.cff`) gives both entries.

```bibtex
@software{chelidze2026ncvd,
  author    = {Chelidze, David and Li, Hewenxuan and Stein, Dalton L.},
  title     = {{NCVD} toolbox: Nonlinear Characteristic Value Decomposition for {Python} and {MATLAB}},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23549069},
  url       = {https://github.com/nld-uri/ncvd-toolbox}
}

@article{li2025cvd,
  author  = {Li, Hewenxuan and Stein, Dalton L. and Chelidze, David},
  title   = {Characteristic value decomposition: A unifying paradigm for data-driven modal analysis},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {222},
  pages   = {111769},
  year    = {2025},
  doi     = {10.1016/j.ymssp.2024.111769}
}
```

`@software` needs biblatex; with BibTeX use `@misc` with the same fields.
For the amplitude-dependent identification and ROM tools, please also cite

- D. L. Stein, D. Chelidze, Continuation of nonlinear normal modes using reduced-order models
  based on generalized characteristic value decomposition, *Nonlinear Dynamics* 113 (2025)
  25–45. https://doi.org/10.1007/s11071-024-10239-0

## License

MIT (see `LICENSE`). The IBM Plex fonts in `docs/*/fonts` are under the SIL Open Font
License (`OFL-LICENSE.txt`). The NLD logos and templates in `docs/` belong to the
Nonlinear Dynamics Laboratory.
