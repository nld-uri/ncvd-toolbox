# NCVD toolbox — MATLAB

Base MATLAB R2022a or later (name-value syntax); no toolboxes required.

```matlab
addpath('path/to/NCVD Toolbox/matlab')
rec = ncvd.decompose(d, v, a);           % X1 = U*Phi',  X2 = U*diag(lam)*Phi' + R
mp  = ncvd.modalParameters(rec.lam);
results = runtests('tests/test_ncvd.m')  % 23 tests, including Python cross-checks
```

`help ncvd` lists every function. `ncvd.track` keeps the 1.1.1 procedure as its
default; `ncvd.windowed` is the opt-in 1.1.2 protocol. `tests/v111/` holds the 1.1.1
versions of the four changed functions, renamed `*_v111`, so the suite can check that
every default call still returns what 1.1.1 returns. Examples are in `examples/`; the manual is
`../docs/manual/NCVD_manual.pdf`.

## How to cite

`ncvd.cite` prints BibTeX for the software
([doi:10.5281/zenodo.23549068](https://doi.org/10.5281/zenodo.23549068))
and the method paper (Li, Stein & Chelidze, MSSP 222 (2025) 111769). Please cite both.
