# NCVD toolbox — MATLAB

Base MATLAB R2022a or later (name-value syntax); no toolboxes required.

```matlab
addpath('path/to/NCVD Toolbox/matlab')
rec = ncvd.decompose(d, v, a);           % X1 = U*Phi',  X2 = U*diag(lam)*Phi' + R
mp  = ncvd.modalParameters(rec.lam);
results = runtests('tests/test_ncvd.m')  % 14 tests, including Python cross-checks
```

`help ncvd` lists every function. Examples are in `examples/`; the manual is
`../docs/manual/NCVD_manual.pdf`.

## How to cite

`ncvd.cite` prints BibTeX for the software
([doi:10.5281/zenodo.23549069](https://doi.org/10.5281/zenodo.23549069))
and the method paper (Li, Stein & Chelidze, MSSP 222 (2025) 111769). Please cite both.
