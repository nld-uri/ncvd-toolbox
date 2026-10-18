# NCVD toolbox — MATLAB

Base MATLAB R2022a or later (name-value syntax); no toolboxes required.

```matlab
addpath('path/to/NCVD Toolbox/matlab')
rec = ncvd.decompose(d, v, a);           % X1 = U*Phi',  X2 = U*diag(lam)*Phi' + R
mp  = ncvd.modalParameters(rec.lam);
results = runtests('tests/test_ncvd.m')  % 13 tests, including Python cross-checks
```

`help ncvd` lists every function. Examples are in `examples/`; the manual is
`../docs/manual/NCVD_manual.pdf`.
