# NCVD toolbox — Python

```
pip install -e .            # from this folder; add [plot,test] for figures and tests
python -m pytest -q         # 94 tests
python examples/ex01_linear_chain.py
```

```python
import ncvd as nc
rec = nc.ncvd(disp, vel, acc)            # X1 = U Phi^H,  X2 = U Lam Phi^H + R
rec.lam, rec.Psi, rec.U, rec.res         # poles, eigenvector shapes, coordinates, residual
nc.modal_parameters(rec.lam)             # wn, zeta, fn, ...
```

See `../docs/manual/NCVD_manual.pdf` for the theory, the API and the worked examples.
`tests/_reference_ncvd.py` is the original `ncvd.py`; the test suite checks that the
toolbox reproduces it exactly for the original arguments.

`tests/test_release_112.py` checks that every default call of 1.1.2 returns what 1.1.1
returns (against the frozen copy `tests/_v1_1_1/`, verified against the tag's git
blobs), pins the four 1.1.1 tracking defects on the default path, and tests the opt-in
remedies on the synthetic records of `tests/_records.py`.
`tests/make_matlab_reference_112.py` writes `matlab/tests/reference_112.mat` for the
MATLAB cross-checks.

## How to cite

`ncvd.cite()` prints BibTeX for the software
([doi:10.5281/zenodo.23549068](https://doi.org/10.5281/zenodo.23549068))
and the method paper (Li, Stein & Chelidze, MSSP 222 (2025) 111769). Please cite both.
