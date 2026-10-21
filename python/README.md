# NCVD toolbox — Python

```
pip install -e .            # from this folder; add [plot,test] for figures and tests
python -m pytest -q         # 29 tests
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

## How to cite

`ncvd.cite()` prints BibTeX for the software
([doi:10.5281/zenodo.23549069](https://doi.org/10.5281/zenodo.23549069))
and the method paper (Li, Stein & Chelidze, MSSP 222 (2025) 111769). Please cite both.
