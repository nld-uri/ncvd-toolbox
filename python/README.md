# NCVD toolbox — Python

```
pip install -e .            # from this folder; add [plot,test] for figures and tests
python -m pytest -q         # 28 tests
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
