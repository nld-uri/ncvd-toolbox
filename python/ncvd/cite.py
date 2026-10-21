"""Citation information for the NCVD toolbox.

    >>> import ncvd
    >>> print(ncvd.cite())             # BibTeX for the software and the method paper
    >>> print(ncvd.cite('text'))       # plain-text references
"""

DOI = '10.5281/zenodo.23549069'
PAPER_DOI = '10.1016/j.ymssp.2024.111769'

_BIBTEX = r"""@software{chelidze2026ncvd,
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
}"""

_TEXT = """D. Chelidze, H. Li, D. L. Stein, NCVD toolbox: Nonlinear Characteristic Value \
Decomposition for Python and MATLAB, version 1.0.0, Zenodo (2026). \
https://doi.org/10.5281/zenodo.23549069

H. Li, D. L. Stein, D. Chelidze, Characteristic value decomposition: A unifying paradigm \
for data-driven modal analysis, Mechanical Systems and Signal Processing 222 (2025) 111769. \
https://doi.org/10.1016/j.ymssp.2024.111769"""


def cite(fmt='bibtex'):
    """Return how to cite the toolbox: fmt = 'bibtex' (default) or 'text'.

    Please cite both the software (Zenodo DOI) and the method paper.
    """
    fmt = fmt.lower()
    if fmt == 'bibtex':
        return _BIBTEX
    if fmt == 'text':
        return _TEXT
    raise ValueError("fmt must be 'bibtex' or 'text'")
