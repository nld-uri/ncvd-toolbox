function out = cite(fmt)
%NCVD.CITE  How to cite the NCVD toolbox (software DOI and method paper).
%   ncvd.cite            prints BibTeX for the software and the method paper
%   ncvd.cite("text")    prints plain-text references
%   s = ncvd.cite(...)   returns the text instead of printing it
%
%   Software: doi:10.5281/zenodo.23549069
%   Method:   doi:10.1016/j.ymssp.2024.111769
arguments
    fmt (1,1) string {mustBeMember(fmt, ["bibtex", "text"])} = "bibtex"
end
nl = newline;
if fmt == "bibtex"
    s = strjoin([ ...
        "@software{chelidze2026ncvd,"
        "  author    = {Chelidze, David and Li, Hewenxuan and Stein, Dalton L.},"
        "  title     = {{NCVD} toolbox: Nonlinear Characteristic Value Decomposition for {Python} and {MATLAB}},"
        "  version   = {1.0.0},"
        "  year      = {2026},"
        "  publisher = {Zenodo},"
        "  doi       = {10.5281/zenodo.23549069},"
        "  url       = {https://github.com/nld-uri/ncvd-toolbox}"
        "}"
        ""
        "@article{li2025cvd,"
        "  author  = {Li, Hewenxuan and Stein, Dalton L. and Chelidze, David},"
        "  title   = {Characteristic value decomposition: A unifying paradigm for data-driven modal analysis},"
        "  journal = {Mechanical Systems and Signal Processing},"
        "  volume  = {222},"
        "  pages   = {111769},"
        "  year    = {2025},"
        "  doi     = {10.1016/j.ymssp.2024.111769}"
        "}"], nl);
else
    s = strjoin([ ...
        "D. Chelidze, H. Li, D. L. Stein, NCVD toolbox: Nonlinear Characteristic Value Decomposition for Python and MATLAB, version 1.0.0, Zenodo (2026). https://doi.org/10.5281/zenodo.23549069"
        ""
        "H. Li, D. L. Stein, D. Chelidze, Characteristic value decomposition: A unifying paradigm for data-driven modal analysis, Mechanical Systems and Signal Processing 222 (2025) 111769. https://doi.org/10.1016/j.ymssp.2024.111769"], nl);
end
if nargout, out = char(s); else, fprintf('%s\n', s); end
end
