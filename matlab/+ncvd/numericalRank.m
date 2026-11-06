function r = numericalRank(s, rtol, gap)
%NCVD.NUMERICALRANK  Numerical rank from singular values (added in 1.1.2).
%
%   r = ncvd.numericalRank(s)              r = #{ i : s(i) > 1e-12*s(1) }
%   r = ncvd.numericalRank(s, rtol)        relative threshold rtol
%   r = ncvd.numericalRank(s, rtol, gap)   ... then cut at the FIRST i < r with
%                                          s(i)/s(i+1) > gap (gap criterion)
%
%   The declared criterion of Rank="auto" in ncvd.decomposePair,
%   ncvd.decompose and ncvd.track. s must be in descending order (as returned
%   by svd). Returns 0 for an all-zero matrix. Python: ncvd.numerical_rank.
if nargin < 2 || isempty(rtol), rtol = 1e-12; end
if nargin < 3, gap = []; end
s = s(:);
if isempty(s) || ~(s(1) > 0), r = 0; return; end
r = sum(s > rtol*s(1));
if ~isempty(gap) && r > 1
    if ~(gap > 1), error('ncvd:gap', 'gap must be a ratio > 1'); end
    ratio = s(1:r-1)./s(2:r);
    k = find(ratio > gap, 1);
    if ~isempty(k), r = k; end
end
end
