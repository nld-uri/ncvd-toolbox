function [v0, corr, mag] = deflate(shapes, ref)
%NCVD.DEFLATE  Near-linear shape v0, leading amplitude correction and its size.
%   shapes: nWindows x n, LAST row = lowest-amplitude window (default ref).
S = shapes.';
svf = svd(S);
if nargin < 2 || isempty(ref), v = S(:,end); else, v = ref(:); end
v0 = v/max(norm(v), 1e-300);
Rr = S - v0*(v0'*S);
[U, Sr, ~] = svd(Rr, 'econ');
corr = U(:,1); [~, k] = max(abs(corr)); if corr(k) < 0, corr = -corr; end
mag = Sr(1,1)/max(svf(1), 1e-300);
end
