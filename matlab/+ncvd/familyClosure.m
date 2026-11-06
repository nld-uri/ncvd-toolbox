function [P, sv] = familyClosure(shapes, tol, ref)
%NCVD.FAMILYCLOSURE  Closure dimension of ONE resonance's shape family (new in 1.1.2).
%
%   [P, sv] = ncvd.familyClosure(shapes)            tol = 0.05
%   [P, sv] = ncvd.familyClosure(shapes, tol, ref)
%
%   P = 1 + #{ i : sv(i) >= tol }, sv the singular values of the deflated
%   family S - v0*(v0'*S) relative to the family's leading singular value
%   (the construction of ncvd.deflate, whose mag is sv(1)). P is the smallest
%   number of directions [v0, c1, ..., c(P-1)], ci the leading deflation
%   directions, containing every shape of the family to within tol. P = 1
%   means the family is flat to tol. shapes: nWindows x n, LAST row = lowest-
%   amplitude window (default v0). Python: ncvd.rom.family_closure.
if nargin < 2 || isempty(tol), tol = 0.05; end
S = shapes.';
svf = svd(S);
if nargin < 3 || isempty(ref), v = S(:,end); else, v = ref(:); end
v0 = v/max(norm(v), 1e-300);
Rr = S - v0*(v0'*S);
sv = svd(Rr)/max(svf(1), 1e-300);
P = 1 + sum(sv >= tol);
end
