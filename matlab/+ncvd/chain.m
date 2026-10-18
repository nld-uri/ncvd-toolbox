function [M, C, K] = chain(n, m, k, c, cStar, ends)
%NCVD.CHAIN  n-mass chain. m (n), k and c (n+1) scalars or vectors; element i
%   joins masses i-1 and i, the first and last join the walls.
%   cStar: [index value] rows of extra grounded dampers. ends: ["fixed" "fixed"].
if nargin < 4, c = 0; end
if nargin < 5, cStar = []; end
if nargin < 6, ends = ["fixed" "fixed"]; end
m = m(:).*ones(n,1); k = k(:).*ones(n+1,1); c = c(:).*ones(n+1,1);
if ends(1) == "free", k(1) = 0; c(1) = 0; end
if ends(2) == "free", k(end) = 0; c(end) = 0; end
M = diag(m); K = assemble(k); C = assemble(c);
for r = 1:size(cStar,1), C(cStar(r,1), cStar(r,1)) = C(cStar(r,1), cStar(r,1)) + cStar(r,2); end
end
function X = assemble(e)
n = numel(e) - 1;
X = diag(e(1:n) + e(2:n+1)) - diag(e(2:n), 1) - diag(e(2:n), -1);
end
