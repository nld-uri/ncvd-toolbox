function P = closure(V0, corrections, k, tol, pMax)
%NCVD.CLOSURE  Smallest P with escape of the first k corrections from
%   span(V0(:,1:P)) below tol; [] if not met. Do not pose with k = P.
if nargin < 3, k = 1; end
if nargin < 4, tol = 0.05; end
if nargin < 5 || isempty(pMax), pMax = size(V0,2); end
C = corrections(:, 1:k);
for P = max(k,1):min(pMax, size(V0,2))
    if max(ncvd.escape(V0(:,1:P), C)) < tol, return; end
end
P = [];
end
