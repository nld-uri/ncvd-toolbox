function [P, sv] = robustnessBasis(shapes, rank, tol)
%NCVD.ROBUSTNESSBASIS  Orthonormal basis of one resonance's shape family
%   (shapes: nWindows x n) and its normalised singular values.
if nargin < 3, tol = 1e-3; end
[U, S, ~] = svd(shapes.', 'econ'); sv = diag(S)/S(1,1);
if nargin < 2 || isempty(rank), rank = max(1, sum(sv > tol)); end
P = U(:, 1:rank);
end
