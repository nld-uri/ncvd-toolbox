function idx = physicalModes(lam, tol)
%NCVD.PHYSICALMODES  Indices of poles with Im > 0, ascending damped frequency.
if nargin < 2, tol = 1e-9; end
lam = lam(:);
idx = find(imag(lam) > tol*max(abs(lam), 1e-300));
[~, o] = sort(imag(lam(idx))); idx = idx(o);
end
