function [lam, E] = trueModes(M, C, K)
%NCVD.TRUEMODES  Exact poles with Im > 0 (ascending) and displacement shapes.
[V, D] = eig(ncvd.stateMatrix(M, C, K)); l = diag(D);
idx = find(imag(l) > 0); [~, o] = sort(imag(l(idx))); idx = idx(o);
n = size(M,1); lam = l(idx); E = V(1:n, idx);
end
