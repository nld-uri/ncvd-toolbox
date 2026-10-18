function [d, v, a] = simulateForcedLinear(M, C, K, f, t, x0)
%NCVD.SIMULATEFORCEDLINEAR  Response to sampled force f (m x n), zero-order hold.
n = size(M,1);
A = ncvd.stateMatrix(M, C, K); Bm = [zeros(n); inv(M)];
dt = t(2) - t(1);
Z = expm([A Bm; zeros(n, 3*n)]*dt);
Ad = Z(1:2*n, 1:2*n); Bd = Z(1:2*n, 2*n+1:end);
if nargin < 6 || isempty(x0), x = zeros(2*n,1); else, x = x0(:); end
X = zeros(numel(t), 2*n);
for i = 1:numel(t)
    X(i,:) = x.'; x = Ad*x + Bd*f(i,:).';
end
d = X(:,1:n); v = X(:,n+1:end);
a = X*A(n+1:end,:).' + f*inv(M).';
end
