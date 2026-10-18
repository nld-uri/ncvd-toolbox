function [d, v, a] = simulateLinear(M, C, K, x0, t)
%NCVD.SIMULATELINEAR  Exact free response by modal superposition.
A = ncvd.stateMatrix(M, C, K);
[E, D] = eig(A); lam = diag(D);
c = E\x0(:);
X = real((exp(t(:)*lam.').*c.')*E.');
Xd = X*A.';
n = size(M,1);
d = X(:,1:n); v = X(:,n+1:end); a = Xd(:,n+1:end);
end
