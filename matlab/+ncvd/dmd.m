function out = dmd(X, dt, rank)
%NCVD.DMD  Exact DMD of snapshot rows X (m x p); lam = log(mu)/dt.
X1 = X(1:end-1,:).'; X2 = X(2:end,:).';
[U, S, V] = svd(X1, 'econ'); s = diag(S);
if nargin < 3 || isempty(rank), rank = numel(s); end
U = U(:,1:rank); s = s(1:rank); V = V(:,1:rank);
At = U'*X2*V./s.';
[Wt, Mu] = eig(At); mu = diag(Mu);
out = struct('lam', log(complex(mu))/dt, 'Phi', X2*V./s.'*Wt, 'mu', mu);
end
