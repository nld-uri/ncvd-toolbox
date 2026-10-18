%EX07_READOUT_ERROR  Published (diagonal) readout against the canonical readout.
%   (a) exact Joukowski tie, (b) nonlinear windows: second-order pole error,
%   first-order shape error.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
ncvd.setStyle("matlab");
n = 4; K = 2*eye(n) - diag(ones(n-1,1),1) - diag(ones(n-1,1),-1); K(end,end) = 1;
C = 0.002*eye(n) + 0.001*K; rng(1);
[d, v, a] = ncvd.simulateLinear(eye(n), C, K, randn(2*n,1), (0:0.05:200-0.05)');
lam = eig(ncvd.stateMatrix(eye(n), C, K));
for c = [1 1+1e-6 1+1e-3]
    r = ncvd.decompose(d, v/c, a/c^2, Readout="diag");
    err = max(arrayfun(@(x) min(abs(lam/c - x)), r.lam))/min(abs(lam/c));
    S = r.S2; fprintf('c = %-10.7g off-diag %.2e  published pole error %.2e\n', c, ...
        norm(S - diag(diag(S)), 'fro')/norm(diag(S)), err);
end
