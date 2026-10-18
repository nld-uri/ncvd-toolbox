%EX06_FORCED_EMA  Measured forcing: exact poles from forced response.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
ncvd.setStyle("matlab");
[M, C, K] = ncvd.chain(4, 1, 2, 0.05); lt = ncvd.trueModes(M, C, K);
t = (0:0.02:60-0.02)'; rng(1);
f = randn(numel(t), 4).*[1 0.5 0 2];                 % mass 3 not driven
[d, v, a] = ncvd.simulateForcedLinear(M, C, K, f, t);
rf = ncvd.decompose(d, v, a, Forcing=f(:, [1 2 4]));
r0 = ncvd.decompose(d, v, a);
fprintf('with forcing: pole error %.1e\n', max(abs(rf.lam(ncvd.physicalModes(rf.lam)) - lt)));
fprintf('without     : pole error %.1e\n', max(abs(r0.lam(ncvd.physicalModes(r0.lam)) - lt)));
