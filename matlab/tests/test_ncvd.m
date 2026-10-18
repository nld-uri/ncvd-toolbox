function tests = test_ncvd
%TEST_NCVD  Unit tests and Python cross-checks for the NCVD toolbox.
%   Run from the matlab folder:  results = runtests('tests/test_ncvd.m')
tests = functiontests(localfunctions);
end

function setupOnce(tc)
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
tc.TestData.ref = load(fullfile(here, 'reference.mat'));
[M, C, K] = ncvd.paperChain10();
t = (0:0.05:60-0.05)';
tc.TestData.M = M; tc.TestData.C = C; tc.TestData.K = K;
tc.TestData.lam = eig(ncvd.stateMatrix(M, C, K));
r = tc.TestData.ref;
tc.TestData.d = r.c10_d; tc.TestData.v = r.c10_v; tc.TestData.a = r.c10_a;
tc.TestData.t = t;
end

function e = nearestErr(est, truth)
e = 0;
for i = 1:numel(est), e = max(e, min(abs(est(i) - truth(:)))); end
end

function o = offd(S)
D = diag(diag(S)); o = norm(S - D, 'fro')/norm(D, 'fro');
end

% ------------------------------------------------------------ exactness
function testExactIdentities(tc)
D = tc.TestData; r = ncvd.decompose(D.d, D.v, D.a);
verifyLessThan(tc, r.exact1, 1e-12); verifyLessThan(tc, r.res, 1e-12);
verifyLessThan(tc, nearestErr(r.lam, D.lam), 1e-10);
X1 = [D.d D.v]; X2 = [D.v D.a];
verifyLessThan(tc, norm(X1 - r.U*r.Phi', 'fro')/norm(X1,'fro'), 1e-12);
verifyLessThan(tc, norm(X2 - (r.U.*r.lam.')*r.Phi', 'fro')/norm(X2,'fro'), 1e-12);
end

function testShapesAndEquivalence(tc)
D = tc.TestData; r = ncvd.decompose(D.d, D.v, D.a);
[lt, Et] = ncvd.trueModes(D.M, D.C, D.K);
for i = 1:numel(lt)
    [~, k] = min(abs(r.lam - lt(i)));
    verifyGreaterThan(tc, ncvd.mac(r.Psi(1:10,k), Et(:,i)), 1 - 1e-10);
end
Y1 = [D.d D.v]; Y2 = [D.v D.a]; B = Y1\Y2;
r = ncvd.decompose(D.d, D.v, D.a, Sort=false);
E = inv(r.V)'*r.T;
verifyLessThan(tc, norm(B*E - E.*r.lam.', 'fro')/norm(B,'fro'), 1e-9);
verifyLessThan(tc, norm(r.Phi - inv(E)', 'fro')/norm(r.Phi,'fro'), 1e-9);
end

function testRankAndTimeScale(tc)
D = tc.TestData;
verifyLessThan(tc, nearestErr(ncvd.decompose(D.d, D.v, D.a, Rank=20).lam, D.lam), 1e-9);
for c = {0.1, 7, "auto"}
    verifyLessThan(tc, nearestErr(ncvd.decompose(D.d, D.v, D.a, TimeScale=c{1}).lam, D.lam), 1e-10);
end
end

function testJoukowskiTie(tc)
n = 4; K = 2*eye(n) - diag(ones(n-1,1),1) - diag(ones(n-1,1),-1); K(end,end) = 1;
C = 0.002*eye(n) + 0.001*K;
r = tc.TestData.ref;
[d, v, a] = ncvd.simulateLinear(eye(n), C, K, [1 0 -1 0.5 0 0.3 0 -0.2], (0:0.05:200-0.05)');
lam = eig(ncvd.stateMatrix(eye(n), C, K));
rc = ncvd.decompose(d, v, a);
verifyLessThan(tc, nearestErr(rc.lam, lam), 1e-9);              % canonical exact
ra = ncvd.decompose(d, v, a, TimeScale="auto");
verifyLessThan(tc, offd(ra.S2), 1e-8);                           % auto scale unfolds
end

% ------------------------------------------------------------ Python cross-checks
function testCrossPythonClean(tc)
r = tc.TestData.ref; m = ncvd.decompose(r.c10_d, r.c10_v, r.c10_a);
verifyLessThan(tc, max(abs(m.lam - r.c10_lam(:))), 1e-10);       % same sort order
for i = 1:numel(m.lam)
    verifyGreaterThan(tc, ncvd.mac(m.Psi(:,i), r.c10_Psi(:,i)), 1 - 1e-10);
end
end

function testCrossPythonNoisyOperators(tc)
r = tc.TestData.ref;
d = r.c10n_d; v = r.c10n_v; a = r.c10n_a;
verifyLessThan(tc, nearestErr(ncvd.decompose(d,v,a).lam, r.c10n_lam_ls), 1e-9);
verifyLessThan(tc, nearestErr(ncvd.decompose(d,v,a,Operator="tls").lam, r.c10n_lam_tls), 1e-7);
nv = ncvd.noiseVariance([d v], 20);
verifyLessThan(tc, max(abs(nv(:) - r.c10n_noisevar(:))./r.c10n_noisevar(:)), 1e-8);
verifyLessThan(tc, nearestErr(ncvd.decompose(d,v,a,Operator="bcls",NoiseVar="auto",Fs=20).lam, r.c10n_lam_bcls), 1e-8);
verifyLessThan(tc, nearestErr(ncvd.decompose(d,v,a,Readout="diag").lam, r.c10n_lam_diag), 1e-8);
verifyLessThan(tc, nearestErr(ncvd.decompose(d,v,a,Rank=16).lam, r.c10n_lam_rank), 1e-8);
end

function testCrossPythonFD(tc)
r = tc.TestData.ref;
[M, C, K] = ncvd.chain(3, 1, 1, 0.05); lam = eig(ncvd.stateMatrix(M, C, K));
for m = ["forward" "central" "central4" "savgol"]
    [d, v, a, f] = ncvd.stateFromDisplacement(r.fd_q, 10, m);
    lr = ncvd.decompose(d, v, a).lam;
    verifyLessThan(tc, nearestErr(lr, r.("fd_" + m + "_raw")), 1e-9);
    lc = ncvd.correctPoles(lr, f, 10);
    verifyLessThan(tc, nearestErr(lc, lam), 1e-10);
    verifyLessThan(tc, abs(ncvd.filterBand(f) - r.("fd_" + m + "_band")), 1e-6);
end
end

function testCrossPythonTracking(tc)
r = tc.TestData.ref;
[Md, ~, ~] = ncvd.duffingChain(3, 1, 1, 0, [1 1]);
tr = ncvd.track(r.duf_d, r.duf_v, r.duf_a, Fs=20, Length=800, Step=400, NTrack=3, M=Md);
verifyLessThan(tc, max(abs(tr.wd - r.duf_wd), [], 'all'), 1e-8);
verifyLessThan(tc, max(abs(tr.res - r.duf_res(:))), 1e-10);
verifyLessThan(tc, max(abs(tr.energy - r.duf_energy)./r.duf_energy, [], 'all'), 1e-8);
end

function testNonlinearSimulationMatchesPython(tc)
r = tc.TestData.ref;
[M, ~, K, fnl] = ncvd.duffingChain(3, 1, 1, 0, [1 1]); C = 0.003*M;
t = (0:0.05:600-0.05)';
[d, ~, ~] = ncvd.simulateNonlinear(M, C, K, fnl, r.duf_x0, t);
verifyLessThan(tc, max(abs(d - r.duf_d), [], 'all'), 1e-6);
end

% ------------------------------------------------------------ other functions
function testForcing(tc)
[M, C, K] = ncvd.chain(4, 1, 2, 0.05); lam = eig(ncvd.stateMatrix(M, C, K));
t = (0:0.02:100-0.02)'; s = RandStream('mt19937ar', 'Seed', 3);
f = randn(s, numel(t), 4);
[d, v, a] = ncvd.simulateForcedLinear(M, C, K, f, t);
verifyLessThan(tc, nearestErr(ncvd.decompose(d, v, a, Forcing=f).lam, lam), 1e-8);
verifyGreaterThan(tc, nearestErr(ncvd.decompose(d, v, a).lam, lam), 1e-3);
end

function testEnsemble(tc)
D = tc.TestData; s = RandStream('mt19937ar', 'Seed', 5); recs = {};
for p = [100 30; 125 20; 200 10]'
    t = (0:1/p(1):p(2)-1/p(1))';
    [d, v, a] = ncvd.simulateLinear(D.M, D.C, D.K, randn(s, 20, 1), t);
    recs{end+1} = {d, v, a}; %#ok<AGROW>
end
r = ncvd.ensemble(recs);
verifyLessThan(tc, nearestErr(r.lam, D.lam), 1e-9);
verifyEqual(tc, numel(r.blocks), 3);
end

function testBenchmarks(tc)
D = tc.TestData; X = [D.d D.v];
verifyLessThan(tc, nearestErr(ncvd.dmd(X, 0.05).lam, D.lam), 1e-8);
verifyLessThan(tc, nearestErr(ncvd.itd(X, 0.05, 20).lam, D.lam), 1e-6);
g = ncvd.gcvd(D.d, D.v, D.a, "forcing");
verifyLessThan(tc, nearestErr(g.lam, D.lam), 1e-9);
end

function testModalHelpers(tc)
s = RandStream('mt19937ar', 'Seed', 2); v = randn(s, 6, 1);
[w, m] = ncvd.realify(v*exp(0.7i));
verifyGreaterThan(tc, m, 1 - 1e-12); verifyGreaterThan(tc, ncvd.mac(w, v), 1 - 1e-12);
verifyGreaterThan(tc, ncvd.macPhaseMatched(v*exp(0.3i), v + 0i), 1 - 1e-12);
v0 = randn(s, 8, 1); v0 = v0/norm(v0); c = randn(s, 8, 1); c = c - v0*(v0'*c); c = c/norm(c);
fam = (v0 + c*linspace(0.3, 0, 12)).'; fam = fam./vecnorm(fam, 2, 2);
[V0, corr, mag] = ncvd.deflate(fam);
verifyGreaterThan(tc, abs(V0'*v0), 1 - 1e-12); verifyGreaterThan(tc, abs(corr'*c), 1 - 1e-9);
verifyGreaterThan(tc, mag, 0); verifyLessThan(tc, ncvd.escape([v0 c], corr), 1e-9);
end
