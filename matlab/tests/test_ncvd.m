function tests = test_ncvd
%TEST_NCVD  Unit tests and Python cross-checks for the NCVD toolbox.
%   Run from the matlab folder:  results = runtests('tests/test_ncvd.m')
%   The 1.1.2 tests at the end check that every default call equals the
%   1.1.1 function (tests/v111) to 1e-12 relative, and cross-check the opt-in
%   tracking options against Python (reference_112.mat, written by
%   python/tests/make_matlab_reference_112.py).
tests = functiontests(localfunctions);
end

function setupOnce(tc)
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
addpath(fullfile(here, 'v111'));                % the 1.1.1 functions, renamed *_v111
tc.TestData.ref = load(fullfile(here, 'reference.mat'));
tc.TestData.r112 = load(fullfile(here, 'reference_112.mat'));
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

function testCite(tc)
s = ncvd.cite();
verifyTrue(tc, contains(s, "10.5281/zenodo.23549068") && contains(s, "10.1016/j.ymssp.2024.111769"));
verifyTrue(tc, contains(ncvd.cite("text"), "10.5281/zenodo.23549068"));
verifyError(tc, @() ncvd.cite("apa"), ?MException);
end

% ------------------------------------------------------------ release 1.1.2
function r = rec112(tc, name)
q = tc.TestData.r112;
r = {q.([name '_d']), q.([name '_v']), q.([name '_a'])};
end

function sameOutput(tc, fNew, fOld, label)
% Both calls must fail with the same identifier or agree to 1e-12 relative.
eNew = ''; eOld = ''; a = []; b = [];
try, a = fNew(); catch ME, eNew = ME.identifier; if isempty(eNew), eNew = ME.message; end, end
try, b = fOld(); catch ME, eOld = ME.identifier; if isempty(eOld), eOld = ME.message; end, end
verifyEqual(tc, eNew, eOld, label + ": error differs");
if isempty(eNew), sameValue(tc, a, b, label); end
end

function sameValue(tc, a, b, label)
if isstruct(a) || isstruct(b)
    verifyTrue(tc, isstruct(a) && isstruct(b), label + ": struct vs non-struct");
    fa = sort(fieldnames(a)); fb = sort(fieldnames(b));
    verifyEqual(tc, fa, fb, label + ": fields differ");
    if ~isequal(fa, fb), return; end
    for i = 1:numel(fa), sameValue(tc, a.(fa{i}), b.(fa{i}), label + "." + fa{i}); end
    return
end
if isstring(a) || ischar(a) || islogical(a) || isempty(a) || isempty(b)
    verifyEqual(tc, a, b, label); return
end
verifyEqual(tc, size(a), size(b), label + ": size");
if ~isequal(size(a), size(b)), return; end
na = isnan(a); nb = isnan(b);
verifyEqual(tc, na, nb, label + ": NaN pattern");
f = ~na & ~nb;
if any(f(:))
    sc = max(max(abs(b(f))), 1e-300);
    verifyLessThanOrEqual(tc, max(abs(a(f) - b(f))), 1e-12*sc, label);
end
end

function testDefaultsMatchV111(tc)
% Every default call of 1.1.2 returns what 1.1.1 returns (tests/v111/*_v111.m
% are the tag's files with only the function name changed).
q = tc.TestData.r112;
A = rec112(tc, 'A'); B = rec112(tc, 'B'); B2 = rec112(tc, 'B2');
C = rec112(tc, 'C'); D = rec112(tc, 'D');
sameOutput(tc, @() ncvd.track(A{:}, Fs=20, NWindows=6, NTrack=1, M=q.A_M, Center=false), ...
    @() track_v111(A{:}, Fs=20, NWindows=6, NTrack=1, M=q.A_M, Center=false), "track A");
sameOutput(tc, @() ncvd.track(A{:}, Fs=20, NWindows=6), @() track_v111(A{:}, Fs=20, NWindows=6), "track A all");
sameOutput(tc, @() ncvd.track(B{:}, Fs=20, NWindows=6, NTrack=3), ...
    @() track_v111(B{:}, Fs=20, NWindows=6, NTrack=3), "track B");
sameOutput(tc, @() ncvd.track(B{:}, Fs=20, Length=1500, Step=750), ...
    @() track_v111(B{:}, Fs=20, Length=1500, Step=750), "track B overlap");
sameOutput(tc, @() ncvd.track(B2{:}, Fs=20, NWindows=6, NTrack=3), ...
    @() track_v111(B2{:}, Fs=20, NWindows=6, NTrack=3), "track B2");
sameOutput(tc, @() ncvd.track(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M), ...
    @() track_v111(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M), "track C");
sameOutput(tc, @() ncvd.track(C{:}, Fs=20, NWindows=10, Rank=4), ...
    @() track_v111(C{:}, Fs=20, NWindows=10, Rank=4), "track C rank 4");
sameOutput(tc, @() ncvd.track(D{:}, Fs=20, NWindows=4), @() track_v111(D{:}, Fs=20, NWindows=4), "track D");
sameOutput(tc, @() ncvd.track(D{:}, NWindows=4), @() track_v111(D{:}, NWindows=4), "track D no Fs");
sameOutput(tc, @() ncvd.decompose(A{:}), @() decompose_v111(A{:}), "decompose A");
sameOutput(tc, @() ncvd.decompose(A{:}, Center=true, TimeScale="auto"), ...
    @() decompose_v111(A{:}, Center=true, TimeScale="auto"), "decompose A centered");
sameOutput(tc, @() ncvd.decomposePair([D{1} D{2}], [D{2} D{3}]), ...
    @() decomposePair_v111([D{1} D{2}], [D{2} D{3}]), "decomposePair D");
sameOutput(tc, @() ncvd.decomposePair([D{1} D{2}], [D{2} D{3}], Rank=2), ...
    @() decomposePair_v111([D{1} D{2}], [D{2} D{3}], Rank=2), "decomposePair D rank 2");
sameOutput(tc, @() ncvd.decomposePair([B{1} B{2}], [B{2} B{3}]), ...
    @() decomposePair_v111([B{1} B{2}], [B{2} B{3}]), "decomposePair B");
tr = track_v111(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M);
[v0, c] = ncvd.deflate(squeeze(tr.shape(:,1,:)));
sameOutput(tc, @() ncvd.closure(v0, c, 1), @() closure_v111(v0, c, 1), "closure one mode");
verifyEmpty(tc, ncvd.closure(v0, c, 1));
K3 = 2*eye(3) - diag(ones(2,1), 1) - diag(ones(2,1), -1); [Phi, ~] = eig(K3);
sameOutput(tc, @() ncvd.closure(Phi, c, 1, 0.5), @() closure_v111(Phi, c, 1, 0.5), "closure modal");
sameOutput(tc, @() ncvd.closure(Phi, c), @() closure_v111(Phi, c), "closure defaults");
end

function testNumericalRank(tc)
s = [1 0.5 1e-6 1e-7 1e-14];
verifyEqual(tc, ncvd.numericalRank(s), 4);
verifyEqual(tc, ncvd.numericalRank(s, 1e-8), 4);
verifyEqual(tc, ncvd.numericalRank(s, 5e-6), 2);
verifyEqual(tc, ncvd.numericalRank(s, [], 1e3), 2);
verifyEqual(tc, ncvd.numericalRank(s, [], 1e9), 4);
verifyEqual(tc, ncvd.numericalRank(zeros(3,1)), 0);
end

function testRankAutoD(tc)
q = tc.TestData.r112; D = rec112(tc, 'D');
tr = ncvd.track(D{:}, Fs=20, NWindows=4, Rank="auto");
verifyEqual(tc, tr.rank, 2); verifyEqual(tc, tr.rankWindows, 2*ones(4,1));
verifyEqual(tc, tr.rankRule.rule, "min over windows");
verifyEqual(tc, size(tr.lam, 2), 1);
verifyLessThan(tc, max(abs(tr.lam - q.D_lam1)), 1e-10);
rec = ncvd.decompose(D{:}, Rank="auto");
verifyEqual(tc, rec.rank, 2);
verifyLessThan(tc, min(abs(rec.lam - q.D_lam1)), 1e-10);
verifyError(tc, @() ncvd.track(D{:}, Fs=20, NWindows=4, Rank="full"), 'ncvd:rank');
C = rec112(tc, 'C');
tc2 = ncvd.track(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M, Rank="auto");
verifyEqual(tc, tc2.rank, double(q.C_rankauto_rank));
verifyLessThan(tc, max(abs(tc2.wd - q.C_rankauto_wd), [], 'all'), 1e-8);
end

function testWindowedA(tc)
q = tc.TestData.r112; A = rec112(tc, 'A');
tr = ncvd.windowed(A{:}, 20, NWindows=6, NTrack=1, M=q.A_M);
verifyLessThan(tc, max(abs(tr.wd - 1)), 0.02);                 % the master, every window
verifyLessThan(tc, max(abs(tr.wd - q.A_windowed_wd)), 1e-8);   % Python
verifyEqual(tc, tr.rank, double(q.A_windowed_rank));
zl = tr.zero.lam; zp = q.A_windowed_zero_lam;
verifyEqual(tc, isnan(zl), isnan(zp));
verifyLessThan(tc, max(abs(zl(~isnan(zl)) - zp(~isnan(zp)))), 1e-8);
verifyGreaterThan(tc, ncvd.mac(squeeze(tr.zero.shape(1,1,:)), q.A_offset(:)), 0.9);
verifyLessThan(tc, abs(imag(zl(1)))*50/(2*pi), 1);             % < 1 cycle per window
t = ncvd.track(A{:}, Fs=20, NWindows=6, NTrack=1, M=q.A_M, Center=false, Rank="auto", ...
    Seed="energy", Matching="identity", Families=true, MaxCost=0.5);
sameValue(tc, tr, t, "windowed vs track");
verifyError(tc, @() ncvd.track(A{:}, NWindows=6, Families=true, Matching="identity"), 'ncvd:fs');
verifyError(tc, @() ncvd.track(A{:}, Fs=20, NWindows=6, Seed="energy"), 'ncvd:matching');
end

function testSeedTargetsA(tc)
q = tc.TestData.r112; A = rec112(tc, 'A');
tr = ncvd.track(A{:}, Fs=20, NWindows=6, NTrack=1, M=q.A_M, Center=false, Seed=2.0, ...
    Families=true, Matching="identity");
verifyLessThan(tc, max(abs(tr.wd - 2)), 0.03);
verifyLessThan(tc, max(abs(tr.wd - q.A_target_wd)), 1e-8);
end

function testIdentityMatchingB(tc)
q = tc.TestData.r112; B = rec112(tc, 'B');
tr = ncvd.track(B{:}, Fs=20, NWindows=6, NTrack=3, Matching="identity", Families=true, NZero=2);
w = imag(q.B_lam(:)).';
verifyLessThan(tc, max(abs(tr.wd([1 2 5 6], :) - w), [], 'all'), 1e-8);
verifyTrue(tc, all(isnan(tr.wd([3 4], 2))) && all(tr.gap([3 4], 2)));
verifyEqual(tc, tr.gap, logical(q.B_identity_gap));
verifyLessThan(tc, max(abs(tr.wd(~tr.gap) - q.B_identity_wd(~tr.gap))), 1e-8);
verifyEqual(tc, tr.nHarmonic, [3 3 2 2 3 3]');
verifyEqual(tc, tr.zero.seedWindow, 3);
verifyLessThan(tc, max(abs(sort(real(tr.zero.lam([3 4], :)), 2) - [-0.45 -0.15]), [], 'all'), 1e-8);
verifyTrue(tc, all(isnan(tr.zero.lam([1 2 5 6], :)), 'all'));
end

function testGateB2(tc)
q = tc.TestData.r112; B2 = rec112(tc, 'B2');
tr = ncvd.track(B2{:}, Fs=20, NWindows=6, NTrack=3, Matching="identity", MaxCost=0.5);
verifyTrue(tc, all(isnan(tr.wd([3 4], 2))) && all(tr.gap([3 4], 2)));
verifyTrue(tc, all(tr.cost([3 4], 2) > 0.5));
verifyEqual(tc, isnan(tr.wd), isnan(q.B2_gate_wd));
f = ~isnan(tr.wd);
verifyLessThan(tc, max(abs(tr.wd(f) - q.B2_gate_wd(f))), 1e-8);
verifyLessThan(tc, max(abs(tr.cost([3 4], 2) - q.B2_gate_cost([3 4], 2))), 1e-8);
end

function testClosureStrict(tc)
q = tc.TestData.r112; C = rec112(tc, 'C');
tr = ncvd.track(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M);
[v0, c] = ncvd.deflate(squeeze(tr.shape(:,1,:)));
verifyEmpty(tc, ncvd.closure(v0, c, 1));                       % 1.1.1 behavior
verifyError(tc, @() ncvd.closure(v0, c, 1, Strict=true), 'ncvd:closure');
K3 = 2*eye(3) - diag(ones(2,1), 1) - diag(ones(2,1), -1); [Phi, ~] = eig(K3);
P = ncvd.closure(Phi, c, 1, 0.5);
verifyEqual(tc, ncvd.closure(Phi, c, 1, 0.5, [], Strict=true), P);
Vp = null(c.');
verifyEmpty(tc, ncvd.closure(Vp, c, 1));
verifyError(tc, @() ncvd.closure(Vp, c, 1, Strict=true), 'ncvd:closure');
verifyError(tc, @() ncvd.closure(Phi, c, 2, Strict=true), 'ncvd:closure');
end

function testFamilyClosure(tc)
q = tc.TestData.r112; C = rec112(tc, 'C');
tr = ncvd.track(C{:}, Fs=20, NWindows=10, NTrack=1, M=q.C_M, Rank="auto");
sh = squeeze(tr.shape(:,1,:));
[~, ~, mag] = ncvd.deflate(sh);
[P, sv] = ncvd.familyClosure(sh, q.C_family_tol);
verifyEqual(tc, P, double(q.C_family_P));
verifyLessThan(tc, abs(sv(1) - mag), 1e-14);
verifyLessThan(tc, abs(sv(1) - q.C_family_sv(1)), 1e-8);
verifyEqual(tc, ncvd.familyClosure(sh, 2*mag), 1);
end
