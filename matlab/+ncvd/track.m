function tr = track(disp, vel, acc, opts)
%NCVD.TRACK  Sliding-window NCVD with resonance tracking (backbone curves).
%
%   tr = ncvd.track(d, v, a, Fs=fs, Length=L, Step=S, NTrack=k, M=M)
%   tr = ncvd.track(d, v, a, Fs=fs, NWindows=N)
%
%   With every option at its default this is EXACTLY the 1.1.1 routine (the
%   local function trackV111 below, unchanged): same seeding, rank behavior
%   and ordering. Consecutive windows are matched by minimising (1 - MAC) +
%   WFreq*|log(w/w_prev)| (matchpairs). Returns a struct with
%   (nWindows x k) fields wn, zeta, wd, lam, mpc, amp, energy, macPrev,
%   (nWindows x k x n) shape (real) and complexShape, and res, edges, t.
%   With M, amp is the mass-normalised modal amplitude and energy
%   0.5*(wn*amp)^2; without M, amp is the RMS displacement contribution of
%   the pair at station RefStation.
%
%   Known weaknesses of the default (1.1.1) path, kept so that published
%   results reproduce: seeding takes the lowest-frequency pole with Im > 0
%   of window 1 (a static-offset pole with Center=false, or a null-space
%   eigenvalue of a rank-deficient window); Rank=[] keeps every direction of
%   [d v]; a track left without a candidate stores a zero reference, so it
%   is not recovered later; the zero-frequency pole is not reported.
%
%   Opt-in options (1.1.2; defaults reproduce 1.1.1; Python names in [])
%     Fs         []        sampling rate; [] means 1 for the center times.
%                          Required with Families=true.
%     Rank       []        [] all directions | integer | "auto": min over
%                          windows of ncvd.numericalRank(svd([d v]_w),
%                          RankRtol, RankGap), returned as tr.rank with
%                          tr.rankWindows, tr.rankRule, tr.sv
%     Seed       "lowest"  "lowest" (1.1.1) | "energy" (largest modal energy
%                          mean_t v_i'*M*v_i of the pole's own velocity
%                          contribution, M = I without a mass matrix) |
%                          "participation" (mean_t d_i'*M*d_i) | vector of
%                          target damped frequencies (nearest distinct poles)
%     Families   false     true: harmonic family (Im > 0 and at least
%                          MinCycles damped cycles per window,
%                          |Im lam|*Tw/(2 pi) >= MinCycles, Tw = samples/Fs)
%                          and zero-frequency family (real poles and slower
%                          pairs: the static offset). Only harmonic poles are
%                          tracked as resonances; the zero-frequency family is
%                          reported in tr.zero and never seeded as a master.
%     Matching   "legacy"  "identity": assignment between tracks and
%                          candidates on (1 - MAC) + WFreq*|log(|lam|/|lam_ref|)|
%                          against each track's LAST ASSIGNED window; a track
%                          with no candidate or a cost above MaxCost is a gap
%                          (NaN, tr.gap true); tr.cost holds every proposed
%                          cost. Required by any other Seed or Families=true.
%     MaxCost    []        largest accepted assignment cost ([] = no gate)
%     NZero      1         zero-frequency tracks ([] = all in the seed window)
%     MinCycles  1         cycle threshold of the family split
%     WFreqZero  0.1       frequency weight of the zero-family matching cost
%     RankRtol   1e-12     relative threshold of Rank="auto"
%     RankGap    []        gap ratio of Rank="auto"
%   Opt-in calls add cost, gap, nHarmonic, seed, seedWindow (1-based),
%   seedScore, matching, maxCost, familyRule, zero and the rank fields. See
%   also ncvd.windowed, which is track with every 1.1.2 remedy switched on.
arguments
    disp (:,:) double
    vel (:,:) double
    acc (:,:) double
    opts.Fs = []
    opts.NWindows = []
    opts.Length = []
    opts.Step = []
    opts.NTrack = []
    opts.M = []
    opts.Center (1,1) logical = true
    opts.Rank = []
    opts.Operator (1,1) string = "ls"
    opts.TimeScale = "auto"
    opts.WFreq (1,1) double = 1
    opts.RefStation = []
    opts.Seed = "lowest"
    opts.Matching (1,1) string = "legacy"
    opts.Families (1,1) logical = false
    opts.MaxCost = []
    opts.NZero = 1
    opts.MinCycles (1,1) double = 1
    opts.WFreqZero (1,1) double = 0.1
    opts.RankRtol (1,1) double = 1e-12
    opts.RankGap = []
end
targets = [];
if isstring(opts.Seed) || ischar(opts.Seed)
    seed = string(opts.Seed);
    if ~ismember(seed, ["lowest" "energy" "participation"])
        error('ncvd:seed', 'Seed must be "lowest", "energy", "participation" or a vector of frequencies');
    end
else
    targets = double(opts.Seed(:)); seed = "targets";
    if isempty(targets), error('ncvd:seed', 'Seed targets must be a non-empty vector'); end
end
if ~ismember(opts.Matching, ["legacy" "identity"])
    error('ncvd:matching', 'Matching must be "legacy" or "identity"');
end
rankAuto = false;
if isstring(opts.Rank) || ischar(opts.Rank)
    if string(opts.Rank) ~= "auto"
        error('ncvd:rank', 'Rank must be [], an integer or "auto"');
    end
    rankAuto = true;
end
newPath = opts.Families || opts.Matching == "identity" || seed ~= "lowest";
if opts.Families && isempty(opts.Fs)
    error('ncvd:fs', 'Families=true needs Fs (the sampling rate) for the cycle criterion');
end
if newPath && opts.Matching == "legacy"
    error('ncvd:matching', ['the 1.1.1 matching (Matching="legacy") is available only with ' ...
        'Seed="lowest" and Families=false; pass Matching="identity"']);
end
if ~isempty(opts.MaxCost) && opts.Matching ~= "identity"
    error('ncvd:matching', 'MaxCost needs Matching="identity"');
end
info = [];
if rankAuto || newPath
    [opts.Rank, info] = rankRule(disp, vel, opts, rankAuto);
end
if ~newPath
    o = struct('Fs', 1, 'NWindows', opts.NWindows, 'Length', opts.Length, 'Step', opts.Step, ...
        'NTrack', opts.NTrack, 'M', opts.M, 'Center', opts.Center, 'Rank', opts.Rank, ...
        'Operator', opts.Operator, 'TimeScale', opts.TimeScale, 'WFreq', opts.WFreq, ...
        'RefStation', opts.RefStation);
    if ~isempty(opts.Fs), o.Fs = double(opts.Fs); end   % 1.1.1 validated Fs as (1,1) double
    tr = trackV111(disp, vel, acc, o);
else
    tr = track112(disp, vel, acc, opts, seed, targets);
end
if ~isempty(info)
    f = fieldnames(info);
    for i = 1:numel(f), tr.(f{i}) = info.(f{i}); end
end
end

% ------------------------------------------------------------------ 1.1.1 routine
function tr = trackV111(disp, vel, acc, opts)
% The body of ncvd.track in release 1.1.1, unchanged (opts already validated).
[m, n] = size(disp);
if isempty(opts.RefStation), opts.RefStation = n; end
ed = ncvd.windowEdges(m, opts.NWindows, opts.Length, opts.Step);
nw = size(ed,1);
res = nan(nw,1); prev = [];
for w = 1:nw
    s = ed(w,1):ed(w,2);
    r = ncvd.decompose(disp(s,:), vel(s,:), acc(s,:), Center=opts.Center, ...
        Rank=opts.Rank, Operator=opts.Operator, TimeScale=opts.TimeScale);
    res(w) = r.res;
    idx = ncvd.physicalModes(r.lam);
    if w == 1
        if isempty(opts.NTrack), k = numel(idx); else, k = min(opts.NTrack, numel(idx)); end
        z = nan(nw,k);
        tr = struct('wn',z,'zeta',z,'wd',z,'mpc',z,'amp',z,'energy',z,'macPrev',z, ...
            'lam',complex(z,z),'shape',nan(nw,k,n),'complexShape',complex(nan(nw,k,n)));
    end
    nc = numel(idx);
    cl = r.lam(idx); cphi = r.Psi(1:n, idx); csh = zeros(n, nc); cmpc = zeros(nc,1);
    for j = 1:nc, [csh(:,j), cmpc(j)] = ncvd.realify(cphi(:,j)); end
    if isempty(prev)
        sel = (1:min(k, nc))';
    else
        cost = (1 - (prev.sh'*csh).^2) + opts.WFreq*abs(log(abs(cl).'./abs(prev.lam)));
        P = matchpairs(cost, 1e6); sel = zeros(k,1); sel(P(:,1)) = P(:,2);
    end
    if opts.Center, dX = disp(s,:) - mean(disp(s,:),1); else, dX = disp(s,:); end
    cur = struct('lam', zeros(k,1), 'sh', zeros(n,k));
    for j = 1:k
        ci = sel(j); if ci == 0, continue; end
        lam = cl(ci);
        tr.lam(w,j) = lam; tr.wn(w,j) = abs(lam); tr.wd(w,j) = abs(imag(lam));
        tr.zeta(w,j) = -real(lam)/abs(lam);
        tr.shape(w,j,:) = csh(:,ci); tr.complexShape(w,j,:) = cphi(:,ci); tr.mpc(w,j) = cmpc(ci);
        if ~isempty(prev), tr.macPrev(w,j) = (prev.sh(:,j)'*csh(:,ci))^2; end
        if ~isempty(opts.M)
            vm = csh(:,ci)/sqrt(csh(:,ci)'*opts.M*csh(:,ci));
            qa = sqrt(2)*std(dX*(opts.M*vm), 1);
            tr.amp(w,j) = qa; tr.energy(w,j) = 0.5*(abs(lam)*qa)^2;
        else
            u = 2*abs(r.U(:, idx(ci)));
            tr.amp(w,j) = sqrt(mean(u.^2))*abs(cphi(opts.RefStation, ci));
        end
        cur.lam(j) = lam; cur.sh(:,j) = csh(:,ci);
    end
    prev = cur;
end
tr.res = res; tr.edges = ed; tr.t = (ed(:,1) - 1 + ed(:,2))/2/opts.Fs;
end

% ------------------------------------------------------------------ 1.1.2 routine
function [rank, info] = rankRule(disp, vel, opts, rankAuto)
ed = ncvd.windowEdges(size(disp,1), opts.NWindows, opts.Length, opts.Step);
nw = size(ed,1);
rw = zeros(nw,1); sv = [];
for w = 1:nw
    s = ed(w,1):ed(w,2);
    Y1 = [disp(s,:), vel(s,:)];
    if opts.Center, Y1 = Y1 - mean(Y1,1); end
    sw = svd(Y1);
    if isempty(sv), sv = nan(nw, numel(sw)); end
    sv(w,1:numel(sw)) = sw.';
    rw(w) = ncvd.numericalRank(sw, opts.RankRtol, opts.RankGap);
end
info = struct('rankWindows', rw, 'sv', sv);
if rankAuto
    rank = min(rw);
    if rank < 1, error('ncvd:rank', 'a window has numerical rank 0'); end
    info.rank = rank;
    info.rankRule = struct('rule', "min over windows", 'rtol', opts.RankRtol, 'gap', opts.RankGap);
else
    rank = opts.Rank;
    info.rank = rank;
    if isempty(rank), rl = "all directions (1.1.1 default)"; else, rl = "fixed"; end
    info.rankRule = struct('rule', rl, 'rtol', opts.RankRtol, 'gap', opts.RankGap);
end
end

function [harm, zero] = candidates(r, n, Mm, Tw, families, minCycles)
% Split the poles of one window into harmonic and zero-frequency candidates.
tol = 1e-9;
proto = struct('lam', 0, 'phi', zeros(n,1), 'shape', zeros(n,1), 'mpc', 0, ...
    'u', zeros(size(r.U,1),1), 'energy', 0, 'part', 0);
harm = proto([]); zero = proto([]);
for i = 1:numel(r.lam)
    l = r.lam(i);
    isReal = abs(imag(l)) <= tol*max(abs(l), 1e-300);
    up = imag(l) > tol*max(abs(l), 1e-300);
    if ~(isReal || up), continue; end
    if isReal, f = 1; else, f = 2; end
    phi = r.Psi(1:n, i);
    [sh, mp] = ncvd.realify(phi);
    dcon = f*real(r.U(:,i)*phi.');
    vcon = f*real(r.U(:,i)*r.Psi(n+1:2*n, i).');
    c = proto;
    c.lam = l; c.phi = phi; c.shape = sh; c.mpc = mp; c.u = 2*abs(r.U(:,i));
    c.energy = mean(sum((vcon*Mm).*vcon, 2));
    c.part = mean(sum((dcon*Mm).*dcon, 2));
    if families
        if isReal || abs(imag(l))*Tw/(2*pi) < minCycles
            zero(end+1) = c; %#ok<AGROW>
        else
            harm(end+1) = c; %#ok<AGROW>
        end
    elseif up
        harm(end+1) = c; %#ok<AGROW>
    end
end
if ~isempty(harm), [~, o] = sort(imag([harm.lam])); harm = harm(o); end
if ~isempty(zero), [~, o] = sort(abs([zero.lam])); zero = zero(o); end
end

function [assign, cost, sw, share] = follow(cands, kReq, rule, targets, wFreq, maxCost)
% Seed and follow one family with identity-preserving assignment.
% assign(w, j) = index into cands{w} of track j's pole (0 = gap).
nw = numel(cands);
sw = find(cellfun(@numel, cands) > 0, 1);
if isempty(sw)
    assign = zeros(nw, 0); cost = nan(nw, 0); share = zeros(0, 1); return
end
c0 = cands{sw};
if rule == "participation" || rule == "zero", score = [c0.part].'; else, score = [c0.energy].'; end
share = score/max(sum(score), 1e-300);
if ~isempty(targets)
    C0 = abs(abs(imag([c0.lam])) - targets(:));
    P = matchpairs(C0, 1e6);
    [~, o] = sort(P(:,1)); sel = P(o,2).';
else
    k = numel(c0); if ~isempty(kReq), k = min(kReq, k); end
    if rule == "lowest"
        sel = 1:k;
    else
        [~, o] = sort(-score); top = o(1:k).';
        if rule == "zero", key = abs([c0(top).lam]); else, key = imag([c0(top).lam]); end
        [~, o2] = sort(key); sel = top(o2);
    end
end
k = numel(sel);
assign = zeros(nw, k); assign(sw, :) = sel;
ref = c0(sel);
cost = nan(nw, k);
share = share(sel);
for w = sw+1:nw
    cw = cands{w};
    if isempty(cw) || k == 0, continue; end
    C = zeros(k, numel(cw));
    for j = 1:k
        for i = 1:numel(cw)
            C(j,i) = (1 - (cw(i).shape.'*ref(j).shape)^2) + wFreq*abs(log( ...
                max(abs(cw(i).lam), 1e-300)/max(abs(ref(j).lam), 1e-300)));
        end
    end
    P = matchpairs(C, 1e6);
    for q = 1:size(P,1)
        j = P(q,1); i = P(q,2);
        cost(w,j) = C(j,i);
        if ~isempty(maxCost) && C(j,i) > maxCost, continue; end   % gap, reference kept
        assign(w,j) = i; ref(j) = cw(i);
    end
end
end

function out = fillTracks(cands, assign, n, dXs, M, refStation, harmonic)
[nw, k] = size(assign);
z = nan(nw, k);
out = struct('wn', z, 'zeta', z, 'wd', z, 'mpc', z, 'amp', z, 'energy', z, 'macPrev', z, ...
    'lam', complex(z, z), 'shape', nan(nw, k, n), 'complexShape', complex(nan(nw, k, n)), ...
    'gap', true(nw, k));
last = cell(1, k);
for w = 1:nw
    for j = 1:k
        ci = assign(w,j); if ci == 0, continue; end
        c = cands{w}(ci); lam = c.lam;
        out.gap(w,j) = false;
        out.lam(w,j) = lam; out.wn(w,j) = abs(lam); out.wd(w,j) = abs(imag(lam));
        out.zeta(w,j) = -real(lam)/abs(lam);
        out.shape(w,j,:) = c.shape; out.complexShape(w,j,:) = c.phi; out.mpc(w,j) = c.mpc;
        if ~isempty(last{j}), out.macPrev(w,j) = (c.shape.'*last{j})^2; end
        last{j} = c.shape;
        if ~harmonic, out.amp(w,j) = sqrt(c.part); continue; end
        if ~isempty(M)
            vm = c.shape/sqrt(c.shape.'*M*c.shape);
            qa = sqrt(2)*std(dXs{w}*(M*vm), 1);
            out.amp(w,j) = qa; out.energy(w,j) = 0.5*(abs(lam)*qa)^2;
        else
            out.amp(w,j) = sqrt(mean(c.u.^2))*abs(c.phi(refStation));
        end
    end
end
end

function tr = track112(disp, vel, acc, opts, seed, targets)
[m, n] = size(disp);
if isempty(opts.RefStation), opts.RefStation = n; end
if isempty(opts.Fs), fs = 1; else, fs = opts.Fs; end
ed = ncvd.windowEdges(m, opts.NWindows, opts.Length, opts.Step);
nw = size(ed,1);
if isempty(opts.M), Mm = eye(n); else, Mm = opts.M; end
Tw = (ed(:,2) - ed(:,1) + 1)/fs;
res = nan(nw,1); harm = cell(nw,1); zero = cell(nw,1); dXs = cell(nw,1);
for w = 1:nw
    s = ed(w,1):ed(w,2);
    r = ncvd.decompose(disp(s,:), vel(s,:), acc(s,:), Center=opts.Center, ...
        Rank=opts.Rank, Operator=opts.Operator, TimeScale=opts.TimeScale);
    res(w) = r.res;
    [harm{w}, zero{w}] = candidates(r, n, Mm, Tw(w), opts.Families, opts.MinCycles);
    if opts.Center, dXs{w} = disp(s,:) - mean(disp(s,:),1); else, dXs{w} = disp(s,:); end
end
[assign, cost, sw, score] = follow(harm, opts.NTrack, seed, targets, opts.WFreq, opts.MaxCost);
tr = fillTracks(harm, assign, n, dXs, opts.M, opts.RefStation, true);
tr.res = res; tr.edges = ed; tr.t = (ed(:,1) - 1 + ed(:,2))/2/fs;
tr.cost = cost; tr.nHarmonic = cellfun(@numel, harm); tr.seed = seed;
tr.seedWindow = sw; tr.seedScore = score; tr.matching = "identity"; tr.maxCost = opts.MaxCost;
if opts.Families
    [za, zc, zsw, zscore] = follow(zero, opts.NZero, "zero", [], opts.WFreqZero, opts.MaxCost);
    zo = fillTracks(zero, za, n, dXs, opts.M, opts.RefStation, false);
    zo = rmfield(zo, {'mpc', 'energy'});
    zo.cost = zc; zo.seedWindow = zsw; zo.seedScore = zscore;
    zo.nCandidates = cellfun(@numel, zero);
    tr.zero = zo;
    tr.familyRule = struct('minCycles', opts.MinCycles, 'realTol', 1e-9, ...
        'windowDuration', Tw, 'wFreqZero', opts.WFreqZero);
else
    tr.zero = []; tr.familyRule = [];
end
end
