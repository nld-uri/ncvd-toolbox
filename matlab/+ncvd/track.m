function tr = track(disp, vel, acc, opts)
%NCVD.TRACK  Sliding-window NCVD with resonance tracking (backbone curves).
%
%   tr = ncvd.track(d, v, a, Fs=fs, Length=L, Step=S, NTrack=k, M=M)
%   tr = ncvd.track(d, v, a, Fs=fs, NWindows=N)
%
%   Consecutive windows are matched by minimising (1 - MAC) +
%   WFreq*|log(w/w_prev)| (matchpairs). Returns a struct with
%   (nWindows x k) fields wn, zeta, wd, lam, mpc, amp, energy, macPrev,
%   (nWindows x k x n) shape (real) and complexShape, and res, edges, t.
%   With M, amp is the mass-normalised modal amplitude and energy
%   0.5*(wn*amp)^2; without M, amp is the RMS displacement contribution of
%   the pair at station RefStation.
arguments
    disp (:,:) double
    vel (:,:) double
    acc (:,:) double
    opts.Fs (1,1) double = 1
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
end
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
