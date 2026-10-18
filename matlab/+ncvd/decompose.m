function rec = decompose(disp, vel, acc, opts)
%NCVD.DECOMPOSE  Canonical NCVD of a state record.
%
%   rec = ncvd.decompose(disp, vel, acc)
%   rec = ncvd.decompose(disp, vel, acc, Name=Value, ...)
%
%   X1 = [disp vel] = U*Phi',   X2 = [vel acc] = U*diag(lam)*Phi' + R.
%   disp, vel, acc are (m x n): m snapshots, n stations.
%
%   Options as in ncvd.decomposePair, plus
%       Structured  true   keep the kinematic block exact (TLS, BCLS)
%       NoiseVar    []     "auto" or 2n noise variances of [disp vel] (BCLS)
%       Fs          []     sampling rate, needed for NoiseVar="auto"
%
%   Example
%       rec = ncvd.decompose(d, v, a);
%       mp  = ncvd.modalParameters(rec.lam);
%
%   See also ncvd.decomposePair, ncvd.modalParameters, ncvd.track.
arguments
    disp (:,:) double
    vel (:,:) double
    acc (:,:) double
    opts.Rank = []
    opts.Center (1,1) logical = false
    opts.Operator (1,1) string = "ls"
    opts.Forcing = []
    opts.TimeScale = []
    opts.Weights = []
    opts.Sort (1,1) logical = true
    opts.Structured (1,1) logical = true
    opts.NoiseVar = []
    opts.Fs = []
    opts.Readout (1,1) string = "eig"
    opts.LegacyBalanced (1,1) logical = false
end
if ~isequal(size(disp), size(vel), size(acc))
    error('ncvd:size', 'disp, vel and acc must have the same size');
end
n = size(disp, 2);
nv = opts.NoiseVar;
if isstring(nv) || ischar(nv)
    if isempty(opts.Fs), error('ncvd:fs', 'NoiseVar="auto" needs Fs'); end
    nv = ncvd.noiseVariance([disp vel], opts.Fs);
end
split = [];
if opts.Structured, split = n; end
rec = ncvd.decomposePair([disp vel], [vel acc], Rank=opts.Rank, Center=opts.Center, ...
    Operator=opts.Operator, Forcing=opts.Forcing, TimeScale=opts.TimeScale, ...
    Weights=opts.Weights, Sort=opts.Sort, StructuredSplit=split, NoiseVar=nv, ...
    Readout=opts.Readout, LegacyBalanced=opts.LegacyBalanced);
end
