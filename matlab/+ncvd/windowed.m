function tr = windowed(disp, vel, acc, fs, opts)
%NCVD.WINDOWED  Windowed free-decay identification, 1.1.2 protocol (new in 1.1.2).
%
%   tr = ncvd.windowed(d, v, a, fs, NWindows=N, NTrack=k, M=M)
%
%   ncvd.track with every 1.1.2 remedy switched on:
%       Center=false        single-equilibrium free decay; the static offset
%                           stays in the data and is reported in tr.zero
%       Rank="auto"         min over windows of the numerical rank (1e-12)
%       Seed="energy"       tracks seeded by modal energy in the first window
%       Families=true       harmonic and zero-frequency poles tracked separately
%       Matching="identity", MaxCost=0.5
%                           identity kept by shape and frequency continuity;
%                           gaps are NaN, never relabeled
%   Any option can be overridden; the others are those of ncvd.track. fs is
%   required. The 1.1.1 windowed path is ncvd.track(..., Center=false) with
%   its defaults, which is unchanged. Python: ncvd.windowed.
arguments
    disp (:,:) double
    vel (:,:) double
    acc (:,:) double
    fs (1,1) double
    opts.NWindows = []
    opts.Length = []
    opts.Step = []
    opts.NTrack = []
    opts.M = []
    opts.Center (1,1) logical = false
    opts.Rank = "auto"
    opts.Seed = "energy"
    opts.Families (1,1) logical = true
    opts.Matching (1,1) string = "identity"
    opts.MaxCost = 0.5
    opts.NZero = 1
    opts.Operator (1,1) string = "ls"
    opts.TimeScale = "auto"
    opts.WFreq (1,1) double = 1
    opts.RefStation = []
    opts.MinCycles (1,1) double = 1
    opts.WFreqZero (1,1) double = 0.1
    opts.RankRtol (1,1) double = 1e-12
    opts.RankGap = []
end
args = namedargs2cell(opts);
tr = ncvd.track(disp, vel, acc, args{:}, Fs=fs);
end
