function rec = ensemble(records, opts)
%NCVD.ENSEMBLE  Joint NCVD of several records (different fs, lengths, noise).
%
%   rec = ncvd.ensemble({ {d1,v1,a1}, {d2,v2,a2}, ... }, Weights="equal")
%
%   Rows of all records are stacked and decomposed once: least-squares
%   fusion of the continuous-time state-rate relation. Weights: "equal"
%   (each record counts equally), "samples", "snr", or a vector.
%   rec.blocks{i} gives record i's rows in rec.U.
arguments
    records cell
    opts.Weights = "equal"
    opts.Center (1,1) logical = false
    opts.Rank = []
    opts.Operator (1,1) string = "ls"
    opts.TimeScale = []
end
N = numel(records); Y1 = cell(N,1); Y2 = cell(N,1); m = zeros(N,1);
blocks = cell(N,1); row = 0; nst = size(records{1}{1}, 2);
for i = 1:N
    [d, v, a] = records{i}{:};
    A1 = [d v]; A2 = [v a];
    if opts.Center, A1 = A1 - mean(A1,1); A2 = A2 - mean(A2,1); end
    Y1{i} = A1; Y2{i} = A2; m(i) = size(A1,1);
    blocks{i} = row + (1:m(i)); row = row + m(i);
end
W = opts.Weights;
if isstring(W) || ischar(W)
    switch string(W)
        case "equal", w = 1./m;
        case "samples", w = ones(N,1);
        case "snr"
            nv = zeros(N,1);
            for i = 1:N
                Bi = Y1{i}\Y2{i};
                nv(i) = mean((Y2{i} - Y1{i}*Bi).^2, 'all')/max(mean(Y2{i}.^2,'all'), 1e-300);
            end
            w = 1./(m.*max(nv, 1e-16));
        otherwise, error('ncvd:weights', 'unknown weights');
    end
else
    w = W(:);
end
w = w/max(w);
rw = repelem(w, m);
rec = ncvd.decomposePair(vertcat(Y1{:}), vertcat(Y2{:}), Weights=rw, ...
    StructuredSplit=nst, Rank=opts.Rank, Operator=opts.Operator, TimeScale=opts.TimeScale);
rec.blocks = blocks; rec.recordWeights = w;
end
