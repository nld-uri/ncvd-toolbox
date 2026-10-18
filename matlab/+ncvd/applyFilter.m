function [Dq, k0] = applyFilter(q, f, fs)
%NCVD.APPLYFILTER  Apply a derivative stencil along dimension 1. Dq(i,:)
%   belongs to sample k0 + i (1-based: row k0+i of q); only full stencils kept.
lo = -min(min(f.off), 0); hi = max(max(f.off), 0);
m = size(q,1);
k = (lo+1:m-hi)';
Dq = zeros(numel(k), size(q,2));
for j = 1:numel(f.c)
    Dq = Dq + f.c(j)*q(k + f.off(j), :);
end
Dq = fs*Dq; k0 = lo;
end
