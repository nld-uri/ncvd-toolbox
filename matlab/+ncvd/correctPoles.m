function lam = correctPoles(lamEst, f, fs)
%NCVD.CORRECTPOLES  Continuous-time poles from poles identified with FIR rates.
%   Solves d(theta) = lamEst (polynomial roots), keeps the root with the
%   smallest |log theta|, returns fs*log(theta). Unique only below
%   ncvd.filterBand(f) (omega*dt); re-sort the result afterwards.
if ~isstruct(f), f = ncvd.derivFilter(f); end
lam = zeros(size(lamEst));
jmin = min(min(f.off), 0); deg = max(f.off) - jmin;
for i = 1:numel(lamEst)
    p = zeros(1, deg+1);                        % ascending powers
    for j = 1:numel(f.c)
        p(f.off(j) - jmin + 1) = p(f.off(j) - jmin + 1) + f.c(j);
    end
    p(-jmin + 1) = p(-jmin + 1) - lamEst(i)/fs;
    r = roots(fliplr(p));
    r = r(abs(r) > 1e-300);
    [~, k] = min(abs(log(complex(r))));
    lam(i) = log(complex(r(k)))*fs;
end
end
