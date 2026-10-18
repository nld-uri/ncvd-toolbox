function w = bandFrequency(sig, dt, lo, hi)
%NCVD.BANDFREQUENCY  Dominant angular frequency of sig in (lo, hi), Hann window
%   and parabolic peak refinement. Use it to validate identified poles.
s = sig(:) - mean(sig);
N = numel(s); h = 0.5 - 0.5*cos(2*pi*(0:N-1)'/(N-1));
S = abs(fft(s.*h)); S = S(1:floor(N/2)+1);
f = (0:floor(N/2))'/(N*dt)*2*pi;
mk = f > lo & f < hi;
if ~any(mk), w = NaN; return; end
ii = find(mk); [~, k] = max(S(mk)); i = ii(k);
if i == 1 || i == numel(S), w = f(i); return; end
d = 0.5*(S(i-1) - S(i+1))/(S(i-1) - 2*S(i) + S(i+1));
w = (i - 1 + d)*(f(2) - f(1));
end
