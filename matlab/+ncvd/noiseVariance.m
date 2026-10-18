function nv = noiseVariance(X, fs, band)
%NCVD.NOISEVARIANCE  Per-column white-noise variance from the PSD plateau
%   between band(1)*fs and band(2)*fs (default 0.40-0.49 fs). Valid when the
%   sampling rate is well above the highest mode. Base MATLAB (Welch with
%   Hann windows, 50% overlap, linear detrend).
if nargin < 3, band = [0.40 0.49]; end
X = X - mean(X,1);
m = size(X,1); L = min(m, 4096); hop = floor(L/2);
w = 0.5 - 0.5*cos(2*pi*(0:L-1)'/L);             % periodic Hann (as scipy)
U = sum(w.^2);
nseg = floor((m - L)/hop) + 1;
P = 0;
t = (0:L-1)';
for k = 0:nseg-1
    seg = X(k*hop + (1:L), :);
    p = [t ones(L,1)]\seg; seg = seg - [t ones(L,1)]*p;   % linear detrend
    Fk = fft(seg.*w);
    P = P + abs(Fk(1:floor(L/2)+1, :)).^2;
end
P = P/(nseg*fs*U);
P(2:end-1, :) = 2*P(2:end-1, :);                % one-sided
f = (0:floor(L/2))'*fs/L;
mk = f >= band(1)*fs & f <= band(2)*fs;
if ~any(mk), mk = f >= 0.8*f(end); end
nv = median(P(mk, :), 1)*(fs/2);
end
