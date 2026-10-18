function y = addNoise(x, snr, opts)
%NCVD.ADDNOISE  White Gaussian noise at signal-to-noise power ratio snr.
%   y = x + sqrt(Ps/SNR)*randn, Ps the mean power (per channel with
%   PerChannel=true). snr is linear unless DB=true. Inf returns x.
arguments
    x double
    snr (1,1) double
    opts.DB (1,1) logical = false
    opts.PerChannel (1,1) logical = false
    opts.Stream = []
end
if isinf(snr), y = x; return; end
if opts.DB, r = 10^(snr/10); else, r = snr; end
if opts.PerChannel, Ps = mean(x.^2, 1); else, Ps = mean(x.^2, 'all'); end
if isempty(opts.Stream), z = randn(size(x)); else, z = randn(opts.Stream, size(x)); end
y = x + sqrt(Ps/r).*z;
end
