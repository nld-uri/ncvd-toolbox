function b = filterBand(f)
%NCVD.FILTERBAND  Largest omega*dt below which the stencil symbol is one-to-one.
%   Two-point stencils: pi. Otherwise the first maximum of Im d(exp(i w))/fs.
if max(f.off) - min(f.off) == 1, b = pi; return; end
w = linspace(1e-4, pi, 20001);
g = imag(ncvd.filterSymbol(f, exp(1i*w), 1));
k = find(diff(g) <= 0, 1);
if isempty(k), b = pi; else, b = w(k); end
end
