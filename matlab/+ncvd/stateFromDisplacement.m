function [d, v, a, f, k0] = stateFromDisplacement(q, fs, method, varargin)
%NCVD.STATEFROMDISPLACEMENT  Consistent (disp, vel, acc) from displacement.
%   v = D q and a = D(D q) with the same stencil, aligned with d. Pass the
%   identified poles through ncvd.correctPoles(lam, f, fs).
%   method: name (see ncvd.derivFilter) or a filter struct.
if nargin < 3, method = "central"; end
if isstruct(method), f = method; else, f = ncvd.derivFilter(method, varargin{:}); end
[v, k1] = ncvd.applyFilter(q, f, fs);
[a, k2] = ncvd.applyFilter(v, f, fs);
k0 = k1 + k2; na = size(a,1);
d = q(k0 + (1:na), :);
v = v(k2 + (1:na), :);
end
