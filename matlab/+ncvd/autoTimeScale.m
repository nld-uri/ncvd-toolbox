function c = autoTimeScale(Y1, Y2, margin, floorRel)
%NCVD.AUTOTIMESCALE  Time-unit factor c putting every pole of (Y1\Y2)/c at
%   modulus >= margin (default 2), where the balanced (Joukowski) map
%   mu = (lam + 1/lam)/2 is one-to-one. Rigid-body poles are ignored.
if nargin < 3, margin = 2; end
if nargin < 4, floorRel = 1e-8; end
a = abs(eig(Y1\Y2));
a = a(a > floorRel*max(max(a), 1e-300));
if isempty(a), c = 1; else, c = min(a)/margin; end
end
