function out = gcvd(disp, vel, acc, quotient, center)
%NCVD.GCVD  GCVD as published (Li, Stein & Chelidze 2025, Eqs. 11-14:
%   W = V^-H, S2 = S1*pinv(P1)*P2): shapes V, values
%   diag(S2)./s1. For benchmarking only.
%   quotient: "balanced" (Eq. 16), "forward" (Eq. 10), "forcing" (Eq. 43)
if nargin < 4, quotient = "balanced"; end
if nargin < 5, center = false; end
Y1 = [disp vel]; Y2 = [vel acc];
if center, Y1 = Y1 - mean(Y1,1); Y2 = Y2 - mean(Y2,1); end
K11 = Y1'*Y1; K21 = Y2'*Y1;
switch string(quotient)
    case "balanced", R = ncvd.balancedQuotient(Y1, Y2);
    case "forward",  R = K21/K11;
    case "forcing",  R = 0.5*(K11/K21 + K21/K11);
    otherwise, error('ncvd:quotient', 'unknown quotient');
end
[V, ~] = eig(R);
W = inv(V)';
P1 = Y1*W; P2 = Y2*W;
s1 = vecnorm(P1, 2, 1).';
S2 = diag(s1)*pinv(P1)*P2;
out = struct('lam', diag(S2)./s1, 'Phi', V, 'U', P1./s1.', 'S2', S2);
end
