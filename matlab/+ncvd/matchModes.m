function j = matchModes(lamEst, PhiEst, lamRef, PhiRef, wFreq)
%NCVD.MATCHMODES  One-to-one match of estimated to reference modes.
%   Cost (1 - MAC) + wFreq*|log(|lamEst|/|lamRef|)| + 10*(opposite Im sign).
%   j(i) is the estimate matched to reference i (0 if none).
if nargin < 5, wFreq = 1; end
Mc = ncvd.macMatrix(PhiRef, PhiEst);
lr = abs(lamRef(:)); le = abs(lamEst(:)).';
cost = (1 - Mc) + wFreq*abs(log(max(le,1e-300)./max(lr,1e-300))) ...
     + 10*(sign(imag(lamRef(:))) ~= sign(imag(lamEst(:))).');
P = matchpairs(cost, 1e6);
j = zeros(numel(lamRef), 1); j(P(:,1)) = P(:,2);
end
