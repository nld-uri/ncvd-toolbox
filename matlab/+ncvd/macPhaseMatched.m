function [m, matched] = macPhaseMatched(est, truth)
%NCVD.MACPHASEMATCHED  MAC after the phase matching of Eq. (55), Li et al. (2025).
Ph = [real(est(:)) imag(est(:))];
coef = pinv(Ph)*[real(truth(:)) imag(truth(:))];
P = Ph*coef; matched = P(:,1) + 1i*P(:,2);
m = ncvd.mac(matched, truth);
end
