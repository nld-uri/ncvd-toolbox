function [w, mpc] = realify(v)
%NCVD.REALIFY  Phase-rotate a complex mode to maximum collinearity.
%   Returns the unit real shape (largest entry positive) and the modal phase
%   collinearity MPC (1 for a normal mode, toward 0 for a complex mode).
v = v(:);
R = real(v) - mean(real(v)); I = imag(v) - mean(imag(v));
S = [R'*R, R'*I; R'*I, I'*I];
[E, D] = eig(S); [ev, o] = sort(diag(D)); E = E(:,o);
mpc = ((ev(2) - ev(1))/max(ev(2) + ev(1), 1e-300))^2;
th = atan2(E(2,2), E(1,2));
w = real(v*exp(-1i*th));
nr = norm(w);
if nr < 1e-300, w = real(v); nr = max(norm(w), 1e-300); end
w = w/nr;
[~, k] = max(abs(w)); w = w*sign(w(k));
end
