function rom = project(M, K, fnl, P, C)
%NCVD.PROJECT  Galerkin projection on a constant basis P, q = P*eta.
Pinv = pinv(P);
rom.Mr = Pinv*M*P; rom.Kr = Pinv*K*P; rom.Pinv = Pinv;
rom.fnlr = @(eta) Pinv*fnl(P*eta);
if nargin > 4 && ~isempty(C), rom.Cr = Pinv*C*P; else, rom.Cr = []; end
end
