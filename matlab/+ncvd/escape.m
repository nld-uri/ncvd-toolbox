function e = escape(Vkept, corrections)
%NCVD.ESCAPE  Norm of each unit correction outside span(Vkept).
[Qk, ~] = qr(Vkept, 0);
e = vecnorm(corrections - Qk*(Qk'*corrections), 2, 1);
end
