function mp = modalParameters(lam)
%NCVD.MODALPARAMETERS  wn = |lam|, wd = |Im lam|, zeta = -Re lam/|lam|, fn, fd.
lam = lam(:);
mp.wn = abs(lam); mp.wd = abs(imag(lam));
mp.zeta = -real(lam)./mp.wn;
mp.fn = mp.wn/(2*pi); mp.fd = mp.wd/(2*pi);
end
