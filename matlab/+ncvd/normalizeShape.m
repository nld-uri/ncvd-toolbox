function v = normalizeShape(v)
%NCVD.NORMALIZESHAPE  Unit 2-norm, largest entry real and positive.
v = v(:); [~, k] = max(abs(v));
v = v*exp(-1i*angle(v(k))); v = v/max(norm(v), 1e-300);
end
