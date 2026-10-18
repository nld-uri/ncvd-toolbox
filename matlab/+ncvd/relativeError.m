function e = relativeError(est, truth)
%NCVD.RELATIVEERROR  ||a*est - truth||/||truth|| with the best complex scale a.
est = est(:); truth = truth(:);
a = (est'*truth)/max(real(est'*est), 1e-300);
e = norm(a*est - truth)/max(norm(truth), 1e-300);
end
