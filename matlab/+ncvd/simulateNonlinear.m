function [d, v, a] = simulateNonlinear(M, C, K, fnl, x0, t, relTol, absTol)
%NCVD.SIMULATENONLINEAR  M q'' + C q' + K q + fnl(q) = 0 by ode89 (ode45 if
%   ode89 is unavailable). Returns d, v and acc from the equations of motion.
if nargin < 7, relTol = 1e-10; end
if nargin < 8, absTol = 1e-12; end
n = size(M,1);
rhs = @(~, x) [x(n+1:end); -M\(C*x(n+1:end) + K*x(1:n) + fnl(x(1:n)))];
o = odeset('RelTol', relTol, 'AbsTol', absTol);
if exist('ode89', 'file'), [~, X] = ode89(rhs, t, x0(:), o); else, [~, X] = ode45(rhs, t, x0(:), o); end
d = X(:,1:n); v = X(:,n+1:end);
a = -(v*C.' + d*K.' + fnl(d))/M.';
end
