function [M, C, K, fnl] = duffingChain(n, m, k, c, k3)
%NCVD.DUFFINGCHAIN  Fixed-fixed chain with cubic springs.
%   k3: [element coefficient] rows (element 1 grounds mass 1, element n+1
%   grounds mass n) or a scalar for every element. fnl(q) is the nonlinear
%   restoring force for q given as rows (m x n) or a column.
if nargin < 5, k3 = [1 1]; end
[M, C, K] = ncvd.chain(n, m, k, c);
if isscalar(k3), k3 = [(1:n+1)' k3*ones(n+1,1)]; end
fnl = @(q) force(q, k3, n);
end
function f = force(q, k3, n)
col = size(q,2) == 1 && size(q,1) == n;
if col, q = q.'; end
f = zeros(size(q));
for r = 1:size(k3,1)
    e = k3(r,1); a = k3(r,2);
    if e == 1
        f(:,1) = f(:,1) + a*q(:,1).^3;
    elseif e == n+1
        f(:,n) = f(:,n) + a*q(:,n).^3;
    else
        s = q(:,e) - q(:,e-1);
        f(:,e) = f(:,e) + a*s.^3; f(:,e-1) = f(:,e-1) - a*s.^3;
    end
end
if col, f = f.'; end
end
