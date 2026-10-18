function out = itd(X, dt, rank)
%NCVD.ITD  Ibrahim time-domain method with two-block Hankel matrices.
%   H1 = [x_k x_{k+1}], H2 = [x_{k+2} x_{k+3}] (rows = time), H2 = H1*B;
%   lam = log(eig(B))/(2 dt), shapes inv(E)' (top block).
[m, p] = size(X);
H1 = [X(1:m-3,:) X(2:m-2,:)]; H2 = [X(3:m-1,:) X(4:m,:)];
if nargin > 2 && ~isempty(rank)
    [~, ~, Wf] = svd(H1, 'econ'); Wr = Wf(:,1:rank);
    H1 = H1*Wr; H2 = H2*Wr;
else
    Wr = [];
end
[E, Mu] = eig(H1\H2);
Phi = inv(E)';
if ~isempty(Wr), Phi = Wr*Phi; end
out = struct('lam', log(complex(diag(Mu)))/(2*dt), 'Phi', Phi(1:p,:));
end
