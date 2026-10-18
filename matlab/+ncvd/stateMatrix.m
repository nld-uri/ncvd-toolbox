function A = stateMatrix(M, C, K)
%NCVD.STATEMATRIX  A of x' = A x, x = [q; q'].
n = size(M,1);
A = [zeros(n) eye(n); -M\K -M\C];
end
