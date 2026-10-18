function M = macMatrix(A, B)
%NCVD.MACMATRIX  MAC between every column of A and every column of B.
M = abs(A'*B).^2 ./ max(sum(abs(A).^2,1).'*sum(abs(B).^2,1), 1e-300);
end
