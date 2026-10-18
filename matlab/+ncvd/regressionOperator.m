function B = regressionOperator(Y1, Y2, operator, split, noiseVar)
%NCVD.REGRESSIONOPERATOR  Operator B with Y2 ~ Y1*B.
%
%   "ls"   ordinary least squares, B = Y1\Y2 (canonical)
%   "tls"  total least squares on column-equilibrated data; measurement
%          noise only, not for unmeasured forcing
%   "bcls" bias-compensated LS, B = (K11 - m*diag(noiseVar)) \ (Y1'*Y2)
%   split  n for a [d v]/[v a] record: only the acceleration block is
%          regressed and v = v is kept exactly.
if nargin < 3 || isempty(operator), operator = "ls"; end
if nargin < 4, split = []; end
if nargin < 5, noiseVar = []; end
operator = lower(string(operator));
switch operator
    case "ls"
        B = Y1\Y2;
    case "bcls"
        if isempty(split) || isempty(noiseVar)
            error('ncvd:bcls', '"bcls" needs a structured record and NoiseVar');
        end
        n = split; m = size(Y1,1);
        K11 = Y1'*Y1 - m*diag(noiseVar(:));
        Ba = K11\(Y1'*Y2(:, n+1:end));
        top = zeros(size(Y1,2), n); top(n+1:end, :) = eye(n);
        B = [top Ba];
    case "tls"
        if ~isempty(split)
            n = split;
            Ba = tls(Y1, Y2(:, n+1:end));
            top = zeros(size(Y1,2), n); top(n+1:end, :) = eye(n);
            B = [top Ba];
        else
            B = tls(Y1, Y2);
        end
    otherwise
        error('ncvd:operator', 'Operator must be "ls", "tls" or "bcls"');
end
end

function X = tls(A, B)
sa = sqrt(mean(abs(A).^2, 1)); sa(sa == 0) = 1;
sb = sqrt(mean(abs(B).^2, 1)); sb(sb == 0) = 1;
[~, ~, W] = svd([A./sa, B./sb], 'econ');
p = size(A,2);
Xs = -W(1:p, p+1:end)/W(p+1:end, p+1:end);
X = (Xs./sa.').*sb;
end
