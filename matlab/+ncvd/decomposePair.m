function rec = decomposePair(Y1, Y2, opts)
%NCVD.DECOMPOSEPAIR  Canonical NCVD of a generic matrix pair, Y2 ~ Y1*B.
%
%   rec = ncvd.decomposePair(Y1, Y2)
%   rec = ncvd.decomposePair(Y1, Y2, Name=Value, ...)
%
%   Rows are time instants, columns coordinates. Returns a struct with
%       lam   (k x 1)  characteristic values; lam(i) pairs with Phi(:,i)
%       Phi   (p x k)  characteristic (regression) shapes, Y1 = U*Phi'
%       Psi   (p x k)  conj(Phi): eigenvector shapes, A*Psi(:,i) = lam(i)*Psi(:,i)
%       U     (m x k)  temporal basis (characteristic coordinates)
%       R     (m x p)  residual of the second identity
%       V, T, S2, B, G, s, share, condT, condV, resid, res, exact1, timeScale
%
%   Name-value options
%       Rank        []        whiten Y1 to this rank (does not change lam)
%       Center      false     remove column means
%       Operator    "ls"      "ls" | "tls" | "bcls"
%       Forcing     []        (m x q) known input; projected out (Frisch-Waugh)
%       TimeScale   []        [] | "auto" | c : time unit used for the balanced quotient
%       Weights     []        (m x 1) row weights
%       Sort        true      ascending |Im lam|, conjugate pairs adjacent
%       StructuredSplit []    n for a [d v]/[v a] record (keeps v = v exact)
%       NoiseVar    []        noise variances of Y1 columns (Operator "bcls")
%       Readout     "eig"     "eig" (canonical) | "diag" (published GCVD)
%       LegacyBalanced false  superseded shapes V with assignment heuristic
%
%   Algorithm: Q = (K21/K11 + K22/K12 + K11/K21 + K12/K22)/4, Kij = Yi'*Yj;
%   V = eig(Q); Xi = Y1/V'; Sigma2 = V'*B/V'; Sigma2 = T*Lam/T;
%   U = Xi*T; Phi = V/T'.  Equivalently, with E = (V')\T: B*E = E*Lam,
%   U = Y1*E, Phi = inv(E)'.  See the manual, Propositions 1-2.
arguments
    Y1 (:,:) double
    Y2 (:,:) double
    opts.Rank = []
    opts.Center (1,1) logical = false
    opts.Operator (1,1) string = "ls"
    opts.Forcing = []
    opts.TimeScale = []
    opts.Weights = []
    opts.Sort (1,1) logical = true
    opts.StructuredSplit = []
    opts.NoiseVar = []
    opts.Readout (1,1) string = "eig"
    opts.LegacyBalanced (1,1) logical = false
end
if size(Y1,1) ~= size(Y2,1)
    error('ncvd:size', 'Y1 and Y2 must have the same number of rows');
end
if opts.Center
    Y1 = Y1 - mean(Y1,1); Y2 = Y2 - mean(Y2,1);
end
F = opts.Forcing;
if ~isempty(opts.Weights)
    w = sqrt(opts.Weights(:));
    Y1 = Y1.*w; Y2 = Y2.*w;
end
if ~isempty(F)
    F = reshape(F, size(Y1,1), []);
    if opts.Center, F = F - mean(F,1); end
    if ~isempty(opts.Weights), F = F.*w; end
end
split = opts.StructuredSplit;
[~, Sall, Wall] = svd(Y1, 'econ'); sAll = diag(Sall);
if isempty(opts.Rank)
    Z1 = Y1; Z2 = Y2; back = [];
else
    r = opts.Rank;
    Wr = Wall(:,1:r); sr = sAll(1:r).';
    Z1 = Y1*(Wr./sr); Z2 = Y2*(Wr./sr);
    back = Wr.*sr;
    if opts.Operator == "tls", split = []; end
end
if ~isempty(F)
    Pf = pinv(F);
    P1 = Z1 - F*(Pf*Z1); P2 = Z2 - F*(Pf*Z2);
else
    P1 = Z1; P2 = Z2;
end
if isempty(opts.TimeScale)
    c = 1;
elseif isstring(opts.TimeScale) || ischar(opts.TimeScale)
    c = ncvd.autoTimeScale(P1, P2);
else
    c = double(opts.TimeScale);
end
resid = norm(P2 - P1*(P1\P2), 'fro')/max(norm(P2,'fro'), 1e-300);

Q = ncvd.balancedQuotient(P1, P2/c);
[V, ~] = eig(Q);
VmH = inv(V)';
Xi = Z1*VmH;                                    % Sigma_1 = I
nv = [];
if opts.Operator == "bcls"
    if ~isempty(opts.Rank) || ~isempty(F) || ~isempty(opts.Weights)
        error('ncvd:bcls', '"bcls" needs an unweighted, unwhitened, unforced record');
    end
    nv = opts.NoiseVar;
end
B = ncvd.regressionOperator(P1, P2, opts.Operator, split, nv);
S2 = V'*B*VmH;
G = [];
if ~isempty(F), G = F\(Z2 - Z1*B); end

if opts.Readout == "diag"
    lam = diag(S2); T = eye(numel(lam)); W = V; U = Xi; share = ones(numel(lam),1);
elseif opts.LegacyBalanced
    [T, L0] = eig(S2);
    [lam, share] = assignLegacy(diag(L0), T);
    W = V; U = Xi;
else
    [T, L0] = eig(S2);                          % Sigma_2 = T*Lam/T
    lam = diag(L0);
    W = V/T'; U = Xi*T;                         % Phi = V*T^-H, U = Xi*T
    share = ones(numel(lam),1);
end
if opts.Sort
    o = sortOrder(lam);
    lam = lam(o); W = W(:,o); U = U(:,o); share = share(o);
    if opts.Readout ~= "diag" && ~opts.LegacyBalanced, T = T(:,o); end
end
if isempty(back), Phi = W; else, Phi = back*W; end
fit = (U.*lam.')*W';
if ~isempty(G), fit = fit + F*G; end
R = Z2 - fit;
rec = struct('lam', lam, 'Phi', Phi, 'Psi', conj(Phi), 'U', U, 'R', R, 'V', V, 'T', T, ...
    'S2', S2, 's', sAll, 'share', share, 'B', B, 'G', G, 'condT', cond(T), 'condV', cond(V), ...
    'resid', resid, 'res', norm(R,'fro')/max(norm(Z2,'fro'),1e-300), ...
    'exact1', norm(Z1 - U*W','fro')/max(norm(Z1,'fro'),1e-300), 'timeScale', c, ...
    'operator', opts.Operator, 'readout', opts.Readout);
end

function o = sortOrder(lam)
% ascending |Im|, positive imaginary part first in each pair, real poles last
isr = abs(imag(lam)) < 1e-12*max(abs(lam), 1e-300);
key = [double(isr), round(abs(imag(lam)), 12), -sign(imag(lam)), abs(lam)];
[~, o] = sortrows(key);
end

function [out, share] = assignLegacy(lam, T)
% Superseded one-to-one pole-to-shape assignment (cost-of-change use only).
P = abs(T).^2; P = P./sum(P,1);
M = matchpairs(-P, 1e6, 'min');                 % rows, cols maximising sum P
out = zeros(size(lam)); share = zeros(size(lam));
out(M(:,1)) = lam(M(:,2));
share(M(:,1)) = P(sub2ind(size(P), M(:,1), M(:,2)));
end
