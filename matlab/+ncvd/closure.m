function P = closure(V0, corrections, k, tol, pMax, opts)
%NCVD.CLOSURE  Smallest P with escape of the first k corrections from
%   span(V0(:,1:P)) below tol; [] if not met. Do not pose with k = P.
%
%   P = ncvd.closure(V0, C, k, tol, pMax, Strict=true)   (opt-in, 1.1.2)
%   never returns []: it raises ncvd:closure with the reason when C has fewer
%   than k columns, when the posing is degenerate (V0 offers no column beyond
%   the k masters, so P can only be k; in particular closure([v0], c, 1) for
%   one tracked mode, whose deflation correction is orthogonal to v0 by
%   construction), or when closure is not met within pMax. For the closure of
%   one shape family use ncvd.familyClosure.
arguments
    V0
    corrections
    k = 1
    tol = 0.05
    pMax = []
    opts.Strict (1,1) logical = false
end
if isempty(pMax), pMax = size(V0,2); end
if opts.Strict
    hi = min(pMax, size(V0,2));
    if size(corrections,2) < k
        error('ncvd:closure', 'closure: corrections has %d columns, fewer than k=%d', ...
            size(corrections,2), k);
    end
    if hi <= k
        error('ncvd:closure', ['closure is degenerate: V0 has %d column(s) (pMax=%d), so the ' ...
            'only candidate is P = k = %d. A correction deflated from a shape family is ' ...
            'orthogonal to that family''s own v0, so with one tracked mode (V0 = [v0], k = 1) ' ...
            'closure can never be met. Pass a candidate basis with columns beyond the k ' ...
            'masters (linear modes, or the other tracked shapes), or use ' ...
            'ncvd.familyClosure(shapes, tol) for the closure of one shape family.'], ...
            size(V0,2), pMax, k);
    end
end
C = corrections(:, 1:k);
for P = max(k,1):min(pMax, size(V0,2))
    if max(ncvd.escape(V0(:,1:P), C)) < tol, return; end
end
if opts.Strict
    error('ncvd:closure', ['closure not met: the first %d correction(s) escape ' ...
        'span(V0(:,1:P)) by at least tol=%g for every P up to %d (escape %.3g at P = %d). ' ...
        'Enlarge V0 or pMax.'], k, tol, min(pMax, size(V0,2)), ...
        max(ncvd.escape(V0(:,1:min(pMax, size(V0,2))), C)), min(pMax, size(V0,2)));
end
P = [];
end
