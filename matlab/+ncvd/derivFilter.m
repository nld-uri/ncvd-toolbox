function f = derivFilter(method, window, polyorder)
%NCVD.DERIVFILTER  FIR first-derivative stencil (D q)_k = fs*sum c_j q(k + off_j).
%   method: "forward" | "backward" | "central" | "central4" | "savgol"
%   window, polyorder: Savitzky-Golay parameters (default 7, 3).
%   Returns struct with fields c, off, name.
if nargin < 2, window = 7; end
if nargin < 3, polyorder = 3; end
switch lower(string(method))
    case "forward",  f = mk([-1 1], [0 1], "forward");
    case "backward", f = mk([-1 1], [-1 0], "backward");
    case "central",  f = mk([-0.5 0.5], [-1 1], "central");
    case "central4", f = mk([1 -8 8 -1]/12, [-2 -1 1 2], "central4");
    case "savgol"
        if mod(window,2) == 0, error('ncvd:savgol', 'window must be odd'); end
        h = (window-1)/2; k = (-h:h)';
        A = k.^(0:polyorder);                   % local polynomial fit
        P = pinv(A);                            % coefficients = P*q
        f = mk(P(2,:), k', sprintf("savgol(%d,%d)", window, polyorder));
    otherwise
        error('ncvd:filter', 'unknown method %s', method);
end
end

function f = mk(c, off, name)
f = struct('c', c(:).', 'off', off(:).', 'name', string(name));
end
