function ed = windowEdges(m, nWindows, len, step)
%NCVD.WINDOWEDGES  [start stop] rows (1-based, inclusive). Either nWindows
%   equal windows, or windows of len samples advanced by step.
if ~isempty(nWindows)
    e = floor(linspace(0, m, nWindows+1));
    ed = [e(1:end-1)'+1, e(2:end)'];
else
    if nargin < 4 || isempty(step), step = len; end
    s = (0:step:m-len)';
    ed = [s+1, s+len];
end
end
