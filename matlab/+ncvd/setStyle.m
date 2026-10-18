function C = setStyle(palette)
%NCVD.SETSTYLE  LaTeX interpreters and a categorical palette.
%   palette: "matlab" (default ColorOrder, unchanged) or "nld" (lab series).
%   Returns the palette as an (k x 3) RGB matrix. Export figures with
%   exportgraphics(gcf, 'fig.pdf', 'ContentType', 'vector').
if nargin < 1, palette = "matlab"; end
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');
switch string(palette)
    case "nld"
        hex = ["#1F5A85" "#B4622A" "#588A75" "#CA863E" "#774860"];
    otherwise
        hex = ["#0072BD" "#D95319" "#EDB120" "#7E2F8E" "#77AC30" "#4DBEEE" "#A2142F"];
end
C = cell2mat(arrayfun(@(h) sscanf(char(extractAfter(h,1)), '%2x%2x%2x').'/255, hex(:), 'UniformOutput', false));
set(groot, 'defaultAxesColorOrder', C);
end
