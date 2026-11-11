function [r, tc] = frame_rms(x, fs, L, hop, window)
%FRAME_RMS Weighted RMS of successive frames of a signal.
if nargin < 5
    window = 'rect';
end
switch lower(window)
    case 'rect'
        w = ones(L, 1);
    case 'hann'
        if L > 1
            w = 0.5 - 0.5*cos(2*pi*(0:L-1)'/(L - 1));
        else
            w = 1;
        end
    otherwise
        error('frame_rms:window', 'Unknown window ''%s''.', window);
end
x = x(:);
starts = 1:hop:(numel(x) - L + 1);
r = zeros(1, numel(starts));
for i = 1:numel(starts)
    s = x(starts(i):starts(i) + L - 1);
    r(i) = sqrt(sum(w .* s.^2) / sum(w));
end
tc = (starts - 1 + (L - 1)/2) / fs;
if isempty(starts)
    r = zeros(1, 0);
    tc = zeros(1, 0);
end
end
