function [fwd, inv] = make_scaler(lo, hi)
%MAKE_SCALER Handles mapping [lo, hi] to [0, 1] and back.
if lo == hi
    error('make_scaler:degenerate', 'lo and hi must differ.');
end
width = hi - lo;
fwd = @(x) (x - lo) ./ width;
inv = @(y) lo + y .* width;
end
