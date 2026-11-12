function [lo, hi, varargout] = cfh_minmax(v, varargin)
% Minimum and maximum, plus one extra output per extra input (its double).
lo = min(v);
hi = max(v);
for k = 1:numel(varargin)
    varargout{k} = 2 * varargin{k};
end
end
