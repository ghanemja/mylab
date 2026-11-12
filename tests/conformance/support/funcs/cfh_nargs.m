function varargout = cfh_nargs(varargin)
% Every requested output is [nargin nargout k] for output number k.
n = nargin;
m = nargout;
varargout = cell(1, max(m, 1));
for k = 1:numel(varargout)
    varargout{k} = [n m k];
end
end
