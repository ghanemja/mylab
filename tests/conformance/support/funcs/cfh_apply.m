function varargout = cfh_apply(f, varargin)
% Calls f with the remaining arguments and passes all outputs through.
[varargout{1:nargout}] = f(varargin{:});
end
