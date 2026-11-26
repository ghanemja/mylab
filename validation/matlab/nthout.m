function r = nthout(n, f, varargin)
[o{1:n}] = f(varargin{:});
r = o{n};
end
