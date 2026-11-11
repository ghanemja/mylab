function h = compose_all(varargin)
%COMPOSE_ALL Compose function handles; the last one is applied first.
for k = 1:nargin
    if ~isa(varargin{k}, 'function_handle')
        error('compose_all:notHandle', 'Input %d is not a function handle.', k);
    end
end
h = @(x) x;
for k = nargin:-1:1
    f = varargin{k};
    g = h;
    h = @(x) f(g(x));
end
end
