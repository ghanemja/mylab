function r = cfh_opt(a, b, c)
% Optional arguments with defaults: b defaults to 10, c to 'none'.
if nargin < 2
    b = 10;
end
if nargin < 3 || isempty(c)
    c = 'none';
end
r = struct('a', a, 'b', b, 'c', c, 'n', nargin, 'has_b', exist('b', 'var'));
end
