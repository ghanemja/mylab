function [s, p, d] = cfh_multi(a, b)
% Sum, product and difference; the product is only computed when requested.
s = a + b;
if nargout > 1
    p = a * b;
end
if nargout > 2
    d = a - b;
end
end
