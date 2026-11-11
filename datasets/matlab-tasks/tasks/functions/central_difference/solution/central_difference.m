function d = central_difference(f, x, h)
%CENTRAL_DIFFERENCE Central-difference derivative estimate.
if nargin < 3
    h = 1e-5;
end
d = (f(x + h) - f(x - h)) / (2 * h);
end
