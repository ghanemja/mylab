function y = affine(x, a, b)
% a small user function with a branch, as in real code
if nargin < 3
    b = 0;
end
y = a * x + b;
end
