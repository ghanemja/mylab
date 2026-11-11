function y = direct_convolution(x, h)
%DIRECT_CONVOLUTION Full linear convolution by explicit summation.
if isempty(x) || isempty(h)
    y = zeros(1, 0);
    return
end
nx = numel(x);
nh = numel(h);
y = zeros(1, nx + nh - 1);
for i = 1:nx
    for j = 1:nh
        y(i + j - 1) = y(i + j - 1) + x(i) * h(j);
    end
end
end
