function x = box_least_squares(A, b, lo, hi)
%BOX_LEAST_SQUARES Bound-constrained linear least squares (accelerated projected gradient).
b = b(:);
lo = lo(:);
hi = hi(:);
n = size(A, 2);
proj = @(v) min(max(v, lo), hi);
L = norm(A)^2;
x = proj(zeros(n, 1));
y = x;
t = 1;
for k = 1:200000
    xNew = proj(y - (A' * (A*y - b)) / L);
    tNew = (1 + sqrt(1 + 4*t^2)) / 2;
    y = xNew + ((t - 1) / tNew) * (xNew - x);
    dx = norm(xNew - x);
    x = xNew;
    t = tNew;
    if dx <= 1e-13 * (1 + norm(x))
        break
    end
end
end
