function [coef, rms] = fit_plane(P)
%FIT_PLANE Least-squares plane z = a*x + b*y + c.
if size(P, 1) < 3
    error('fit_plane:points', 'Need at least three points.');
end
M = [P(:, 1), P(:, 2), ones(size(P, 1), 1)];
coef = M \ P(:, 3);
res = M * coef - P(:, 3);
rms = sqrt(mean(res.^2));
end
