Q = [4 1 0; 1 3 -1; 0 -1 2]; b = [1; 2; 3];
[x, k] = newton_minimize(@(x) Q*x - b, @(x) Q, [0 0 0], 1e-10, 20);
assert(isequal(size(x), [3 1]) && norm(x - Q\b) < 1e-12 && k == 1, 'quadratic in one step');
g = @(p) [exp(p(1)+p(2)-1) + exp(p(1)-p(2)-1) - exp(-p(1)-1); exp(p(1)+p(2)-1) - exp(p(1)-p(2)-1) + p(2)];
H = @(p) [exp(p(1)+p(2)-1) + exp(p(1)-p(2)-1) + exp(-p(1)-1), exp(p(1)+p(2)-1) - exp(p(1)-p(2)-1); ...
          exp(p(1)+p(2)-1) - exp(p(1)-p(2)-1), exp(p(1)+p(2)-1) + exp(p(1)-p(2)-1) + 1];
[x, k] = newton_minimize(g, H, [1; 1], 1e-12, 50);
assert(all(abs(x - [-0.34657359027997264; 0]) < 1e-8), 'smooth convex function');
assert(k >= 3 && k <= 10, 'few iterations (quadratic convergence)');
[x, k] = newton_minimize(@(x) 4*x^3, @(x) 12*x^2, 1, 1e-6, 100);
assert(abs(x) < 0.02 && k > 5, 'degenerate minimum converges only linearly');
[x, k] = newton_minimize(@(x) x - 3, @(x) 1, 3, 1e-8, 5);
assert(x == 3 && k == 0, 'already at the minimum');
threw = false;
try
    newton_minimize(@(x) 4*x^3, @(x) 12*x^2, 1, 1e-12, 3);
catch
    threw = true;
end
assert(threw, 'exceeding maxit must throw');
