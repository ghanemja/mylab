[x, h] = newton_solve(@(x) x^2 - 2, @(x) 2*x, 1, 1e-12, 50);
assert(abs(x - sqrt(2)) < 1e-14, 'sqrt(2)');
assert(isequal(size(h), [7 1]), 'history is a column with all iterates');
assert(all(abs(h - [1; 1.5; 1.4166666666666667; 1.4142156862745099; 1.4142135623746899; 1.4142135623730951; 1.414213562373095]) < 1e-14), 'iterates for sqrt(2)');
assert(h(end) == x && h(1) == 1, 'history starts at x0 and ends at x');
f = @(x) x*exp(x) - 2;
df = @(x) exp(x)*(1 + x);
[w, hw] = newton_solve(f, df, 1, 1e-13, 50);
assert(abs(w - 0.8526055020137254) < 1e-13, 'root of x*exp(x) = 2');
e = abs(hw - 0.8526055020137254);
C = 0.7698901625070828;
for k = 1:numel(e) - 1
    if e(k) < 0.1 && e(k + 1) > 1e-12
        assert(e(k + 1) / e(k)^2 > 0.5*C && e(k + 1) / e(k)^2 < 1.5*C, 'quadratic convergence');
    end
end
[z, hz] = newton_solve(@(x) 3*x - 6, @(x) 3, 10, 1e-9, 5);
assert(z == 2 && numel(hz) == 3, 'linear function: exact after one step, confirmed by the second');
threw = false;
try
    newton_solve(@(x) x^2 + 1, @(x) 2*x, 0, 1e-10, 20);
catch
    threw = true;
end
assert(threw, 'zero derivative must throw');
threw = false;
try
    newton_solve(@(x) x^2 + 1, @(x) 2*x, 0.5, 1e-10, 20);
catch
    threw = true;
end
assert(threw, 'non-convergence must throw');
