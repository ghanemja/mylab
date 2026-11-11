t = 0:0.25:2;
v = fd_derivative(4.9*t.^2 + 3*t + 1, 0.25);
assert(isequal(size(v), [1 9]), 'row in, row out');
assert(all(abs(v - (9.8*t + 3)) < 1e-12), 'exact for a quadratic');
vc = fd_derivative((t.^2)', 0.25);
assert(isequal(size(vc), [9 1]) && all(abs(vc - 2*t') < 1e-12), 'column in, column out');
d3 = fd_derivative([1 4 9], 1);
assert(all(abs(d3 - [2 4 6]) < 1e-12), 'three samples');
h1 = pi/20; x1 = 0:h1:pi;
h2 = pi/40; x2 = 0:h2:pi;
e1 = max(abs(fd_derivative(sin(x1), h1) - cos(x1)));
e2 = max(abs(fd_derivative(sin(x2), h2) - cos(x2)));
assert(e1/e2 > 3.5 && e1/e2 < 4.5, 'second order including the end points');
de = fd_derivative(exp(0:0.01:1), 0.01);
assert(abs(de(1) - 1) < 1e-4 && abs(de(end) - exp(1)) < 3e-4, 'end point accuracy');

threw = false;
try
    fd_derivative([1 2], 0.1);
catch
    threw = true;
end
assert(threw, 'fewer than 3 samples must throw');
