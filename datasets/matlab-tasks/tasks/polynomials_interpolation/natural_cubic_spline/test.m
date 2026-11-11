yq = natural_spline([0 1 2], [0 1 0], [0.5 1.5]);
assert(all(abs(yq - [0.6875 0.6875]) < 1e-14), 'example');
x = [0 1 2.5 3 4.5 6];
y = [0 0.8 0.9 0.1 -0.8 -0.3];
yq = natural_spline(x, y', [0; 0.4; 1; 1.7; 2.75; 3.9; 5.2; 6]);
assert(isequal(size(yq), [8 1]) && all(abs(yq - [0; 0.32432689450222885; 0.8; 1.2240088492653127; 0.5162147102526002; -0.7218234769687965; -0.6416316548346266; -0.3]) < 1e-12), 'cam profile');
assert(all(abs(natural_spline(x, y, x) - y) < 1e-14), 'passes through the knots');
d = 1e-6;
for k = 2:5
    sl = (natural_spline(x, y, x(k)) - natural_spline(x, y, x(k) - d)) / d;
    sr = (natural_spline(x, y, x(k) + d) - natural_spline(x, y, x(k))) / d;
    assert(abs(sl - sr) < 1e-4, 'continuous slope at interior knots');
end
e = 1e-3;
c2 = (natural_spline(x, y, 2*e) - 2*natural_spline(x, y, e) + natural_spline(x, y, 0)) / e^2;
assert(abs(c2) < 0.05, 'zero curvature at the left end');
yl = natural_spline([0 2 3 7], [1 5 7 15], [0.5 2.5 6]);
assert(all(abs(yl - [2 6 13]) < 1e-12), 'collinear data give the straight line');
yo = natural_spline(x, y, [-0.1 6.1]);
assert(all(isnan(yo)), 'outside the range');
threw = false;
try
    natural_spline([0 1], [1 2], 0.5);
catch
    threw = true;
end
assert(threw, 'two points must throw');
threw = false;
try
    natural_spline([0 2 1], [1 2 3], 0.5);
catch
    threw = true;
end
assert(threw, 'unsorted x must throw');
