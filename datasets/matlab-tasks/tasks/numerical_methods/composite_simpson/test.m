assert(abs(simpson_composite(@(x) x.^3, 0, 2, 2) - 4) < 1e-13, 'cubic is exact');
assert(abs(simpson_composite(@(x) 5*x.^3 - x + 2, -1, 3, 4) - 104) < 1e-12, 'another cubic');
W = simpson_composite(@(x) 50*x./(1 + x.^2), 0, 3, 200);
assert(abs(W - 57.564627324851145) < 1e-6, 'work example');
I10 = simpson_composite(@(x) exp(-x.^2), 0, 2, 10);
assert(abs(I10 - 0.882074876854494) < 1e-12, 'matches the Simpson formula for n = 10');
e1 = abs(simpson_composite(@(x) exp(-x.^2), 0, 2, 8) - 0.8820813907624215);
e2 = abs(simpson_composite(@(x) exp(-x.^2), 0, 2, 16) - 0.8820813907624215);
assert(e1/e2 > 12 && e1/e2 < 20, 'error drops about 16x when n doubles');
assert(abs(simpson_composite(@(x) sin(x), pi, 0, 20) + 2) < 1e-5, 'reversed limits give a negative result');
threw = false;
try
    simpson_composite(@(x) x, 0, 1, 3);
catch
    threw = true;
end
assert(threw, 'odd n must throw');
threw = false;
try
    simpson_composite(@(x) x, 0, 1, 0);
catch
    threw = true;
end
assert(threw, 'n = 0 must throw');
