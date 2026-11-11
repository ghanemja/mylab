p = Poly([-1 0 1]);
[q, r] = divmod(p, Poly([1 1]));
assert(isa(q, 'Poly') && isequal(q.Coeffs, [-1 1]) && isequal(r.Coeffs, 0), 'example division');
assert(isequal(at(p * 2, [0 3]), [-2 16]), 'scalar product and evaluation');
p1 = Poly([5 0 2 0 0]); p2 = Poly([0 0]); p3 = Poly([]);
assert(isequal(p1.Coeffs, [5 0 2]) && isequal(p2.Coeffs, 0) && isequal(p3.Coeffs, 0), 'trimming');
assert(degree(Poly([5 0 2])) == 2 && degree(Poly(7)) == 0 && degree(Poly(0)) == -1, 'degree');
s = Poly([1 2 3]) + Poly([0 0 -3]);
assert(isequal(s.Coeffs, [1 2]) && Poly([1 2 3]) - Poly([1 2 3]) == Poly(0), 'sum cancels leading term');
t = 3 - Poly([0 1]);
m = Poly([1 1]) * Poly([-1 1]) * Poly([0 0 1]);
assert(isequal(t.Coeffs, [3 -1]) && isequal(m.Coeffs, [0 0 -1 0 1]), 'mixed operands and product');
X = [0 1; -2 0.5];
assert(isequal(size(at(m, X)), [2 2]) && all(all(abs(at(m, X) - (X.^4 - X.^2)) < 1e-12)), 'matrix evaluation');
d = deriv(Poly([4 -3 0 2]));
assert(isequal(d.Coeffs, [-3 0 6]) && deriv(Poly(9)) == Poly(0), 'derivative');
a = Poly([3 -2 0 4 1]); b = Poly([1 0 1]);
[q, r] = divmod(a, b);
chk = q * b + r;
assert(chk == a && degree(r) < degree(b) && isequal(q.Coeffs, [-1 4 1]) && isequal(r.Coeffs, [4 -6]), 'division identity');
[q, r] = divmod(Poly([1 2]), Poly([0 0 5]));
assert(q == Poly(0) && isequal(r.Coeffs, [1 2]), 'divisor of higher degree');
threw = false;
try
    divmod(a, Poly(0));
catch err
    threw = strcmp(err.identifier, 'Poly:divideByZero');
end
assert(threw, 'zero divisor');
