q = Ratio(6, -4);
assert(q.Num == -3 && q.Den == 2, 'lowest terms, positive denominator');
a = Ratio(1, 6) + Ratio(1, 3);
assert(isa(a, 'Ratio') && a.Num == 1 && a.Den == 2, 'addition');
b = 2 - a;
assert(b.Num == 3 && b.Den == 2, 'scalar on the left');
c = b / Ratio(3, 4);
assert(c.Num == 2 && c.Den == 1 && double(c) == 2, 'division and double');
d = Ratio(2, 3) * Ratio(-9, 4);
assert(d.Num == -3 && d.Den == 2, 'multiplication');
z = Ratio(5, 7) - Ratio(10, 14);
assert(z.Num == 0 && z.Den == 1, 'zero normalised');
m = -Ratio(1, 3);
assert(m.Num == -1 && m.Den == 3 && Ratio(2, 6) == Ratio(1, 3) && ~(Ratio(1, 3) == m), 'unary minus and equality');
assert(Ratio(1, 3) < Ratio(1, 2) && ~(Ratio(1, 2) < Ratio(1, 3)) && Ratio(-1, 2) < 0, 'less than');
assert(Ratio(4, 2) == 2 && islogical(Ratio(4, 2) == 2), 'compare with double');
threw = false;
try
    Ratio(1, 0);
catch err
    threw = strcmp(err.identifier, 'Ratio:zeroDenominator');
end
assert(threw, 'zero denominator');
threw = false;
try
    Ratio(1, 2) / Ratio(0, 5);
catch err
    threw = strcmp(err.identifier, 'Ratio:zeroDenominator');
end
assert(threw, 'division by zero');
threw = false;
try
    q.Num = 4;
catch
    threw = true;
end
assert(threw, 'Num read-only');
