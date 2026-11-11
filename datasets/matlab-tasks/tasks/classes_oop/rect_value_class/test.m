r = Rect(3, 4);
assert(r.Area == 12 && r.Perimeter == 14, 'dependent values');
s = scale(r, 2);
assert(isa(s, 'Rect') && s.Perimeter == 28 && s.Area == 48, 'scaled copy');
assert(r.Width == 3 && r.Height == 4, 'original unchanged by scale');
b = r;
b.Width = 1;
assert(r.Width == 3 && b.Area == 4, 'value semantics and live dependent property');
assert(~isSquare(r) && islogical(isSquare(r)), 'not square');
assert(isSquare(Rect(2.5, 2.5)), 'square');
z = Rect(0, 7);
assert(z.Area == 0 && z.Perimeter == 14, 'zero width allowed');
threw = false;
try
    Rect(-1, 2);
catch err
    threw = strcmp(err.identifier, 'Rect:negative');
end
assert(threw, 'negative side error identifier');
