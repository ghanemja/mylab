c = Circle(1);
s = Square(2);
tri = Triangle(3, 4, 5);
assert(isa(c, 'Shape') && isa(s, 'Shape') && isa(tri, 'Shape'), 'subclasses of Shape');
assert(strcmp(c.Name, 'circle') && strcmp(s.Name, 'square') && strcmp(tri.Name, 'triangle'), 'names');
assert(abs(area(c) - pi) < 1e-12 && abs(perimeter(c) - 2 * pi) < 1e-12, 'circle');
assert(area(s) == 4 && perimeter(s) == 8 && s.Side == 2, 'square');
assert(abs(area(tri) - 6) < 1e-12 && perimeter(tri) == 12 && isequal(tri.Sides, [3 4 5]), 'triangle');
assert(abs(compactness(Circle(7)) - 1) < 1e-12 && abs(compactness(s) - pi / 4) < 1e-12, 'compactness');
assert(isLargerThan(s, c) && ~isLargerThan(c, s) && islogical(isLargerThan(c, s)), 'comparison');
assert(abs(Shape.totalArea({c, s, tri}) - (pi + 10)) < 1e-12 && Shape.totalArea({}) == 0, 'static total');
threw = false;
try
    Triangle(1, 2, 3);
catch err
    threw = strcmp(err.identifier, 'Triangle:invalid');
end
assert(threw, 'degenerate triangle');
threw = false;
try
    Triangle(1, 10, 2);
catch err
    threw = strcmp(err.identifier, 'Triangle:invalid');
end
assert(threw, 'impossible triangle');
threw = false;
try
    Shape('blob');
catch
    threw = true;
end
assert(threw, 'abstract class cannot be instantiated');
