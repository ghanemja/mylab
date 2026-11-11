Write a small class hierarchy of plane shapes in four files.

**`Shape.m`** — an abstract base class (it must not be instantiable):

- property `Name` (character vector), set by the subclass constructors;
- abstract methods `a = area(obj)` and `p = perimeter(obj)`;
- concrete method `c = compactness(obj)` returning
  `4*pi*area(obj) / perimeter(obj)^2` (1 for a circle, smaller for other
  shapes);
- concrete method `tf = isLargerThan(obj, other)` returning logical true when
  `area(obj) > area(other)`;
- static method `t = Shape.totalArea(shapes)` where `shapes` is a cell array
  of shape objects (possibly of different classes); returns the sum of their
  areas, or 0 for an empty cell array.

**Subclasses** of `Shape`, each with a read-only-from-outside size property:

- `Circle(r)` — property `Radius`; `Name` is `'circle'`.
- `Square(s)` — property `Side`; `Name` is `'square'`.
- `Triangle(a, b, c)` — property `Sides` holding `[a b c]` as a 1-by-3 row;
  `Name` is `'triangle'`; area by Heron's formula. If the three lengths do
  not form a triangle with positive area (some side is greater than or equal
  to the sum of the other two), throw an error with identifier
  `'Triangle:invalid'`.

Example:

```matlab
c = Circle(1); s = Square(2);
isLargerThan(s, c)                    % true (4 > pi)
Shape.totalArea({c, s, Triangle(3, 4, 5)})   % pi + 4 + 6
```
