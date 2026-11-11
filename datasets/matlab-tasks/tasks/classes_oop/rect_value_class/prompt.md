Write a MATLAB **value** class `Rect` (file `Rect.m`) describing an
axis-aligned rectangle.

API:

- `r = Rect(w, h)` creates a rectangle with `Width = w` and `Height = h`.
  If either argument is negative, throw an error with identifier
  `'Rect:negative'`. Zero sizes are allowed.
- Public properties `Width` and `Height` (ordinary, settable).
- Dependent properties `Area` (`Width*Height`) and `Perimeter`
  (`2*(Width+Height)`), which always reflect the current width and height.
- `r2 = scale(r, f)` returns a **new** `Rect` with both sides multiplied by
  `f`; `r` itself is not changed.
- `tf = isSquare(r)` returns logical true when width equals height.

Because `Rect` is a value class, `b = a; b.Width = 1;` must not change `a`.

Example:

```matlab
r = Rect(3, 4);
r.Area          % 12
s = scale(r, 2);
s.Perimeter     % 28
r.Width         % 3
```
