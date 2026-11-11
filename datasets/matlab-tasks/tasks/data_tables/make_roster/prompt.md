Write a MATLAB function `T = make_roster(names, ages)` that builds a MATLAB
`table` describing club members.

- `names` is a cell array of character vectors (row or column).
- `ages` is a numeric vector of the same length (row or column).

Return a table with one row per member, in input order, and exactly these
three variables, in this order:

| Variable | Contents |
|----------|----------|
| `Name`   | the names, as an n-by-1 cell array of character vectors |
| `Age`    | the ages, as an n-by-1 double column |
| `Adult`  | n-by-1 `logical`, true when the age is 18 or more |

If `names` and `ages` are empty, return a table with zero rows and the same
three variables.

Example:

```matlab
T = make_roster({'Ines', 'Tomas'}, [17 18]);
T.Adult   % [false; true]
```
