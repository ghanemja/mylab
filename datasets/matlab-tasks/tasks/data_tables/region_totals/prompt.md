Write a MATLAB function `S = region_totals(T)`.

`T` is a table of sales records with (at least) the variables `Region`, a cell
array of character vectors, and `Sales`, a double column. Return a summary
table `S` with one row per distinct region and exactly these variables, in
this order:

- `Region` — the region name (cell array of character vectors),
- `Total` — the sum of `Sales` for that region (double),
- `Count` — the number of records for that region (double).

Rows of `S` are sorted by region name in ascending order (the order produced
by `sort` on a cell array of character vectors). Region names are
case-sensitive.

Example:

```matlab
T = table({'north'; 'east'; 'north'}, [10; 5; 2.5], 'VariableNames', {'Region', 'Sales'});
S = region_totals(T);
% S.Region = {'east'; 'north'}, S.Total = [5; 12.5], S.Count = [1; 2]
```
