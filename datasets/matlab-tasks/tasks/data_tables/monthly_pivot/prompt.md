Write a MATLAB function `P = monthly_pivot(T)` that turns a long list of
expenses into a wide summary table.

`T` is a table with the variables `Month` (double, whole numbers 1–12),
`Category` (cell array of character vectors) and `Amount` (double). It has at
least one row.

Return a table `P` with:

- one row per distinct category, sorted in ascending order (as `sort` orders
  a cell array of character vectors);
- a first variable `Category` holding the category names;
- then one variable for every month that occurs in `T`, in increasing month
  order, named `M` followed by the month number without padding (`M1`, `M7`,
  `M12`, ...). It holds the total `Amount` of that category in that month, or
  `0` if the category has no expense that month.

Months that never occur in `T` get no variable.

Example:

```matlab
T = table([3; 1; 3; 3], {'food'; 'rent'; 'food'; 'rent'}, [10; 500; 5; 450], ...
          'VariableNames', {'Month', 'Category', 'Amount'});
P = monthly_pivot(T);
% P.Properties.VariableNames = {'Category', 'M1', 'M3'}
% P.Category = {'food'; 'rent'}, P.M1 = [0; 500], P.M3 = [15; 450]
```
