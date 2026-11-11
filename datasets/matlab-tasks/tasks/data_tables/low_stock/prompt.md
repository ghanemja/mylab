Write a MATLAB function `R = low_stock(T, threshold)`.

`T` is a table with at least the variables `Item` (cell array of character
vectors) and `Qty` (double); it may have other variables too. Return a table
`R` that contains only the rows whose `Qty` is **strictly less** than
`threshold`, sorted by `Qty` in ascending order. Rows with equal `Qty` keep
their original relative order. `R` keeps all variables of `T`, in the same
order. If no row qualifies, `R` has zero rows (and the same variables).

Example:

```matlab
T = table({'bolt'; 'nut'; 'gear'}, [12; 3; 7], 'VariableNames', {'Item', 'Qty'});
R = low_stock(T, 10);
R.Item   % {'nut'; 'gear'}
```
