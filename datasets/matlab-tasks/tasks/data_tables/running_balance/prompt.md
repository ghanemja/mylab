Write a MATLAB function `T = running_balance(T)`.

`T` is a table of transactions in chronological order with (at least) the
variables `Account` (cell array of character vectors) and `Amount` (double;
deposits positive, withdrawals negative). Every account starts at a balance of
0.

Return the table with a new last variable `Balance` (double column): for each
row, the balance of that row's account immediately after the transaction,
i.e. the running sum of `Amount` over the rows of the same account up to and
including this row. The row order and all other variables are unchanged. For
a table with zero rows, `Balance` is an empty `zeros(0,1)` column.

Example:

```matlab
T = table({'A'; 'B'; 'A'}, [100; 20; -30], 'VariableNames', {'Account', 'Amount'});
T = running_balance(T);
T.Balance   % [100; 20; 70]
```
