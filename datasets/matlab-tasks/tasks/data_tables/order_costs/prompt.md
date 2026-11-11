Write a MATLAB function `T = order_costs(orders, prices)`.

- `orders` is a table with (at least) the variables `OrderID` (double),
  `Product` (cell array of character vectors) and `Qty` (double). It may
  contain other variables, which are dropped.
- `prices` is a table with the variables `Product` (cell array of character
  vectors, each product listed once) and `UnitPrice` (double).

Return a table `T` with one row per order, **in the same row order as
`orders`**, and exactly these variables in this order: `OrderID`, `Product`,
`Qty`, `UnitPrice`, `Cost`, where `UnitPrice` is looked up in `prices` and
`Cost = Qty .* UnitPrice`. Orders for a product missing from `prices` keep
their row but get `NaN` for both `UnitPrice` and `Cost`. Product names are
matched exactly (case-sensitive).

Example:

```matlab
orders = table([7; 8], {'pen'; 'ink'}, [3; 2], 'VariableNames', {'OrderID', 'Product', 'Qty'});
prices = table({'ink'; 'pen'}, [4.5; 1.2], 'VariableNames', {'Product', 'UnitPrice'});
T = order_costs(orders, prices);
% T.UnitPrice = [1.2; 4.5], T.Cost = [3.6; 9]
```
