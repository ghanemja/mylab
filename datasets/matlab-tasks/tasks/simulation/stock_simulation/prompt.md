Write a MATLAB function
`[stock, lost, orderDays] = stock_simulation(start, demand, reorderPoint, orderQty, leadTime)`
that simulates a shop's stock of one product, day by day.

Inputs: `start` is the stock on hand before day 1; `demand` is a vector whose
element d is the number of units customers want on day d; `reorderPoint`,
`orderQty` and `leadTime` (an integer >= 1) are nonnegative numbers.

For each day d = 1, 2, ..., numel(demand), in this order:

1. **Delivery.** Every order due on day d is added to the stock on hand.
2. **Sales.** Sell `min(stock, demand(d))` units. Demand that cannot be met is
   lost (there are no back orders).
3. **Reordering.** At the end of the day, compute the *inventory position*:
   stock on hand plus all units ordered but not yet delivered. If it is less
   than or equal to `reorderPoint`, place one order of `orderQty` units; it
   will be delivered at the start of day `d + leadTime`. At most one order is
   placed per day. Orders due after the last day never arrive.

Return:

- `stock` — a row vector, the stock on hand at the end of each day;
- `lost` — the total number of units of lost demand;
- `orderDays` — a row vector of the days on which orders were placed
  (`zeros(1,0)` if none).

For empty `demand`, return `zeros(1,0)`, `0` and `zeros(1,0)`.

Example:

```matlab
[s, l, o] = stock_simulation(10, [4 4 4 4 4 4], 5, 10, 2)
% s = [6 2 0 6 2 0], l = 4, o = [2 5]
```
