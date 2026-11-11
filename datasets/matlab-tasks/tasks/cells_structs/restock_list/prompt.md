A shop keeps its inventory in a struct array `items` with fields `name`
(char vector), `qty` (current stock) and `reorder_level` (numbers). Write a
MATLAB function

    [names, units] = restock_list(items)

An item needs restocking when `qty <= reorder_level`. Return

- `names`: an `m-by-1` **cell array** with the names of those items in
  alphabetical (character code) order;
- `units`: an `m-by-1` double column vector with, for each listed item in the
  same order, the number of units to order so that its stock becomes twice its
  reorder level (`2*reorder_level - qty`).

If nothing needs restocking (or `items` is empty), return a `0-by-1` cell and
a `0-by-1` double.

Example:

    items = struct('name', {'tea', 'flour', 'eggs'}, ...
                   'qty', {4, 30, 6}, 'reorder_level', {5, 10, 6});
    [n, u] = restock_list(items)
    % n = {'eggs'; 'tea'}, u = [6; 6]
