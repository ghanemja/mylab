Write a MATLAB function
`[m, short] = merge_inventories(names1, qty1, names2, qty2)`.

Two warehouses report stock changes. `names1` and `names2` are cell arrays of
item names (character vectors); `qty1` and `qty2` are numeric vectors of the
same lengths giving a quantity for each entry (negative quantities mean the
item is owed). An item may appear several times in either list, and lists may
be empty.

Return:

- `m` — a `containers.Map` (`KeyType` `'char'`) from item name to the total
  quantity over both lists (a double). Items whose total is exactly 0 must not
  appear in the map.
- `short` — a 1-by-k cell array of the names whose total is negative, sorted
  alphabetically (as `sort` orders character vectors); `cell(1,0)` if none.

Item names are case-sensitive.

Example:

```matlab
[m, s] = merge_inventories({'bolt', 'nut'}, [5 2], {'nut', 'gear', 'bolt'}, [-2 -1 1]);
m('bolt')     % 6
isKey(m, 'nut')   % false (total 0)
s             % {'gear'}
```
