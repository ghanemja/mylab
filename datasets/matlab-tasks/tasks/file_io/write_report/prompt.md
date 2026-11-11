Write a MATLAB function `total = write_report(filename, names, values)` that
writes a plain-text report with aligned columns.

- `names` is a cell array of n character vectors (n >= 0).
- `values` is a numeric vector with n elements.

Let `W` be the length of the longest name, counting the word `TOTAL` as a
name too, plus 2. Write the file (overwriting it) as follows, every line
ending with a newline character `\n`:

1. One line per item, in input order: the name padded on the right with spaces
   to exactly `W` characters, followed by the value printed with two decimals
   and right-aligned in a field of 10 characters (as `%10.2f` does).
2. A line consisting of `W + 10` hyphens (`-`).
3. A line formatted like the item lines with the name `TOTAL` and the sum of
   all values.

Return the sum of the values as a double (0 for no items).

Example: `write_report('r.txt', {'rent', 'electricity'}, [950 61.5])` returns
`1011.5` and writes (W = 13)

```
rent             950.00
electricity       61.50
-----------------------
TOTAL           1011.50
```
