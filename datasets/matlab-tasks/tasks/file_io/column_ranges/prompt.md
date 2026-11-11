Write a MATLAB function `R = column_ranges(infile, outfile)`.

`infile` is a comma-separated file of numbers with no header line; every line
has the same number of fields, but some fields may be empty (missing values).
Read it with `readmatrix` (missing values become `NaN`).

Return a 2-by-N double matrix `R`, where N is the number of columns: row 1
holds the minimum and row 2 the maximum of each column, ignoring missing
values. A column whose values are all missing gets `NaN` in both rows.

Also write `R` to `outfile` with `writematrix` (a comma-separated text file
with two lines).

Example: for a file containing

```
3,10,-1
,4,2
7,,0
```

`R` is `[3 4 -1; 7 10 2]`.
