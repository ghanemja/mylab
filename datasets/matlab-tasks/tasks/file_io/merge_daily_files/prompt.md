Write a MATLAB function `[nDays, total] = merge_daily_files(files, outfile)`.

`files` is a cell array of file names. Each file is a small CSV export whose
first line is the header `date,amount`; every following non-empty line has a
date in ISO form `YYYY-MM-DD`, a comma, and an amount (a decimal number,
possibly negative). Blank lines (empty or whitespace only) may appear anywhere
and are skipped. The same date may appear several times, in one file or in
different files.

Combine all records, add up the amounts of records with the same date, and
write `outfile` (overwriting it) with:

- the header line `date,amount`;
- one line per distinct date, in increasing date order, of the form
  `YYYY-MM-DD,<amount>` where the amount is written with exactly two decimals
  (`%.2f`).

Every line ends with `\n`. Return `nDays`, the number of distinct dates, and
`total`, the sum of all amounts (unrounded). If there are no records at all,
the output file contains only the header and both outputs are 0.

Example: with `a.csv` containing `2024-05-02,10.5` and `2024-05-01,3`, and
`b.csv` containing `2024-05-02,-0.25`, the output file is

```
date,amount
2024-05-01,3.00
2024-05-02,10.25
```

and the outputs are `nDays = 2`, `total = 13.25`.
