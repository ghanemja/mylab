# Rain summary

A text file holds the rainfall measured on each day of a period, in
millimetres. All the numbers are on one line, separated by commas, for
example the file `rain_week.csv`:

```
0,12.5,3,0,0,7.5,1
```

Write a function

```matlab
[total, wettest_day, dry_days] = rain_summary(filename)
```

that reads the file and returns:

- `total`: the total rainfall (the sum of all values);
- `wettest_day`: the number of the day with the most rain (the first day is
  day 1). If several days share the highest value, return the earliest;
- `dry_days`: a **row vector** with the numbers of the days that had no rain
  at all (value 0), in increasing order. If there are none, it is empty.

`filename` is text such as `'rain_week.csv'`. You can read the numbers with
`readmatrix(filename)`.

## Example

```matlab
>> [total, wettest_day, dry_days] = rain_summary('rain_week.csv')
total =
    24
wettest_day =
     2
dry_days =
     1     4     5
```

The file `rain_week.csv` is provided so that you can try your function.
