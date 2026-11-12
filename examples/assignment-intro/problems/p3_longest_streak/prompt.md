# Longest streak

A weather station records `1` for every day with sunshine and `0` for every
day without. A *streak* is a group of equal values next to each other.

Write a function

```matlab
n = longest_streak(v)
```

that returns the length of the longest streak of equal values in the vector
`v`.

- `v` is a row or column vector of numbers (not only 0 and 1).
- If `v` is empty, return `0`.
- A single value on its own is a streak of length 1.

## Examples

```matlab
>> longest_streak([1 1 0 1 1 1 0])
ans =
     3

>> longest_streak([4 4 4 4])
ans =
     4

>> longest_streak([7 8 9])
ans =
     1
```

Hint: walk through the vector with a `for` loop and count how long the
current streak is.
