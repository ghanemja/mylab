A ticket machine accepts coins of a few fixed values and has an unlimited
supply of each. Write a MATLAB function

    [ways, fewest] = coin_combinations(amount, coins)

- `amount` is a nonnegative integer.
- `coins` is a vector (row or column, possibly empty) of distinct positive
  integer coin values, in any order.

Return:

- `ways`: the number of different ways to pay exactly `amount`, where two
  ways are the same if they use the same number of each coin (order does not
  matter). Paying `0` can be done in exactly one way (no coins).
- `fewest`: the smallest number of coins that pays exactly `amount`, or `Inf`
  if it cannot be paid. Paying `0` needs `0` coins.

Amounts in the tests are at most a few hundred, so an exhaustive enumeration
of all combinations is too slow; use dynamic programming.

Examples:

- `[w, f] = coin_combinations(6, [1 2 3])` gives `w = 7`, `f = 2`.
- `[w, f] = coin_combinations(7, [2 4])` gives `w = 0`, `f = Inf`.
