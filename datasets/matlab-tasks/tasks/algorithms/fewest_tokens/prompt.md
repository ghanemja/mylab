Write a MATLAB function `k = fewest_tokens(values, amount)`.

A vending machine accepts tokens whose face values are listed in the vector
`values` (distinct positive integers, in any order). There is an unlimited
supply of every kind of token. `amount` is a nonnegative integer.

Return the smallest number of tokens whose values add up to exactly `amount`.
Return `0` when `amount` is 0 and `-1` when the amount cannot be formed.

Note that always taking the largest token that still fits does **not** always
give the smallest count.

Examples:

- `fewest_tokens([1 3 4], 6)` returns `2` (3 + 3).
- `fewest_tokens([5 7], 3)` returns `-1`.
