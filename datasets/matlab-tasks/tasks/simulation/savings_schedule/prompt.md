Write a MATLAB function
`b = savings_schedule(principal, annualRate, monthly, months)` that simulates
a savings account month by month.

The account starts with `principal`. At the end of every month, interest at a
monthly rate of `annualRate/12` is first added to the balance (so the balance
is multiplied by `1 + annualRate/12`), and then the deposit `monthly` is added.
`annualRate` is a fraction (0.06 means 6 %), and `monthly` may be negative
(a withdrawal).

Return a 1-by-`months` row vector `b` whose element k is the balance after
month k. For `months = 0`, return `zeros(1,0)`.

Example:

```matlab
savings_schedule(1000, 0.12, 100, 2)
% month 1: 1000*1.01 + 100 = 1110
% month 2: 1110*1.01 + 100 = 1221.1
% returns [1110 1221.1]
```
