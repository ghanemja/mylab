A small car park charges drivers by the length of their stay. Write a MATLAB
function

    fee = parking_fee(minutes)

Rules for a single stay of `minutes` minutes:

- A stay of 30 minutes or less is free.
- Otherwise the driver pays 2.50 for every started hour of the whole stay
  (counted from arrival, so 61 minutes counts as 2 hours).
- The fee for one stay never exceeds 18.00.

`minutes` is an array of any shape holding nonnegative real numbers (one stay
per element). Return `fee` as a double array of the same size, with the fee of
each stay.

Examples:

- `parking_fee(30)` returns `0`.
- `parking_fee(31)` returns `2.5`.
- `parking_fee([61; 600])` returns `[5; 18]`.
