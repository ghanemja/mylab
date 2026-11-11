Any fraction between 0 and 1 can be written as a sum of distinct unit
fractions `1/d`. One way to find such a sum is greedy: repeatedly take the
largest unit fraction that does not exceed what remains, subtract it, and stop
when nothing remains.

Write a MATLAB function

    dens = greedy_unit_fractions(num, den)

where `num` and `den` are positive integers with `num < den`. Return the
denominators chosen by the greedy method, in the order chosen (which is
increasing), as a **row vector** of doubles. Use exact integer arithmetic
(keep the remainder as a reduced numerator/denominator pair); the inputs in
the tests keep all intermediate values well below `2^53`.

Examples:

- `greedy_unit_fractions(2, 3)` returns `[2 6]` (2/3 = 1/2 + 1/6).
- `greedy_unit_fractions(4, 13)` returns `[4 18 468]`.
- `greedy_unit_fractions(1, 7)` returns `7`.
