Write a MATLAB function

    [gap, p, q] = largest_prime_gap(n)

Consider all primes less than or equal to the positive integer `n`, in
increasing order. Return the largest difference `gap` between two consecutive
primes in that list, and the pair of primes `p < q` where it occurs. If the
largest gap occurs more than once, return the pair with the smallest `p`.

If there are fewer than two primes up to `n`, return `gap = 0`, `p = NaN` and
`q = NaN`.

Examples:

- `[g, p, q] = largest_prime_gap(10)` gives `g = 2`, `p = 3`, `q = 5`.
- `[g, p, q] = largest_prime_gap(30)` gives `g = 6`, `p = 23`, `q = 29`.
- `[g, p, q] = largest_prime_gap(2)` gives `g = 0`, `p = NaN`, `q = NaN`.
