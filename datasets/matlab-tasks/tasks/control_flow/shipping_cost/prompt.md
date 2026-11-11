A web shop computes parcel shipping costs from the destination region code and
the parcel weight. Write a MATLAB function

    cost = shipping_cost(region, kg)

`region` is a char vector; compare it **case-insensitively**. `kg` is a
positive real scalar. The billed weight is `kg` rounded **up** to a whole
number of kilograms. Use a `switch` statement for the regions:

| region codes          | base fee | per billed kg |
|-----------------------|---------:|--------------:|
| `'HOME'`              | 4.00     | 0.50          |
| `'EU'`, `'UK'`, `'CH'`| 9.00     | 1.20          |
| `'US'`, `'CA'`        | 15.00    | 2.00          |

`cost = base fee + billed kg * per-kg rate`. Any other region code throws an
error with identifier `shipping_cost:unknownRegion`.

Examples:

- `shipping_cost('home', 2)` returns `5`.
- `shipping_cost('UK', 0.3)` returns `10.2`.
