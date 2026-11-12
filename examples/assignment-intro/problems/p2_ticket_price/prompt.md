# Museum ticket price

A museum charges by age:

| Age         | Price |
|-------------|-------|
| under 5     | 0     |
| 5 to 17     | 6     |
| 18 to 64    | 11    |
| 65 and over | 8     |

On Mondays (`is_monday` is `true`) every ticket that costs money is 2 cheaper.
Free tickets stay free.

Write a function

```matlab
price = ticket_price(age, is_monday)
```

- `age` is a whole number, 0 or more.
- `is_monday` is `true` or `false`.
- `price` is the price of one ticket, as a number.

## Examples

```matlab
>> ticket_price(30, false)
ans =
    11

>> ticket_price(12, true)
ans =
     4

>> ticket_price(3, true)
ans =
     0
```
