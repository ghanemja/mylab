Write a MATLAB function

    [total, details] = price_quote(unitPrice, varargin)

that computes the price of an order. After the unit price come optional
name-value pairs; names are char vectors compared case-insensitively:

| name        | meaning                       | default |
|-------------|-------------------------------|---------|
| `Quantity`  | number of units               | 1       |
| `Discount`  | fraction taken off the subtotal (0.15 means 15%) | 0 |
| `TaxRate`   | tax as a fraction, applied after the discount | 0 |
| `Round`     | logical; round money amounts to cents | true |

If a name appears more than once, the last value wins. Compute

    subtotal = unitPrice * Quantity
    discount = subtotal * Discount
    tax      = (subtotal - discount) * TaxRate
    total    = subtotal - discount + tax

When `Round` is true, round each of `subtotal`, `discount` and `tax` to cents
(`round(v * 100) / 100`) **before** computing `total`. `details` is a struct
with fields `subtotal`, `discount` and `tax` holding the (possibly rounded)
amounts.

Errors: an odd number of optional arguments gives identifier
`price_quote:pairs`; an unknown name gives `price_quote:unknownOption`.

Examples:

- `price_quote(19.99, 'Quantity', 3)` returns `59.97`.
- `price_quote(100, 'quantity', 2, 'discount', 0.15, 'taxrate', 0.2)` returns `204`.
- `price_quote(1/3, 'Round', false)` returns `1/3`.
