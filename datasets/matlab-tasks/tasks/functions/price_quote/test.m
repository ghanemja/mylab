assert(abs(price_quote(10) - 10) < 1e-9, 'defaults');
assert(abs(price_quote(19.99, 'Quantity', 3) - 59.97) < 1e-9, 'quantity');
[t, d] = price_quote(100, 'quantity', 2, 'discount', 0.15, 'taxrate', 0.2);
assert(abs(t - 204) < 1e-9, 'all options, lowercase');
assert(abs(d.subtotal - 200) < 1e-9 && abs(d.discount - 30) < 1e-9 && abs(d.tax - 34) < 1e-9, 'details');
assert(abs(price_quote(1/3, 'Round', false) - 1/3) < 1e-15, 'no rounding');
assert(abs(price_quote(1/3) - 0.33) < 1e-12, 'rounded to cents');
assert(abs(price_quote(2.5, 'QUANTITY', 4, 'Quantity', 2) - 5) < 1e-9, 'last value wins');
[t, d] = price_quote(0.99, 'TaxRate', 0.075, 'Quantity', 7);
assert(abs(d.tax - 0.52) < 1e-9 && abs(t - 7.45) < 1e-9, 'tax rounded before total');
ok = false;
try
    price_quote(5, 'Quantity');
catch err
    ok = strcmp(err.identifier, 'price_quote:pairs');
end
assert(ok, 'odd number of arguments');
ok = false;
try
    price_quote(5, 'Colour', 'red');
catch err
    ok = strcmp(err.identifier, 'price_quote:unknownOption');
end
assert(ok, 'unknown option');
