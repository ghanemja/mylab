assert(abs(shipping_cost('home', 2) - 5) < 1e-12, 'home');
assert(abs(shipping_cost('UK', 0.3) - 10.2) < 1e-12, 'partial kg rounds up');
assert(abs(shipping_cost('ch', 3) - 12.6) < 1e-12, 'lowercase europe');
assert(abs(shipping_cost('Eu', 10) - 21) < 1e-12, 'mixed case');
assert(abs(shipping_cost('CA', 4.01) - 25) < 1e-12, 'north america');
assert(abs(shipping_cost('US', 1) - 17) < 1e-12, 'exact kg');
ok = false;
try
    shipping_cost('MARS', 1);
catch err
    ok = strcmp(err.identifier, 'shipping_cost:unknownRegion');
end
assert(ok, 'unknown region');
ok = false;
try
    shipping_cost('E', 1);
catch err
    ok = strcmp(err.identifier, 'shipping_cost:unknownRegion');
end
assert(ok, 'prefix of a region is not a region');
