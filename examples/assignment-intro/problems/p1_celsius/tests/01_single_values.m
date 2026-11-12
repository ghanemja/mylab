% name: Single temperatures
% points: 8
assert(abs(c2f(0) - 32) < 1e-9, 'c2f(0) should be 32 (water freezes)');
assert(abs(c2f(100) - 212) < 1e-9, 'c2f(100) should be 212 (water boils)');
assert(abs(c2f(-40) - (-40)) < 1e-9, 'c2f(-40) should be -40');
assert(abs(c2f(37) - 98.6) < 1e-9, 'c2f(37) should be 98.6');
