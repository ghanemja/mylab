assert(apply_repeatedly(@(v) v * 2, 1, 10) == 1024, 'doubling');
assert(strcmp(apply_repeatedly(@(s) [s 'ab'], 'x', 3), 'xababab'), 'char growth');
assert(apply_repeatedly(@sqrt, 81, 0) == 81, 'zero times');
assert(abs(apply_repeatedly(@sqrt, 81, 2) - 3) < 1e-12, 'named function handle');
assert(isequal(apply_repeatedly(@(v) v(2:end), 1:5, 2), [3 4 5]), 'shrinking vector');
assert(isequal(apply_repeatedly(@(v) [v sum(v)], [1 1], 4), [1 1 2 4 8 16]), 'growing vector');
assert(isequal(apply_repeatedly(@(v) v', [1 2 3], 3), [1; 2; 3]), 'odd number of transposes');
ok = false;
try
    apply_repeatedly(@(v) v, 1, -1);
catch err
    ok = strcmp(err.identifier, 'apply_repeatedly:badCount');
end
assert(ok, 'negative count');
