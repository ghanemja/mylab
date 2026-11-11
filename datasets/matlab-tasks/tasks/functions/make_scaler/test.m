[f, g] = make_scaler(10, 20);
assert(isequal(f([10 15 25]), [0 0.5 1.5]), 'forward');
assert(g(0.25) == 12.5, 'inverse');
assert(isa(f, 'function_handle') && isa(g, 'function_handle'), 'handles');
x = [3 -1; 7 0.5];
assert(all(all(abs(g(f(x)) - x) < 1e-12)), 'round trip on matrix');
[f, g] = make_scaler(1, -1);
assert(f(0) == 0.5 && f(-1) == 1 && g(0) == 1, 'reversed range');
lo = 100;
[f, ~] = make_scaler(lo, 200);
lo = 0; %#ok<NASGU>
assert(f(150) == 0.5, 'captured at creation');
[f1, ~] = make_scaler(0, 4);
[f2, ~] = make_scaler(0, 8);
assert(f1(2) == 0.5 && f2(2) == 0.25, 'independent closures');
ok = false;
try
    make_scaler(3, 3);
catch err
    ok = strcmp(err.identifier, 'make_scaler:degenerate');
end
assert(ok, 'degenerate range');
