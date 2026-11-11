fns = {@(v) v(10), @(v) v(2) * 100, @(v) v(1)};
[y, idx] = first_success(fns, [4 5 6]);
assert(y == 500 && idx == 2, 'example');
[y, idx] = first_success({@(v) upper(v)}, 'abc');
assert(strcmp(y, 'ABC') && idx == 1, 'first handle works');
fns = {@(v) error('t:a', 'nope'), @(v) [v v], @(v) error('t:b', 'never called')};
[y, idx] = first_success(fns, [1 2]);
assert(isequal(y, [1 2 1 2]) && idx == 2, 'stops at first success');
fns = {@(v) error('t:a', 'x'), @(v) v{1}, @(v) v(0)};
ok = false;
try
    first_success(fns, 3);
catch err
    ok = strcmp(err.identifier, 'first_success:allFailed') && ~isempty(strfind(err.message, '3'));
end
assert(ok, 'all failed');
ok = false;
try
    first_success({}, 1);
catch err
    ok = strcmp(err.identifier, 'first_success:allFailed') && ~isempty(strfind(err.message, '0'));
end
assert(ok, 'empty list fails');
[y, idx] = first_success({@(v) [v, [1 2 3]], @(v) v * 2}, [1; 1]);
assert(isequal(y, [2; 2]) && idx == 2, 'dimension mismatch is an error');
