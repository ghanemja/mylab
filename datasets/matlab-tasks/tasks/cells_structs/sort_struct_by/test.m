s = struct('name', {'Ola', 'Bea', 'Kim', 'Ari'}, 'age', {31, 25, 31, 40});
r = sort_struct_by(s, 'age');
assert(isequal({r.name}, {'Bea', 'Ola', 'Kim', 'Ari'}), 'ascending default, stable tie');
r = sort_struct_by(s, 'age', 'descend');
assert(isequal({r.name}, {'Ari', 'Ola', 'Kim', 'Bea'}), 'descending keeps tie order');
r = sort_struct_by(s, 'name', 'ascend');
assert(isequal({r.name}, {'Ari', 'Bea', 'Kim', 'Ola'}) && isequal([r.age], [40 25 31 31]), 'by text');
r = sort_struct_by(s, 'name', 'descend');
assert(isequal({r.name}, {'Ola', 'Kim', 'Bea', 'Ari'}), 'text descending');
c = struct('id', {'b', 'B', 'a', 'b'}, 'n', {1, 2, 3, 4})';
r = sort_struct_by(c, 'id', 'descend');
assert(isequal(size(r), [4 1]) && isequal([r.n], [1 4 3 2]), 'column, char codes, stable');
e = struct('x', {});
assert(isempty(sort_struct_by(e, 'x')), 'empty struct array');
ok = false;
try
    sort_struct_by(s, 'height');
catch err
    ok = strcmp(err.identifier, 'sort_struct_by:noField');
end
assert(ok, 'missing field');
