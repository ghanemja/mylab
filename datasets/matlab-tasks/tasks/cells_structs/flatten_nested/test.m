out = flatten_nested({1, {2, {3, 'abc'}, {}}, [4 5], {6}});
assert(isequal(out, {1, 2, 3, 'abc', [4 5], 6}), 'example');
out = flatten_nested({});
assert(iscell(out) && isequal(size(out), [1 0]), 'empty');
out = flatten_nested({{{}}, {}});
assert(isequal(size(out), [1 0]), 'only empty cells');
out = flatten_nested({[], {''}});
assert(isequal(size(out), [1 2]) && isnumeric(out{1}) && isempty(out{1}) && ischar(out{2}), 'empty values kept');
out = flatten_nested({1, {2}; {3, {4}}, 5});
assert(isequal(out, {1, 3, 4, 2, 5}), 'linear index order of a 2x2 cell');
deep = {7};
for k = 1:30
    deep = {k, deep};
end
out = flatten_nested(deep);
assert(numel(out) == 31 && out{1} == 30 && out{end} == 7, 'deep nesting');
s = struct('a', 1);
out = flatten_nested({s, {true}});
assert(isstruct(out{1}) && islogical(out{2}), 'struct and logical leaves');
