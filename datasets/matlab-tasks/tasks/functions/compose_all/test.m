h = compose_all(@(x) x + 1, @(x) x * 2);
assert(h(3) == 7, 'order of application');
h = compose_all(@(x) x * 2, @(x) x + 1);
assert(h(3) == 8, 'reversed order');
h = compose_all(@upper, @fliplr);
assert(strcmp(h('abc'), 'CBA'), 'char functions');
h = compose_all();
assert(h(5) == 5 && isequal(h('ab'), 'ab'), 'identity');
h = compose_all(@numel);
assert(h(1:7) == 7, 'single function');
h = compose_all(@sum, @(v) v.^2, @(v) v(v > 0));
assert(h([-3 1 2 -1 3]) == 14, 'three functions');
fs = repmat({@(x) x + 1}, 1, 50);
h = compose_all(fs{:});
assert(h(0) == 50, 'many functions');
ok = false;
try
    compose_all(@sin, 3);
catch err
    ok = strcmp(err.identifier, 'compose_all:notHandle');
end
assert(ok, 'non-handle input');
