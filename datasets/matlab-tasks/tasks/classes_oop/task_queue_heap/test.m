q = TaskQueue();
add(q, 5, 'write'); add(q, 1, 'plan'); add(q, 5, 'test');
assert(q.Count == 3 && peekPriority(q) == 1, 'count and peek');
[n, p] = next(q);
assert(strcmp(n, 'plan') && p == 1, 'most urgent first');
assert(strcmp(next(q), 'write') && strcmp(next(q), 'test') && q.Count == 0, 'FIFO among equal priorities');
threw = false;
try
    next(q);
catch err
    threw = strcmp(err.identifier, 'TaskQueue:empty');
end
assert(threw, 'next on empty');
threw = false;
try
    peekPriority(q);
catch err
    threw = strcmp(err.identifier, 'TaskQueue:empty');
end
assert(threw, 'peek on empty');
% many ties interleaved with other priorities
pri = mod((1:40) * 7, 4);
for k = 1:40
    add(q, pri(k), sprintf('t%02d', k));
end
out = cell(1, 40);
ps = zeros(1, 40);
for k = 1:40
    [out{k}, ps(k)] = next(q);
end
[~, order] = sort(pri);   % stable sort gives the expected order
expected = arrayfun(@(k) sprintf('t%02d', k), order, 'UniformOutput', false);
assert(isequal(out, expected) && isequal(ps, sort(pri)), 'stable priority order');
% interleaved adds and removals, negative and fractional priorities
add(q, 0.5, 'a'); add(q, -2, 'b'); add(q, 0.5, 'c');
assert(strcmp(next(q), 'b'));
add(q, -1, 'd'); add(q, 0.5, 'e');
got = {next(q), next(q), next(q), next(q)};
assert(isequal(got, {'d', 'a', 'c', 'e'}) && q.Count == 0, 'interleaved operations');
% larger queue: 1000 additions and removals
r = TaskQueue();
for k = 1:1000
    add(r, mod(k * 7919, 1000), 'x');
end
last = -Inf;
ok = true;
for k = 1:1000
    [~, pk] = next(r);
    ok = ok && pk >= last;
    last = pk;
end
assert(ok && r.Count == 0, 'heap order on a large queue');
