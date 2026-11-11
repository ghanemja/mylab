t = Tally(10);
increment(t);
u = t;
u.increment(-4);
assert(t.Count == 7, 'handle semantics');
assert(total(t) == 5, 'total moved');
z = Tally();
assert(z.Count == 0 && total(z) == 0, 'default start');
assert(t.Count == 7, 'independent objects');
reset(t);
assert(u.Count == 0, 'reset to zero');
t.increment(2.5);
assert(t.Count == 2.5 && total(u) == 7.5, 'total survives reset');
threw = false;
try
    t.Count = 99;
catch
    threw = true;
end
assert(threw, 'Count is read-only');
assert(t.Count == 2.5, 'value unchanged after failed set');
assert(isa(t, 'handle'), 'handle class');
