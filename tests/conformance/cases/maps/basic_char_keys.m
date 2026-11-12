m = containers.Map();
m('one') = 1;
m('two') = 2;
m('three') = 3;
k1 = keys(m);
v1 = values(m);
n1 = m.Count;
n2 = length(m);
t1 = isKey(m, 'one');
t2 = isKey(m, 'four');
t3 = isKey(m, {'one', 'x', 'two'});
g1 = m('two');
m('two') = 22;
g2 = m('two');
remove(m, 'one');
k2 = keys(m);
n4 = m.Count;
e1 = 0; try, tmp = m('missing'); catch, e1 = 1; end
v2 = values(m, {'two'});
c1 = class(m);
kt = m.KeyType;
vt = m.ValueType;
i1 = isobject(m);
e2 = 0; try, remove(m, 'nothere'); catch, e2 = 1; end
