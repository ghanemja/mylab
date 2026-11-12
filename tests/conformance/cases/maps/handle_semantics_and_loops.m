m = containers.Map();
m('a') = 1;
alias = m;
alias('b') = 2;
n1 = m.Count;
ks = keys(m);
total = 0;
for i = 1:numel(ks)
    total = total + m(ks{i});
end
counts = containers.Map();
words = {'x', 'y', 'x', 'z', 'x', 'y'};
for i = 1:numel(words)
    w = words{i};
    if isKey(counts, w)
        counts(w) = counts(w) + 1;
    else
        counts(w) = 1;
    end
end
ck = keys(counts);
cv = cell2mat(values(counts));
m2 = containers.Map({'p'}, {struct('f', 1)});
s = m2('p');
f = s.f;
m3 = containers.Map();
m3('list') = [1 2 3];
m3('list') = [m3('list') 4];
l3 = m3('list');
e1 = isempty(containers.Map());
c1 = m.isKey('a');
c2 = m.keys();
c3 = m.values();
c4 = m.length();
