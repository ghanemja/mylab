function inv = invert_map(m)
%INVERT_MAP Map each value to the sorted list of keys that have it.
inv = containers.Map('KeyType', 'char', 'ValueType', 'any');
ks = sort(keys(m));
for k = 1:numel(ks)
    v = m(ks{k});
    if isKey(inv, v)
        inv(v) = [inv(v), ks(k)];
    else
        inv(v) = ks(k);
    end
end
end
