function [m, short] = merge_inventories(names1, qty1, names2, qty2)
%MERGE_INVENTORIES Combine item quantities, dropping zero totals.
m = containers.Map('KeyType', 'char', 'ValueType', 'double');
names = [names1(:); names2(:)];
qty = [qty1(:); qty2(:)];
for k = 1:numel(names)
    if isKey(m, names{k})
        m(names{k}) = m(names{k}) + qty(k);
    else
        m(names{k}) = qty(k);
    end
end
ks = keys(m);
for k = 1:numel(ks)
    if m(ks{k}) == 0
        remove(m, ks{k});
    end
end
ks = keys(m);
short = cell(1, 0);
for k = 1:numel(ks)
    if m(ks{k}) < 0
        short{end + 1} = ks{k}; %#ok<AGROW>
    end
end
short = sort(short);
end
