function r = b09_cell_and_string_build()
c = {};
for k = 1:20000
    c{end + 1} = sprintf('item%d', k);
end
n = 0;
for k = 1:numel(c)
    n = n + numel(strrep(c{k}, 'item', 'x'));
end
r = n;
end
