function r = b16_text_parsing()
lines = cell(1, 20000);
for k = 1:20000
    lines{k} = sprintf('%d,%g,name%d', k, k / 7, mod(k, 13));
end
total = 0;
for k = 1:numel(lines)
    parts = strsplit(lines{k}, ',');
    total = total + str2double(parts{2}) + numel(parts{3});
end
r = total;
end
