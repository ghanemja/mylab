function r = b12_map_inserts()
m = containers.Map('KeyType', 'double', 'ValueType', 'double');
for k = 1:30000
    m(k) = k * 2;
end
s = 0;
for k = 1:30000
    s = s + m(k);
end
r = s;
end
