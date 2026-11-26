function r = b08_struct_fields()
s.count = 0; s.total = 0; s.v = zeros(1, 100);
for k = 1:200000
    s.count = s.count + 1;
    s.total = s.total + k;
    s.v(mod(k, 100) + 1) = k;
end
r = s.total;
end
