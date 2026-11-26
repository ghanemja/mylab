function r = b11_object_methods()
c = Counter();
for k = 1:100000
    c.inc(1);
end
r = c.n;
end
