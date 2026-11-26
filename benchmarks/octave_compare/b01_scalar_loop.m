function r = b01_scalar_loop()
s = 0;
for k = 1:1000000
    s = s + k^2;
end
r = s;
end
