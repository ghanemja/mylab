function r = b07_call_anonymous()
f = @(x) x * 2 + 1;
s = 0;
for k = 1:200000
    s = s + f(k);
end
r = s;
end
