function r = b06_call_user_function()
s = 0;
for k = 1:200000
    s = s + affine(k, 2, 1);
end
r = s;
end
