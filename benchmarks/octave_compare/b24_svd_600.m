function r = b24_svd_600()
A = reshape(mod((1:600^2) * 7919, 1009), 600, 600);
s = svd(A);
r = s(1);
end
