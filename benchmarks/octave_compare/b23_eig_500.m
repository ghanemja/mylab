function r = b23_eig_500()
A = reshape(mod((1:500^2) * 7919, 1009), 500, 500);
e = eig(A);
r = abs(e(1));
end
