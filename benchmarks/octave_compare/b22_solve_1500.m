function r = b22_solve_1500()
A = reshape(mod((1:1500^2) * 7919, 1009), 1500, 1500) + 2000 * eye(1500);
b = (1:1500)';
x = A \ b;
r = x(1);
end
