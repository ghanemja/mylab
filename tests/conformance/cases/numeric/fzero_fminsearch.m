% tol: 1e-6
z1 = fzero(@(x) x^2 - 2, [0 2]);
z2 = fzero(@(x) x^2 - 2, 1);
z3 = fzero(@cos, 1);
z4 = fzero(@(x) x^3 - 2*x - 5, 2);
[z5, fv5] = fzero(@(x) x - 1, 0);
fv5small = abs(fv5) < 1e-10;
[z6, fv6, flag6] = fzero(@(x) exp(x) - 3, [0 2]);
z7 = fzero(@(x) x, [-1 1]);
t1 = 0; try, tmp = fzero(@(x) x^2 + 1, [0 1]); catch, t1 = 1; end
z8 = fzero(@(x) sin(x), 3);
z9 = fzero(@(x) (x - 1.5) * (x + 100), [0 10]);
