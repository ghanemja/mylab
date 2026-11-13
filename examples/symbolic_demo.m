% symbolic_demo.m - exact math with symbolic variables
%   Needs SymPy:  pip install "mylab-lang[symbolic]"

syms x y t

% --- expressions stay exact -------------------------------------------
f = x^3 - 6*x^2 + 11*x - 6;
disp('A cubic and its factors:');
disp(f);
disp(factor(f));
fprintf('One third plus one sixth is %s\n', char(sym(1)/3 + sym(1)/6));

% --- calculus ---------------------------------------------------------
g = exp(-x)*sin(2*x);
dg = diff(g, x);
G = int(g, x);
fprintf('d/dx of %s is %s\n', char(g), char(dg));
fprintf('Area under exp(-x) from 0 to Inf: %s\n', char(int(exp(-x), x, 0, inf)));
fprintf('Check: diff(int(g)) - g simplifies to %s\n', char(simplify(diff(G, x) - g)));
fprintf('limit of (1 - cos(x))/x^2 at 0: %s\n', char(limit((1 - cos(x))/x^2, x, 0)));
disp('Taylor polynomial of cos(x), terms below x^6:');
disp(taylor(cos(x), x, 'Order', 6));

% --- solving equations ------------------------------------------------
roots_f = solve(f == 0, x);
fprintf('Roots of the cubic: %s\n', mat2str(double(roots_f)'));
[sx, sy] = solve([2*x + y == 7, x - y == -1], [x, y]);
fprintf('2x + y = 7 and x - y = -1 gives x = %s, y = %s\n', char(sx), char(sy));

% --- a differential equation -------------------------------------------
syms h(t)
Dh = diff(h, t);
sol = dsolve(diff(h, t, 2) + 4*h == 0, h(0) == 1, Dh(0) == 0);
fprintf('h'''' + 4h = 0, h(0) = 1, h''(0) = 0  ->  h(t) = %s\n', char(sol));

% --- matrices ---------------------------------------------------------
syms a b
M = [a b; b a];
disp('Determinant and eigenvalues of [a b; b a]:');
disp(det(M));
disp(eig(M));

% --- from symbols to numbers and plots --------------------------------
fprintf('pi to 40 digits: %s\n', char(vpa(sym(pi), 40)));
gn = matlabFunction(g);            % a normal function handle
xs = linspace(0, 4, 5);
fprintf('g at %s is %s\n', mat2str(xs), mat2str(gn(xs), 4));

fplot(g, [0 6]);
hold on
fplot(dg, [0 6]);
legend('g(x)', 'dg/dx');
title('A symbolic function and its derivative');
saveas(gcf, 'symbolic_demo.png');
