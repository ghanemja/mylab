% tol: 1e-3
m1 = fminsearch(@(x) (x - 2)^2, 0);
m2 = fminsearch(@(v) (v(1) - 1)^2 + (v(2) + 2)^2, [0 0]);
[m3, f3] = fminsearch(@(x) (x - 2)^2 + 3, 0);
b1 = fminbnd(@(x) (x - 1)^2, -3, 3);
[b2, fb2] = fminbnd(@cos, 0, 2 * pi);
b3 = fminbnd(@(x) x, 1, 2);
sm2 = size(m2);
m4 = fminsearch(@(v) sum((v - [1; 2; 3]).^2), [0; 0; 0]);
sm4 = size(m4);
