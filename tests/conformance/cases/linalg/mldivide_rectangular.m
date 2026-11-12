% tol: 1e-8
A = [1 1; 1 2; 1 3; 1 4];
b = [6; 5; 7; 10];
x1 = A \ b;
r1 = norm(A' * (A * x1 - b)) < 1e-9;
B = [1 2 3; 4 5 6];
c = [14; 32];
tmpx2 = B \ c;
res2 = norm(B * tmpx2 - c) < 1e-9;
sx2 = size(tmpx2);
x3 = [1; 2; 3] \ [2; 4; 6];
x4 = pinv(B) * c;
res4 = norm(B * x4 - c) < 1e-9;
x5 = [1 2 3] / [2 4 6];
x6 = A \ [b b];
sx6 = size(x6);
x7 = pinv(A) * b;
