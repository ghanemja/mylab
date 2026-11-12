A = [4 7; 2 6];
i1 = inv(A);
d1 = det(A);
r1 = rank(A);
i2 = inv(2);
d2 = det(2);
d3 = det([]);
d4 = det(eye(3));
d5 = det([1 2; 2 4]);
r2 = rank([1 2; 2 4]);
r3 = rank(zeros(3));
r4 = rank([]);
r5 = rank(5);
r6 = rank(0);
r7 = rank([1 2 3; 4 5 6]);
r8 = rank(magic(4));
i3 = inv(eye(3) * 2);
i4 = A * inv(A);
d6 = det([1 2 3; 4 5 6; 7 8 10]);
t1 = trace(A);
t2 = trace(5);
t3 = trace([]);
t4 = trace(magic(3));
i5 = inv([1 2; 3 4]);
i6 = size(inv([]));
err1 = 0; try, tmp = inv([1 2 3; 4 5 6]); catch, err1 = 1; end
err2 = 0; try, tmp = det([1 2 3; 4 5 6]); catch, err2 = 1; end
