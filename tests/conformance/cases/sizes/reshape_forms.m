A = 1:12;
r1 = reshape(A, 3, 4);
r2 = reshape(A, [3 4]);
r3 = reshape(A, [], 4);
r4 = reshape(A, 3, []);
r5 = size(reshape(A, 2, 3, 2));
r6 = reshape(A, 2, [], 2);
s6 = size(r6);
r7 = reshape(A, 12, 1);
r8 = reshape(r1, 1, []);
r9 = size(reshape([], 0, 5));
r10 = size(reshape(zeros(0, 3), 3, 0));
r11 = reshape('abcdef', 2, 3);
r12 = reshape({1, 2, 3, 4}, 2, 2);
r13 = size(reshape(A, 2, 2, 3, 1));
r14 = size(reshape(A, [2 2 3 1 1]));
r15 = reshape(5, 1, 1);
r16 = reshape(r1, [], 1)';
r17 = size(reshape(A, 1, 1, []));
t1 = 0; try, tmp = reshape(A, 5, []); catch, t1 = 1; end
t2 = 0; try, tmp = reshape(A, 3, 5); catch, t2 = 1; end
t3 = 0; try, tmp = reshape(A, [], []); catch, t3 = 1; end
r18 = reshape(int8(1:4), 2, 2);
c18 = class(r18);
r19 = size(reshape(1:6, [], 3, 2));
