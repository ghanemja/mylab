c1 = class(int8(1) + 1);
c2 = class(1 + int8(1));
c3 = class(int8(1) + true);
c4 = class(int8(1) + 'a');
c5 = class(single(1) + 1);
c6 = class(single(1) + true);
c7 = class(true + true);
c8 = class('a' + 'b');
c9 = class('a' + 1);
c10 = class(true * 3);
t1 = 0; try, tmp = int8(1) + int16(1); catch, t1 = 1; end
t2 = 0; try, tmp = uint8(1) .* int8(1); catch, t2 = 1; end
t3 = 0; try, tmp = int8(1) + single(1); catch, t3 = 1; end
t4 = 0; try, tmp = int8([1 2]) + [1.5 2.5]; catch, t4 = 1; end
v12 = int8([1 2]) + [1.5 2.5];
v13 = int8(5) + 'a';
v14 = true + true;
v15 = 'a' + 1;
c16 = class(-true);
c17 = class(+true);
c18 = class(-'a');
c19 = class(~5);
c20 = class(int8(1) == 1);
