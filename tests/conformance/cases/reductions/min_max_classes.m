c1 = class(max(int8([1 2])));
c2 = class(max(int8(1), 5));
v2 = max(int8(1), 5.7);
v3 = max(int8(1), 500);
c4 = class(max(single(1), 2));
c6 = class(max('a', 'b'));
v6 = max('a', 'b');
c7 = class(max('abc'));
v7 = max('abc');
c8 = class(max(true, 2));
t9 = 0; try, tmp = min(uint8(5), int8(3)); catch, t9 = 1; end
v11 = min(int8(-5), -200);
c12 = class(max(1, single(2)));
c5 = class(max(true, false));
c10 = class(max([true false]));
