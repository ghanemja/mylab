a = [1 0 2 0];
b = [1 1 0 0];
l1 = a & b;
l2 = a | b;
l3 = xor(a, b);
l4 = ~a;
l5 = and(a, b);
l6 = or(a, b);
l7 = not(a);
l8 = class(a & b);
l9 = a & 1;
l10 = a | 0;
l11 = any(a & b);
l12 = all(a | b);
l13 = xor(true, [true false]);
t1 = 0; try, tmp = NaN & 1; catch, t1 = 1; end
t2 = 0; try, tmp = ~NaN; catch, t2 = 1; end
l14 = 'a' & 1;
l15 = int8(5) | 0;
l16 = [] & [];
