t1 = true(2);
t2 = false(1, 3);
t3 = size(true(0, 2));
t4 = logical([2 0 -1 0.5]);
t6 = logical(int8([0 3]));
t7 = class(true(2));
t8 = true(2, 1)';
t9 = logical([]);
t10 = class(logical([]));
err1 = 0; try, tmp = logical(NaN); catch, err1 = 1; end
err2 = 0; try, tmp = logical([1 NaN]); catch, err2 = 1; end
t11 = logical(Inf);
t12 = double(true) + double(false);
t13 = ~true(1, 2);
t14 = true(1, 2) * 3;
t15 = xor(true, true);
t16 = islogical(true(2) & 1);
t17 = false(2, 0);
t18 = size(t17);
t19 = true;
t20 = false;
