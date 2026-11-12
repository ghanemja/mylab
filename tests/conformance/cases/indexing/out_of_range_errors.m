v = 1:3;
c = {1, 2};
s = struct('a', 1);
t1 = 0; try, tmp = c{3}; catch, t1 = 1; end
t2 = 0; try, tmp = c(3); catch, t2 = 1; end
t3 = 0; try, tmp = s(2); catch, t3 = 1; end
t4 = 0; try, tmp = s.b; catch, t4 = 1; end
t5 = 0; try, tmp = v(2, 2); catch, t5 = 1; end
t6 = 0; try, tmp = v(logical([1 0 0 1])); catch, t6 = 1; end
t7 = 0; try, tmp = v(logical([1 0 0 0])); catch, t7 = 1; end
t8 = 0; try, tmp = v('a'); catch, t8 = 1; end
t9 = 0; try, tmp = v(NaN); catch, t9 = 1; end
t10 = 0; try, tmp = v(Inf); catch, t10 = 1; end
