v1 = validatestring('app', {'apple', 'banana'});
v2 = validatestring('BAN', {'apple', 'banana'});
t1 = 0; try, validatestring('x', {'apple', 'banana'}); catch, t1 = 1; end
t2 = 0; try, validateattributes(5, {'numeric'}, {'positive'}); catch, t2 = 1; end
t3 = 0; try, validateattributes(-5, {'numeric'}, {'positive'}); catch, t3 = 1; end
t4 = 0; try, validateattributes('a', {'numeric'}, {}); catch, t4 = 1; end
t5 = 0; try, assert(true); catch, t5 = 1; end
t6 = 0; try, assert(false); catch, t6 = 1; end
try, assert(1 == 2, 'my:assert', 'failed with %d', 7); catch ae, am = ae.message; aid = ae.identifier; end
try, assert(false, 'plain assert text'); catch ae2, am2 = ae2.message; end
t7 = 0; try, assert([1 1 0]); catch, t7 = 1; end
t8 = 0; try, assert([1 1 1]); catch, t8 = 1; end
