n1 = cfh_nargs();
n2 = cfh_nargs(1, 2, 3);
[n3, n4] = cfh_nargs('a');
[n5, n6, n7] = cfh_nargs([], {}, 1, 2);
n8 = cfh_nargs(n1, n1);
c = {1, 2, 3, 4};
n9 = cfh_nargs(c{:});
e = {};
n10 = cfh_nargs(e{:});
o1 = cfh_opt(1);
o2 = cfh_opt(1, 2);
o3 = cfh_opt(1, 2, 'c');
o4 = cfh_opt(1, [], []);
s1 = cfh_multi(3, 4);
[s2, p2] = cfh_multi(3, 4);
[s3, p3, d3] = cfh_multi(3, 4);
t1 = 0; try, cfh_multi(1); catch, t1 = 1; end
t2 = 0; try, cfh_multi(1, 2, 3); catch, t2 = 1; end
t3 = 0; try, [a, b, c2, d] = cfh_multi(1, 2); catch, t3 = 1; end
t4 = 0; try, cfh_check(); catch, t4 = 1; end
k1 = cfh_check(1);
k2 = cfh_check(1, 2);
try, na = nargin('cfh_multi'); catch, end
try, no = nargout('cfh_multi'); catch, end
try, na2 = nargin('cfh_nargs'); catch, end
try, no2 = nargout('cfh_nargs'); catch, end
try, na3 = nargin(@cfh_opt); catch, end
