[v1, v2, v3] = cfh_varout(3);
v4 = cfh_varout(1);
t1 = 0; try, [tmpx1, tmpx2] = cfh_varout(1); catch, t1 = 1; end
[lo, hi] = cfh_minmax([3 1 2]);
[lo2, hi2, e1, e2] = cfh_minmax([3 1 2], 10, 20);
t2 = 0; try, [tmpa, tmpb, tmpc] = cfh_minmax([3 1 2]); catch, t2 = 1; end
[p, q] = cfh_swap(1, 2);
r1 = cfh_apply(@max, [4 9 2]);
[r2, r3] = cfh_apply(@max, [4 9 2]);
r4 = cfh_apply(@plus, 1, 2);
[r5, r6] = cfh_apply(@size, ones(2, 3));
r7 = cfh_apply(@() 42);
c = cell(1, 2);
[c{:}] = cfh_varout(2);
t3 = 0; try, y = cfh_noout(1); catch, t3 = 1; end
t4 = 0; try, y = cfh_unset(false); catch, t4 = 1; end
u = cfh_unset(true);
cfh_varout(0);
orig = [1 2 3];
changed = cfh_modify(orig);
