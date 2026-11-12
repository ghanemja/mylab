t1 = 0; try, tmp = [1 2 3] + [1 2]; catch, t1 = 1; end
t2 = 0; try, tmp = [1 2; 3 4] * [1 2 3]; catch, t2 = 1; end
t3 = 0; try, tmp = [1 2 3] .* [1 2; 3 4]; catch, t3 = 1; end
t4 = 0; try, tmp = ones(2, 3) + ones(3, 2); catch, t4 = 1; end
t5 = 0; try, tmp = [1 2 3] == [1 2]; catch, t5 = 1; end
t6 = 0; try, tmp = [1 2 3] & [1 0]; catch, t6 = 1; end
t7 = 0; try, tmp = [1 2 3] * [4 5 6]; catch, t7 = 1; end
t8 = 0; try, tmp = [1 2; 3 4] ^ [1 2; 3 4]; catch, t8 = 1; end
t9 = 0; try, tmp = ones(2, 3) ^ 2; catch, t9 = 1; end
t10 = 0; try, tmp = [] + [1 2]; catch, t10 = 1; end
