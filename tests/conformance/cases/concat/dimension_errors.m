t1 = 0; try, tmp = [1 2; 3]; catch, t1 = 1; end
t2 = 0; try, tmp = [[1; 2], [1 2 3]]; catch, t2 = 1; end
t3 = 0; try, tmp = [ones(2, 2); ones(1, 3)]; catch, t3 = 1; end
t4 = 0; try, tmp = ['ab'; 'c']; catch, t4 = 1; end
t6 = 0; try, tmp = horzcat([1; 2], [1; 2; 3]); catch, t6 = 1; end
t7 = 0; try, tmp = vertcat([1 2], [1 2 3]); catch, t7 = 1; end
t8 = 0; try, tmp = cat(3, [1 2], [1 2 3]); catch, t8 = 1; end
t9 = 0; try, tmp = [struct('a', 1), struct('b', 2)]; catch, t9 = 1; end
