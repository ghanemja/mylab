vals = {5, [1 2 3], [1; 2; 3], [1 2; 3 4], [], zeros(1, 0), zeros(0, 1), zeros(2, 2, 2), 'a', 'abc', {}, {1, 2}, zeros(1, 1, 3), zeros(0, 3)};
p1 = cellfun(@isscalar, vals);
p2 = cellfun(@isvector, vals);
p3 = cellfun(@isrow, vals);
p4 = cellfun(@iscolumn, vals);
p5 = cellfun(@ismatrix, vals);
p6 = cellfun(@isempty, vals);
