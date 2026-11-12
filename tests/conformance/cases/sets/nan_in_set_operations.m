% NaN never equals NaN, so every NaN stays in the result and sorts last.
n1 = union([1 NaN], [NaN 2]);
n2 = setdiff([1 NaN 2], NaN);
n3 = intersect([1 NaN 2], [NaN 2]);
n4 = setxor([1 NaN], [NaN 1]);
n6 = ismember(NaN, [1 NaN]);
n7 = union([3 NaN 1], [2 NaN]);
n8 = unique([2 NaN 1 NaN 2]);
