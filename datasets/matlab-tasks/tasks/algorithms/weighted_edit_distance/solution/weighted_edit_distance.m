function d = weighted_edit_distance(a, b, costIns, costDel, costSub)
%WEIGHTED_EDIT_DISTANCE Minimum cost edit script from a to b.
m = numel(a);
n = numel(b);
D = zeros(m + 1, n + 1);
D(:, 1) = (0:m)' * costDel;
D(1, :) = (0:n) * costIns;
for i = 1:m
    for j = 1:n
        if a(i) == b(j)
            sub = D(i, j);
        else
            sub = D(i, j) + costSub;
        end
        D(i + 1, j + 1) = min([sub, D(i, j + 1) + costDel, D(i + 1, j) + costIns]);
    end
end
d = D(m + 1, n + 1);
end
