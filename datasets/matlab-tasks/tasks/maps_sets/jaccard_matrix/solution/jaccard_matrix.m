function J = jaccard_matrix(sets)
%JACCARD_MATRIX Pairwise Jaccard similarity of numeric sets.
n = numel(sets);
J = ones(n);
for i = 1:n
    a = unique(sets{i});
    for j = i + 1:n
        b = unique(sets{j});
        u = numel(union(a, b));
        if u == 0
            s = 1;
        else
            s = numel(intersect(a, b)) / u;
        end
        J(i, j) = s;
        J(j, i) = s;
    end
end
end
