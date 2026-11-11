function pair = first_pair_with_sum(x, target)
%FIRST_PAIR_WITH_SUM First pair (smallest j, then i) with x(i)+x(j) == target.
for j = 2:numel(x)
    for i = 1:j-1
        if x(i) + x(j) == target
            pair = [i j];
            return
        end
    end
end
pair = [];
end
