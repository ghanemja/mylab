function P = pairs_with_sum(v, target)
%PAIRS_WITH_SUM Index pairs i<j whose values add up to target.
pos = containers.Map('KeyType', 'double', 'ValueType', 'any');
P = zeros(0, 2);
for j = 1:numel(v)
    need = target - v(j);
    if isKey(pos, need)
        is = pos(need);
        P = [P; is(:), repmat(j, numel(is), 1)]; %#ok<AGROW>
    end
    if isKey(pos, v(j))
        pos(v(j)) = [pos(v(j)), j];
    else
        pos(v(j)) = j;
    end
end
P = sortrows(P);
end
