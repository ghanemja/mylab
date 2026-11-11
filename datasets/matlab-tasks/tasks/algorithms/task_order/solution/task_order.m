function order = task_order(n, deps)
%TASK_ORDER Lexicographically smallest topological order, empty on a cycle.
A = false(n);
for k = 1:size(deps, 1)
    A(deps(k, 1), deps(k, 2)) = true;
end
indeg = sum(A, 1);
done = false(1, n);
order = zeros(1, n);
for step = 1:n
    ready = find(~done & indeg == 0, 1);
    if isempty(ready)
        order = zeros(1, 0);
        return
    end
    order(step) = ready;
    done(ready) = true;
    indeg = indeg - A(ready, :);
end
end
