function d = hop_distances(A, src)
%HOP_DISTANCES Breadth-first search distances from src.
n = size(A, 1);
d = inf(1, n);
d(src) = 0;
queue = src;
head = 1;
while head <= numel(queue)
    u = queue(head);
    head = head + 1;
    nbrs = find(A(u, :));
    for v = nbrs
        if isinf(d(v))
            d(v) = d(u) + 1;
            queue(end + 1) = v; %#ok<AGROW>
        end
    end
end
end
