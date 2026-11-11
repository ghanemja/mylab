function labels = component_labels(n, edges)
%COMPONENT_LABELS Label connected components of an undirected graph.
parent = 1:n;
for k = 1:size(edges, 1)
    ra = findroot(parent, edges(k, 1));
    rb = findroot(parent, edges(k, 2));
    if ra ~= rb
        parent(max(ra, rb)) = min(ra, rb);
    end
end
labels = zeros(1, n);
next = 0;
rootLabel = zeros(1, n);
for v = 1:n
    r = findroot(parent, v);
    if rootLabel(r) == 0
        next = next + 1;
        rootLabel(r) = next;
    end
    labels(v) = rootLabel(r);
end
end

function r = findroot(parent, v)
r = v;
while parent(r) ~= r
    r = parent(r);
end
end
