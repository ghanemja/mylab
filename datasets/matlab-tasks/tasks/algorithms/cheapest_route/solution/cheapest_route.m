function [cost, path] = cheapest_route(W, s, t)
%CHEAPEST_ROUTE Dijkstra's algorithm on a dense cost matrix.
n = size(W, 1);
dist = inf(1, n);
prev = zeros(1, n);
done = false(1, n);
dist(s) = 0;
for iter = 1:n
    cand = dist;
    cand(done) = Inf;
    [best, u] = min(cand);
    if isinf(best)
        break
    end
    done(u) = true;
    if u == t
        break
    end
    for v = 1:n
        if v ~= u && ~done(v) && ~isinf(W(u, v))
            alt = dist(u) + W(u, v);
            if alt < dist(v)
                dist(v) = alt;
                prev(v) = u;
            end
        end
    end
end
cost = dist(t);
if isinf(cost)
    path = zeros(1, 0);
    return
end
path = t;
while path(1) ~= s
    path = [prev(path(1)), path]; %#ok<AGROW>
end
end
