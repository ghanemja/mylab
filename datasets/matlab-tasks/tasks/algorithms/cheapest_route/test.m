W = [Inf 4 1 Inf; Inf Inf Inf 1; Inf 2 Inf 6; Inf Inf Inf Inf];
[c, p] = cheapest_route(W, 1, 4);
assert(c == 4 && isequal(p, [1 3 2 4]), 'example');
[c, p] = cheapest_route(W, 2, 2);
assert(c == 0 && isequal(p, 2), 'start equals target');
[c, p] = cheapest_route(W, 4, 1);
assert(isinf(c) && isempty(p), 'unreachable');
% direct edge more expensive than a detour; zero-cost edge present
V = inf(5);
V(1,5) = 10; V(1,2) = 3; V(2,3) = 0; V(3,4) = 2; V(4,5) = 1; V(2,5) = 7;
[c, p] = cheapest_route(V, 1, 5);
assert(c == 6 && isequal(p, [1 2 3 4 5]), 'detour with zero edge');
[c, p] = cheapest_route(V, 3, 5);
assert(c == 3 && isequal(p, [3 4 5]), 'middle start');
% directed: reverse direction unavailable
[c, p] = cheapest_route(V, 5, 1);
assert(isinf(c) && isempty(p), 'directed edges');
% grid-like graph where greedy nearest-neighbour fails
G = inf(6);
G(1,2) = 1; G(2,3) = 10; G(1,4) = 2; G(4,5) = 2; G(5,3) = 2; G(3,6) = 1; G(2,6) = 20;
[c, p] = cheapest_route(G, 1, 6);
assert(abs(c - 7) < 1e-12 && isequal(p, [1 4 5 3 6]), 'non-greedy route');
assert(isequal(size(p), [1 5]), 'row path');
