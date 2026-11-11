function [v, Isrc] = node_voltages(edges, nnodes, vs)
%NODE_VOLTAGES Nodal analysis of a resistor network driven at node 1.
N = nnodes + 1;                 % matrix index = node number + 1
G = zeros(N, N);
for r = 1:size(edges, 1)
    i = edges(r, 1) + 1;
    j = edges(r, 2) + 1;
    g = 1 / edges(r, 3);
    G(i, i) = G(i, i) + g;
    G(j, j) = G(j, j) + g;
    G(i, j) = G(i, j) - g;
    G(j, i) = G(j, i) - g;
end
full = zeros(N, 1);
full(2) = vs;
unk = 3:N;
if ~isempty(unk)
    full(unk) = G(unk, unk) \ (-G(unk, 2) * vs);
end
v = full(2:end);
Isrc = G(2, :) * full;
end
