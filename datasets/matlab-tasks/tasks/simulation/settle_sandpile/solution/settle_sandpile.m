function [G, topples] = settle_sandpile(G)
%SETTLE_SANDPILE Topple squares with >= 4 grains until the pile is stable.
G = double(G);
topples = 0;
[m, n] = size(G);
while any(G(:) >= 4)
    T = floor(G / 4);          % number of topplings for each square this sweep
    topples = topples + sum(T(:));
    G = G - 4 * T;
    G(2:m, :) = G(2:m, :) + T(1:m-1, :);
    G(1:m-1, :) = G(1:m-1, :) + T(2:m, :);
    G(:, 2:n) = G(:, 2:n) + T(:, 1:n-1);
    G(:, 1:n-1) = G(:, 1:n-1) + T(:, 2:n);
end
end
