function B = life_on_torus(A, k)
%LIFE_ON_TORUS Game of Life generations with wrap-around edges.
B = logical(A);
for g = 1:k
    N = zeros(size(B));
    for dr = -1:1
        for dc = -1:1
            if dr ~= 0 || dc ~= 0
                N = N + circshift(B, [dr dc]);
            end
        end
    end
    B = (B & (N == 2 | N == 3)) | (~B & N == 3);
end
end
