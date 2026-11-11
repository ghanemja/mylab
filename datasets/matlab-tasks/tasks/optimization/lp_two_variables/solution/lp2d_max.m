function [x, fval] = lp2d_max(c, A, b)
%LP2D_MAX Maximize c'*x over a 2-D polygon by checking every vertex.
c = c(:);
b = b(:);
G = [A; -eye(2)];            % all constraints as G*x <= h
h = [b; 0; 0];
tol = 1e-9 * max(1, max(abs(b)));
m = size(G, 1);
fval = -Inf;
x = [0; 0];
for i = 1:m-1
    for j = i+1:m
        M = G([i j], :);
        if abs(det(M)) < 1e-12
            continue
        end
        p = M \ h([i j]);
        if all(G*p <= h + tol)
            v = c' * p;
            if v > fval
                fval = v;
                x = p;
            end
        end
    end
end
end
