function [B, count] = flood_fill(A, r, c, newValue)
%FLOOD_FILL Fill the 4-connected region of equal values containing A(r,c).
[m, n] = size(A);
if r < 1 || c < 1 || r > m || c > n || r ~= round(r) || c ~= round(c)
    error('flood_fill:index', 'Start position is outside the matrix.');
end
target = A(r, c);
seen = false(m, n);
queue = zeros(m * n, 2);
queue(1, :) = [r c];
seen(r, c) = true;
head = 1;
tail = 1;
moves = [-1 0; 1 0; 0 -1; 0 1];
while head <= tail
    p = queue(head, :);
    head = head + 1;
    for d = 1:4
        q = p + moves(d, :);
        if q(1) >= 1 && q(1) <= m && q(2) >= 1 && q(2) <= n ...
                && ~seen(q(1), q(2)) && A(q(1), q(2)) == target
            seen(q(1), q(2)) = true;
            tail = tail + 1;
            queue(tail, :) = q;
        end
    end
end
B = A;
B(seen) = newValue;
count = tail;
end
