function count = rook_placements(board)
%ROOK_PLACEMENTS Number of non-attacking full rook placements on usable squares.
n = size(board, 1);
count = place(board, 1, false(1, n), n);
end

function c = place(board, row, used, n)
if row > n
    c = 1;
    return
end
c = 0;
for col = 1:n
    if board(row, col) && ~used(col)
        used(col) = true;
        c = c + place(board, row + 1, used, n);
        used(col) = false;
    end
end
end
