Write a MATLAB function `count = rook_placements(board)`.

`board` is an n-by-n logical matrix describing a chess-like board with holes:
`true` marks a square where a piece may stand, `false` a square that is
missing. Count the number of ways to place exactly `n` rooks on usable squares
so that no two rooks share a row or a column (so every row and every column
holds exactly one rook).

Return the count as a double scalar. For an empty 0-by-0 board the answer is
1 (there is exactly one way to place zero rooks). Boards up to about 9-by-9
must be handled within a second; use backtracking that abandons a partial
placement as soon as a row has no free usable square.

Examples:

- `rook_placements(true(3))` returns `6`.
- `rook_placements(~eye(4))` returns `9`.
- `rook_placements(logical([1 1; 0 0]))` returns `0`.
