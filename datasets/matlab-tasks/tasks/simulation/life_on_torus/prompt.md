Write a MATLAB function `B = life_on_torus(A, k)` that advances Conway's Game
of Life by `k` generations (k >= 0) on a board that wraps around at the edges.

`A` is an m-by-n logical (or 0/1) matrix of live cells. The board is a torus:
the left neighbour of a cell in column 1 is in column n, the cell above row 1
is in row m, and so on. Each cell has eight neighbours (horizontal, vertical
and diagonal). In each generation, all cells update simultaneously:

- a live cell with 2 or 3 live neighbours stays alive, otherwise it dies;
- a dead cell with exactly 3 live neighbours becomes alive.

Return the board after `k` generations as an m-by-n **logical** matrix. You
may assume m, n >= 3.

Example: a horizontal line of three cells in the middle of a 5-by-5 board
becomes a vertical line after one generation and returns after two.
