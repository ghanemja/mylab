function P = lattice_walk(n)
%LATTICE_WALK Positions of a simple random walk on the square lattice.
moves = [1 0; -1 0; 0 1; 0 -1];
pick = randi(4, n, 1);
P = [0 0; cumsum(moves(pick, :), 1)];
end
