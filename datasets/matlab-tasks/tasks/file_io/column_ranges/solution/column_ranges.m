function R = column_ranges(infile, outfile)
%COLUMN_RANGES Per-column min and max ignoring NaN, written to a CSV file.
M = readmatrix(infile);
R = [min(M, [], 1); max(M, [], 1)];
writematrix(R, outfile);
end
