function n = write_numbered(filename, lines)
%WRITE_NUMBERED Write lines prefixed by zero-padded line numbers.
fid = fopen(filename, 'w');
if fid < 0
    error('write_numbered:open', 'Cannot open %s', filename);
end
n = numel(lines);
for k = 1:n
    fprintf(fid, '%s\n', [sprintf('%03d: ', k), lines{k}]);
end
fclose(fid);
end
