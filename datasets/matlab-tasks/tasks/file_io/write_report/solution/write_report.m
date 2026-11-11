function total = write_report(filename, names, values)
%WRITE_REPORT Aligned name/value report with a total line.
W = max(cellfun(@numel, [names(:); {'TOTAL'}])) + 2;
fmt = sprintf('%%-%ds%%10.2f\\n', W);
total = sum(double(values(:)));
fid = fopen(filename, 'w');
if fid < 0
    error('write_report:open', 'Cannot open %s', filename);
end
for k = 1:numel(names)
    fprintf(fid, fmt, names{k}, values(k));
end
fprintf(fid, '%s\n', repmat('-', 1, W + 10));
fprintf(fid, fmt, 'TOTAL', total);
fclose(fid);
end
