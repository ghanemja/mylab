function r = b33_regexprep_cellstr()
c = arrayfun(@(k) sprintf('file_%05d_v%d.txt', k, mod(k, 9)), 1:50000, 'UniformOutput', false);
d = regexprep(c, '_(\d+)_v(\d)', '-$2-$1');
t = regexp(c, '\d+', 'match');
r = numel(d) + numel(t{end});
end
