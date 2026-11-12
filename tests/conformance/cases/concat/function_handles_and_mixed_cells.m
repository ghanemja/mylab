f = @sin;
g = @(x) x + 1;
c = {f, g};
n = numel(c);
r1 = c{2}(1);
r2 = c{1}(0);
k = class(c{1});
d = {1, 'a', {2}, [], @cos, true, int8(3)};
classes = cellfun(@class, d, 'UniformOutput', false);
t1 = 0; try, tmp = [f g]; catch, t1 = 1; end
