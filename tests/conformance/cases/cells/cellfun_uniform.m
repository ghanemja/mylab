c = {1, [2 3], [], 'four'};
u1 = cellfun(@numel, c);
u2 = cellfun(@isempty, c);
u3 = cellfun(@ischar, c);
u4 = cellfun(@(x) numel(x) * 2, c);
u5 = class(u2);
u6 = cellfun(@numel, {1 2; 3 4});
u7 = cellfun(@(x, y) x + y, {1, 2}, {10, 20});
u8 = cellfun(@numel, {});
su8 = size(u8);
u9 = cellfun(@numel, cell(0, 3));
su9 = size(u9);
u10 = cellfun(@(x) x > 1, {1, 2, 3});
u11 = class(u10);
u12 = cellfun(@(x) 'a', {1, 2});
u13 = class(u12);
u14 = cellfun(@(x) int8(x), {1, 2});
u15 = class(u14);
u16 = cellfun(@numel, c');
su16 = size(u16);
e1 = 0; try, tmp = cellfun(@(x) x, c); catch, e1 = 1; end
e2 = 0; try, tmp = cellfun(@(x, y) x + y, {1, 2}, {1}); catch, e2 = 1; end
e3 = 0; try, tmp = cellfun(@(x) [x x], {1, 2}); catch, e3 = 1; end
