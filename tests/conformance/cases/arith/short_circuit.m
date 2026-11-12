count = 0;
f = @(x) x;
s1 = true || error('should not run');
s2 = false && error('should not run');
s3 = 1 || 0;
s4 = 0 && 1;
s5 = 2 && 3;
s6 = class(2 && 3);
s7 = [] | 1;
s8 = size([] & 1);
t1 = 0; try, tmp = [1 1] && 1; catch, t1 = 1; end
t2 = 0; try, tmp = [] || 1; catch, t2 = 1; end
x = [];
s9 = isempty(x) || x(1) > 0;
s10 = ~isempty(x) && x(1) > 0;
s11 = 0;
if [1 1 1], s11 = 1; end
s12 = 0;
if [1 0 1], s12 = 1; end
s13 = 0;
if [], s13 = 1; end
s14 = 0;
if 'a' < 'b', s14 = 1; end
s15 = 0;
if [1 2] | [0 0], s15 = 1; end
