yq = local_linear_smooth([0 1 2 3], [1 3 5 7], [0.5 4], 1);
assert(isequal(size(yq), [1 2]) && all(abs(yq - [2 9]) < 1e-12), 'example');
x = linspace(0, 10, 21);
y = sin(x) + 0.3*cos(3.7*x) + 0.05*x;
xq = [0 0.25 2.5 5 7.75 10 11];
y1 = local_linear_smooth(x, y, xq, 0.8);
assert(all(abs(y1 - [0.25817260094742744 0.3721434470897271 0.5565556394802282 -0.4427943959656134 1.1075896014049436 0.059584109766052884 -0.29000755467229206]) < 1e-10), 'bandwidth 0.8');
y2 = local_linear_smooth(x', y', xq', 3.0);
assert(isequal(size(y2), [7 1]) && all(abs(y2 - [0.7029189838925477; 0.6670295799771299; 0.3598431203827775; 0.29421070997713733; 0.6025598396457663; 0.8182374217312322; 0.7858839969627402]) < 1e-10), 'bandwidth 3, column query');
yu = local_linear_smooth([0 0.1 0.3 1.5 2 4.2 4.3], [1 1.4 0.9 2.2 2.8 4.1 3.6], [0.2 1 3 4.25], 0.5);
assert(all(abs(yu - [1.12614537772445 1.7813768032014072 3.28242529060933 3.8497601562643182]) < 1e-10), 'irregular spacing');
yl = local_linear_smooth(x, 4 - 0.5*x, [-1 0.3 6.6 11], 1);
assert(all(abs(yl - (4 - 0.5*[-1 0.3 6.6 11])) < 1e-9), 'straight line reproduced, even when extrapolating');
yc = local_linear_smooth([2 2 2], [1 4 7], [0 5], 1);
assert(all(abs(yc - 4) < 1e-12), 'identical x values fall back to the weighted mean');
Q = local_linear_smooth(x, y, [1 2; 3 4], 1);
assert(isequal(size(Q), [2 2]), 'output has the shape of xq');
threw = false;
try
    local_linear_smooth([1 2 3], [1 2 3], 2, 0);
catch
    threw = true;
end
assert(threw, 'zero bandwidth must throw');
threw = false;
try
    local_linear_smooth([1 2 3], [1 2], 2, 1);
catch
    threw = true;
end
assert(threw, 'length mismatch must throw');
