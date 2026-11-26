function r = b14_fminsearch_rosen()
f = @(x) 100 * (x(2) - x(1)^2)^2 + (1 - x(1))^2 + 100 * (x(4) - x(3)^2)^2 + (1 - x(3))^2;
x = fminsearch(f, [-1.2 1 -1.2 1], optimset('TolX', 1e-8, 'TolFun', 1e-8, 'MaxFunEvals', 20000, 'MaxIter', 20000));
r = x(1);
end
