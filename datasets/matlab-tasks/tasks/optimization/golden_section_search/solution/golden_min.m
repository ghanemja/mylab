function [xmin, fmin, nevals] = golden_min(f, a, b, tol)
%GOLDEN_MIN Golden-section search for the minimum of a unimodal function.
g = (sqrt(5) - 1) / 2;
c = b - g*(b - a);
d = a + g*(b - a);
fc = f(c);
fd = f(d);
nevals = 2;
while (b - a) > tol
    if fc < fd
        b = d;
        d = c;
        fd = fc;
        c = b - g*(b - a);
        fc = f(c);
    else
        a = c;
        c = d;
        fc = fd;
        d = a + g*(b - a);
        fd = f(d);
    end
    nevals = nevals + 1;
end
xmin = (a + b) / 2;
fmin = f(xmin);
nevals = nevals + 1;
end
