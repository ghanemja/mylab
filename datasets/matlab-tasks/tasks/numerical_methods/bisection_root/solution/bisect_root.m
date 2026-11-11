function [r, nsteps] = bisect_root(f, a, b, tol)
%BISECT_ROOT Root of f in [a, b] by interval halving.
fa = f(a);
fb = f(b);
if fa * fb > 0
    error('bisect_root:bracket', 'f(a) and f(b) must have opposite signs.');
end
nsteps = 0;
if fa == 0
    r = a;
    return
end
if fb == 0
    r = b;
    return
end
while (b - a) > 2 * tol
    m = (a + b) / 2;
    nsteps = nsteps + 1;
    fm = f(m);
    if fm == 0
        r = m;
        return
    end
    if sign(fm) == sign(fa)
        a = m;
        fa = fm;
    else
        b = m;
    end
end
r = (a + b) / 2;
end
