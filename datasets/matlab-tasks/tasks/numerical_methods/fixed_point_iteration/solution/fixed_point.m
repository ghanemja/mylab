function [x, iters, converged] = fixed_point(g, x0, tol, maxit)
%FIXED_POINT Iterate x = g(x) until successive iterates agree.
x = x0;
converged = false;
iters = 0;
for k = 1:maxit
    xn = g(x);
    iters = k;
    if max(abs(xn(:) - x(:))) < tol
        x = xn;
        converged = true;
        return
    end
    x = xn;
end
end
