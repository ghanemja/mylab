function [x, iters] = newton_minimize(gradf, hessf, x0, tol, maxit)
%NEWTON_MINIMIZE Pure Newton iteration on the gradient.
x = x0(:);
iters = 0;
while true
    g = gradf(x);
    if norm(g) < tol
        return
    end
    if iters >= maxit
        error('newton_minimize:maxit', 'No convergence in %d iterations.', maxit);
    end
    x = x - hessf(x) \ g;
    iters = iters + 1;
end
end
