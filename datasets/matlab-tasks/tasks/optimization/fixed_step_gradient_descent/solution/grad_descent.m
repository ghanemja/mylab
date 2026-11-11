function [x, iters] = grad_descent(gradf, x0, alpha, tol, maxit)
%GRAD_DESCENT Gradient descent with constant step size.
x = x0(:);
iters = 0;
while true
    g = gradf(x);
    if norm(g) < tol || iters >= maxit
        return
    end
    x = x - alpha * g;
    iters = iters + 1;
end
end
