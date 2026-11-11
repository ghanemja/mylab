function [x, hist] = newton_solve(f, df, x0, tol, maxit)
%NEWTON_SOLVE Newton iteration for f(x) = 0, recording every iterate.
hist = zeros(maxit + 1, 1);
hist(1) = x0;
x = x0;
for k = 1:maxit
    d = df(x);
    if d == 0
        error('newton_solve:zeroDerivative', 'Derivative is zero at x = %g.', x);
    end
    xn = x - f(x) / d;
    hist(k + 1) = xn;
    if abs(xn - x) <= tol
        x = xn;
        hist = hist(1:k + 1);
        return
    end
    x = xn;
end
error('newton_solve:noConvergence', 'No convergence in %d iterations.', maxit);
end
