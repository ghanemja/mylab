function [lam, v, iters] = power_iteration(A, x0, tol, maxit)
%POWER_ITERATION Dominant eigenvalue and eigenvector by the power method.
v = x0(:) / norm(x0);
lam = v' * A * v;
iters = 0;
for k = 1:maxit
    w = A * v;
    v = w / norm(w);
    lamNew = v' * A * v;
    iters = k;
    done = abs(lamNew - lam) <= tol * abs(lamNew);
    lam = lamNew;
    if done
        break
    end
end
[~, i] = max(abs(v));
if v(i) < 0
    v = -v;
end
end
