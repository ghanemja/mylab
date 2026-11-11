function [x, fx, iters] = nelder_mead(f, x0, step, tol, maxit)
%NELDER_MEAD Derivative-free minimization with the Nelder-Mead simplex.
n = numel(x0);
V = repmat(x0(:)', n + 1, 1);
for i = 1:n
    V(i + 1, i) = V(i + 1, i) + step;
end
F = zeros(n + 1, 1);
for i = 1:n + 1
    F(i) = f(reshape(V(i, :), size(x0)));
end
fv = @(v) f(reshape(v, size(x0)));
iters = 0;
while iters < maxit
    [F, idx] = sort(F);
    V = V(idx, :);
    spread = max(max(abs(V - V(1, :))));
    if F(end) - F(1) <= tol && spread <= tol
        break
    end
    iters = iters + 1;
    xb = V(1, :);
    xw = V(end, :);
    xc = mean(V(1:n, :), 1);
    xr = xc + (xc - xw);
    fr = fv(xr);
    if fr < F(1)
        xe = xc + 2*(xc - xw);
        fe = fv(xe);
        if fe < fr
            V(end, :) = xe; F(end) = fe;
        else
            V(end, :) = xr; F(end) = fr;
        end
    elseif fr < F(n)
        V(end, :) = xr; F(end) = fr;
    else
        if fr < F(end)
            xo = xc + 0.5*(xr - xc);
            fo = fv(xo);
            accept = fo <= fr;
            xnew = xo; fnew = fo;
        else
            xi = xc + 0.5*(xw - xc);
            fi = fv(xi);
            accept = fi < F(end);
            xnew = xi; fnew = fi;
        end
        if accept
            V(end, :) = xnew; F(end) = fnew;
        else
            for i = 2:n + 1
                V(i, :) = xb + 0.5*(V(i, :) - xb);
                F(i) = fv(V(i, :));
            end
        end
    end
end
[F, idx] = sort(F);
V = V(idx, :);
x = reshape(V(1, :), size(x0));
fx = F(1);
end
