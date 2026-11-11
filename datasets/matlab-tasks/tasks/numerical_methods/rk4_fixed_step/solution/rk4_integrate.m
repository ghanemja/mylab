function [t, Y] = rk4_integrate(f, tspan, y0, h)
%RK4_INTEGRATE Classical fixed-step fourth-order Runge-Kutta.
t0 = tspan(1);
tf = tspan(2);
N = ceil((tf - t0)/h - 1e-10);
t = t0 + (0:N)' * h;
t(end) = tf;
y = y0(:);
Y = zeros(N + 1, numel(y));
Y(1, :) = y.';
for k = 1:N
    hk = t(k + 1) - t(k);
    k1 = f(t(k), y);
    k2 = f(t(k) + hk/2, y + hk/2*k1);
    k3 = f(t(k) + hk/2, y + hk/2*k2);
    k4 = f(t(k) + hk, y + hk*k3);
    y = y + hk/6*(k1 + 2*k2 + 2*k3 + k4);
    Y(k + 1, :) = y.';
end
end
