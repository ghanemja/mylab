function T = cooling_profile(T0, Tamb, k, dt, n, method)
%COOLING_PROFILE Integrate Newton's law of cooling with Euler or Heun steps.
rate = @(t, y) -k * (y - Tamb(t));
useHeun = strcmpi(method, 'heun');
if ~useHeun && ~strcmpi(method, 'euler')
    error('cooling_profile:method', 'method must be ''euler'' or ''heun''.');
end
T = zeros(n + 1, 1);
T(1) = T0;
for i = 1:n
    t = (i - 1) * dt;
    s1 = rate(t, T(i));
    if useHeun
        s2 = rate(t + dt, T(i) + dt*s1);
        T(i + 1) = T(i) + dt/2*(s1 + s2);
    else
        T(i + 1) = T(i) + dt*s1;
    end
end
end
