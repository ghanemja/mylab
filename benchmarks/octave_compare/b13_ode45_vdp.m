function r = b13_ode45_vdp()
f = @(t, y) [y(2); (1 - y(1)^2) * y(2) - y(1)];
[t, y] = ode45(f, [0 200], [2; 0], odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
r = y(end, 1);
end
