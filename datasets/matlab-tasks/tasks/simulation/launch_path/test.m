[x, y, r] = launch_path(10, 90, 0.5);
assert(isequal(size(x), [6 1]) && isequal(size(y), [6 1]), 'example sizes (columns)');
assert(all(x == 0) && max(abs(y - [0; 5; 7.5475; 7.6425; 5.285; 0.475])) < 1e-9 && abs(r) < 1e-12, 'example');
% compare with the closed form of the Euler recursion
v0 = 10; a = 45; dt = 0.1;
[x, y, r] = launch_path(v0, a, dt);
vx = v0 * cosd(a); vy = v0 * sind(a); g = 9.81;
K = floor(2 * vy / (g * dt)) + 1;
k = (0:K)';
assert(numel(y) == K + 1, 'number of recorded points');
assert(max(abs(x - vx * dt * k)) < 1e-9, 'x positions');
assert(max(abs(y - (vy * dt * k - g * dt^2 * k .* (k - 1) / 2))) < 1e-9, 'y positions');
yn = vy * dt * (K + 1) - g * dt^2 * (K + 1) * K / 2;
assert(abs(r - (vx * dt * K + vx * dt * y(end) / (y(end) - yn))) < 1e-9 && all(y >= 0), 'interpolated range');
[x, y, r] = launch_path(30, 30, 0.01);
assert(abs(r - 30^2 * sind(60) / 9.81) < 0.5, 'close to the exact range for a small step');
assert(issorted(x) && y(1) == 0 && x(1) == 0, 'monotone x, starts at origin');
