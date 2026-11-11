function [x, y, range] = launch_path(v0, angleDeg, dt)
%LAUNCH_PATH Explicit-Euler projectile trajectory until it falls below ground.
g = 9.81;
vx = v0 * cosd(angleDeg);
vy = v0 * sind(angleDeg);
x = 0;
y = 0;
cx = 0;
cy = 0;
while true
    nx = cx + vx * dt;
    ny = cy + vy * dt;
    vy = vy - g * dt;
    if ny < 0
        break
    end
    x(end + 1, 1) = nx; %#ok<AGROW>
    y(end + 1, 1) = ny; %#ok<AGROW>
    cx = nx;
    cy = ny;
end
range = cx + (nx - cx) * cy / (cy - ny);
end
