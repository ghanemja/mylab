function R = axis_angle_rotation(u, theta)
%AXIS_ANGLE_ROTATION Rodrigues rotation matrix.
u = u(:);
nu = norm(u);
if nu == 0
    error('axis_angle_rotation:axis', 'Axis must be nonzero.');
end
u = u / nu;
K = [0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
R = eye(3) + sin(theta)*K + (1 - cos(theta))*(K*K);
end
