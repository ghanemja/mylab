% ode_demo.m - damped harmonic oscillator with ode45
%   x'' + 2*zeta*w*x' + w^2*x = 0, written as a first-order system.
w = 2*pi;      % natural frequency
zeta = 0.1;    % damping ratio
f = @(t, y) [y(2); -2*zeta*w*y(2) - w^2*y(1)];

[t, y] = ode45(f, [0 5], [1; 0]);

fprintf('Solver took %d steps\n', numel(t));
fprintf('Position at t = 5: %.4f\n', y(end, 1));

plot(t, y(:, 1));
xlabel('time (s)'); ylabel('position');
title('Damped oscillator');
saveas(gcf, 'oscillator.png');
