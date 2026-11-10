// Starter files created the first time someone opens the app. All content is original to mylab.

export const EXAMPLES = [
  {
    name: "welcome.m",
    content: `%% Welcome to mylab!
% mylab runs MATLAB-language code right here in your browser.
% Nothing is installed and nothing is sent to a server: Python and the
% mylab engine run on your own computer through WebAssembly.
%
% Quick tour
%   * Press Ctrl+Enter (or the Run button) to run this file.
%   * Select a few lines and press Ctrl+Shift+Enter to run only those.
%   * Type in the Command Window below and press Enter. Arrow keys
%     bring back earlier commands.
%   * Click a variable in the Workspace panel to see its full value.
%   * Plots show up in the Plots panel, where you can download them.
%   * Your files are saved in this browser. Use the Files panel to add,
%     rename, import or download them.

% --- Matrices --------------------------------------------------------
A = [4 -2; 1 1]          % no semicolon, so the result is shown
[V, D] = eig(A);         % with a semicolon, nothing is printed
fprintf('The eigenvalues of A are %g and %g\\n', D(1,1), D(2,2));

% --- Vectors and element-wise math -----------------------------------
t = linspace(0, 4*pi, 400);
y = exp(-t/6) .* sin(2*t);
fprintf('Peak value: %.3f at t = %.2f\\n', max(y), t(y == max(y)));

% --- A plot ----------------------------------------------------------
plot(t, y, 'LineWidth', 1.5)
hold on
plot(t, exp(-t/6), '--')
plot(t, -exp(-t/6), '--')
hold off
title('A damped oscillation')
xlabel('t'), ylabel('y(t)')
legend('signal', 'envelope')
grid on

% --- Functions in other files ----------------------------------------
% stats_demo.m calls describe_data, which lives in describe_data.m.
% Open stats_demo.m from the Files panel and run it next.
`,
  },
  {
    name: "describe_data.m",
    content: `function [m, s, r] = describe_data(x)
%DESCRIBE_DATA  Print a short summary of a numeric vector.
%   [m, s, r] = describe_data(x) returns the mean, the standard
%   deviation and the range of x, and prints them as a table.
    m = mean(x);
    s = std(x);
    r = max(x) - min(x);
    fprintf('  n      = %d\\n', numel(x));
    fprintf('  mean   = %.4f\\n', m);
    fprintf('  std    = %.4f\\n', s);
    fprintf('  range  = %.4f\\n', r);
end
`,
  },
  {
    name: "stats_demo.m",
    content: `%% Simulated measurements and a fitted line
% This script uses describe_data, a function saved in describe_data.m.

rng(7);                              % make the random numbers repeatable
hours  = 0:0.5:10;
growth = 2.5 + 0.8*hours + randn(size(hours))*0.6;

disp('Summary of the measurements:')
describe_data(growth);

% Fit a straight line (degree-1 polynomial)
p = polyfit(hours, growth, 1);
fprintf('Fitted line: growth = %.3f*hours + %.3f\\n', p(1), p(2));

subplot(2, 1, 1)
plot(hours, growth, 'o', hours, polyval(p, hours), '-')
title('Measurements and fitted line')
xlabel('hours'), ylabel('growth')
grid on

subplot(2, 1, 2)
bar(hours, growth - polyval(p, hours))
title('Residuals')
xlabel('hours')
`,
  },
  {
    name: "surface_demo.m",
    content: `%% A 3-D surface
[X, Y] = meshgrid(linspace(-3, 3, 50));
Z = sin(X) .* cos(Y) .* exp(-(X.^2 + Y.^2)/8);
surf(X, Y, Z)
title('sin(x) cos(y) with a Gaussian fade')
xlabel('x'), ylabel('y'), zlabel('z')
`,
  },
];
