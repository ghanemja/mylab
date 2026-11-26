// Starter files created the first time someone opens the app. All content is original to mylab.

/** The folder with the examples of what is new in 1.1; they are added once for people who already had the others. */
export const NEW_EXAMPLES_FOLDER = "whats_new";

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
%   * Press "Step through" to walk through a file line by line
%     (try it on step_demo.m).
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
    name: "step_demo.m",
    content: `% Step through this file: press "Step through" in the toolbar, then use
% the arrow buttons (or the left and right arrow keys) to go forwards
% and backwards. Watch the Workspace panel while you step.

total = 0;
for k = 1:4
    total = total + k^2;
    fprintf('k = %d, total = %d\\n', k, total);
end

average = total / 4;
if average > 5
    disp('The average square is bigger than 5.')
else
    disp('The average square is 5 or less.')
end
`,
  },
  {
    name: "sections_demo.m",
    content: `%% Sections and reports
% Lines that start with two percent signs split a file into sections.
% Put the cursor in a section and press Ctrl+Enter to run just that
% section. "Publish" runs every section and builds a report with the
% code, the output and the figures, ready to print or hand in.

%% Make some data
% A falling ball, measured every tenth of a second.
g = 9.81;
t = 0:0.1:2;
height = 20 - 0.5*g*t.^2;
fprintf('The ball lands after about %.2f s\\n', sqrt(2*20/g));

%% Plot it
plot(t, height, 'o-')
xlabel('time (s)'), ylabel('height (m)')
title('A falling ball')
grid on

%% Summary
% The comment lines under a section title become text in the report.
fprintf('Highest point: %.1f m, lowest point: %.1f m\\n', max(height), min(height));
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
  // ---- new in 1.1 (also listed in the What's new dialog) ----
  {
    name: "whats_new/fit_curve.m",
    content: `%% Curve fitting with fit
% fit finds the coefficients of a model that best match your data.
% Here: a cup of tea cooling down, measured every two minutes, fitted
% with the library model 'exp1', which is a*exp(b*x).

rng(3);
minutes = (0:2:40)';
excess = 55*exp(-0.07*minutes) + randn(size(minutes));   % degrees above the room

f = fit(minutes, excess, 'exp1')       % no semicolon: shows the model and its bounds
bounds = confint(f);                    % 95% bounds, one column per coefficient
fprintf('b = %.4f, between %.4f and %.4f\\n', f.b, bounds(1, 2), bounds(2, 2));
fprintf('Half of the excess heat is gone after %.1f minutes.\\n', log(2)/-f.b);

% plot(f, x, y) draws the data points and the fitted curve together.
plot(f, minutes, excess)
xlabel('time (min)'), ylabel('degrees above the room')
title('Cooling with an exponential fit')

%% A model of your own
% fittype turns a formula into a model. Its coefficients are the names
% in the formula other than x. Click f or g in the Workspace panel to see
% the fit in the variable viewer.
ft = fittype('c + a*exp(-x/tau)');
g = fit(minutes, excess, ft, 'StartPoint', [50 1 15]);
fprintf('Own model: tau = %.1f min, offset c = %.2f\\n', g.tau, g.c);
`,
  },
  {
    name: "whats_new/hdf5_files.m",
    content: `%% HDF5 files and version 7.3 MAT-files
% HDF5 is a common format for large scientific data, and version 7.3
% MAT-files are HDF5 files too. The first time a program uses HDF5, the
% app downloads the support for it, which takes a few seconds.

%% Write a dataset, then read it back
hours = 1:24;
temps = 14 + 6*sin((hours - 9)/24*2*pi);      % one day of hourly readings
if isfile('weather.h5'), delete('weather.h5'); end
h5create('weather.h5', '/day1/temperature', [1 24]);
h5write('weather.h5', '/day1/temperature', temps);
h5writeatt('weather.h5', '/day1/temperature', 'units', 'degC');

h5disp('weather.h5')                          % what is in the file
back = h5read('weather.h5', '/day1/temperature');
units = h5readatt('weather.h5', '/day1/temperature', 'units');
fprintf('Read %d values in %s; the warmest hour was %.1f.\\n', numel(back), units, max(back));

%% Save and load a version 7.3 MAT-file
station.name = 'Rooftop';
station.temps = temps;
save('station.mat', 'station', '-v7.3');
clear station
load('station.mat')
fprintf('%s: %d readings, mean %.2f\\n', station.name, numel(station.temps), mean(station.temps));

% weather.h5 and station.mat are now in the Files panel, next to this
% file. Double-click one there to load it into the workspace.
`,
  },
  {
    name: "whats_new/interpolation.m",
    content: `%% griddedInterpolant: an interpolant you call like a function
% Build it once from data on a grid, then evaluate it at any points.

x = 0:5;
y = [0 0.8 0.9 0.1 -0.7 -1];
F = griddedInterpolant(x, y, 'spline');
xq = linspace(0, 5, 101);
plot(x, y, 'o', xq, F(xq), '-')
hold on
F.Method = 'linear';                % same data, another method
plot(xq, F(xq), '--')
hold off
legend('data', 'spline', 'linear', 'Location', 'southwest')
title('One interpolant, two methods')

%% On a 2-D grid
% ndgrid makes the grid; the interpolant takes one coordinate per input.
[X, Y] = ndgrid(-2:2, -2:2);
G = griddedInterpolant(X, Y, X.^2 - Y.^2);
fprintf('G(0.5, 1.5) = %.3f, exact value %.3f\\n', G(0.5, 1.5), 0.5^2 - 1.5^2);
`,
  },
  {
    name: "whats_new/graphics_properties.m",
    content: `%% Graphics objects and their properties
% Plotting functions return handles. Change a plot after drawing it by
% setting properties with a dot, like the fields of a struct.

%% Reference lines with xline and yline
t = 0:0.1:10;
v = 3*(1 - exp(-t/2));
p = plot(t, v, 'LineWidth', 1.5);
p.Color = [0.2 0.5 0.8];
xl = xline(2, '--', 'time constant');
yline(3, ':', 'final value');
xl.Color = 'r';
xlabel('t (s)'), ylabel('voltage (V)')
ylim([0 3.5])
title('Charging a capacitor')

%% Two y-axes with yyaxis
figure
days = 1:7;
yyaxis left
bar(days, [12 15 9 20 18 7 11])
ylabel('visitors')
xlabel('day of the week')
yyaxis right
plot(days, [21 23 19 25 24 18 20], '-o')
ylabel('temperature (C)')
ax = gca;
ax.YAxis(2).Color = [0.85 0.33 0.1];    % the right-hand ruler

%% A pie chart
figure
pie([40 25 20 15], {'rent', 'food', 'travel', 'other'})
title('Where the money goes')
`,
  },
];
