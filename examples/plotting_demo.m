% plotting_demo.m - basic 2-D plots, saved to a PNG file
x = linspace(0, 2*pi, 200);

figure;
plot(x, sin(x), 'b-', 'LineWidth', 2);
hold on
plot(x, cos(x), 'r--');
hold off
xlabel('x (radians)');
ylabel('value');
title('sin and cos');
legend('sin(x)', 'cos(x)');
grid on

saveas(gcf, 'trig.png');
disp('Saved trig.png');
