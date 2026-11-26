function r = b32_plot_to_png()
f = figure('Visible', 'off');
x = linspace(0, 10, 2000);
hold on
for k = 1:10
    plot(x, sin(k * x));
end
xlabel('x'); ylabel('y'); title('benchmark'); legend('a');
fn = [tempname '.png'];
print(f, '-dpng', fn);
d = dir(fn); r = d.bytes;
delete(fn); close(f);
end
