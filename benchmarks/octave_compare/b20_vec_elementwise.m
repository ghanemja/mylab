function r = b20_vec_elementwise()
x = linspace(0, 1, 1e7);
y = sin(x).^2 + cos(x).^2 + x .* x - 2 * x;
r = sum(y);
end
