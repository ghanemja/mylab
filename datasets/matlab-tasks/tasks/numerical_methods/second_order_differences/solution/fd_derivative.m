function dy = fd_derivative(y, h)
%FD_DERIVATIVE Second-order accurate derivative of uniformly sampled data.
n = numel(y);
if n < 3
    error('fd_derivative:short', 'Need at least 3 samples.');
end
dy = zeros(size(y));
dy(2:n-1) = (y(3:n) - y(1:n-2)) / (2*h);
dy(1) = (-3*y(1) + 4*y(2) - y(3)) / (2*h);
dy(n) = (3*y(n) - 4*y(n-1) + y(n-2)) / (2*h);
end
