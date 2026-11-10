% linear_algebra.m - common matrix computations
A = [4 -2 1; -2 4 -2; 1 -2 4];

fprintf('det(A)  = %g\n', det(A));
fprintf('rank(A) = %d\n', rank(A));

[V, D] = eig(A);
disp('Eigenvalues:');
disp(diag(D)');

% Check A*V = V*D
residual = norm(A*V - V*D);
fprintf('Eigen residual: %.2e\n', residual);

% Least-squares line fit through noisy points
rng(1);
t = (0:0.5:5)';
y = 3*t + 2 + 0.1*randn(size(t));
coeffs = [t ones(size(t))] \ y;
fprintf('Fitted line: y = %.3f*t + %.3f\n', coeffs(1), coeffs(2));
