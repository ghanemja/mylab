function [x, w] = gauss_legendre(n)
%GAUSS_LEGENDRE Nodes and weights of the n-point Gauss-Legendre rule.
if ~isscalar(n) || n < 1 || n ~= round(n)
    error('gauss_legendre:n', 'n must be a positive integer.');
end
k = (1:n-1)';
beta = k ./ sqrt(4*k.^2 - 1);
J = diag(beta, 1) + diag(beta, -1);
[V, D] = eig(J);
[x, idx] = sort(diag(D));
w = 2 * V(1, idx)'.^2;
% Symmetrize to remove round-off asymmetry.
x = (x - flipud(x)) / 2;
w = (w + flipud(w)) / 2;
end
