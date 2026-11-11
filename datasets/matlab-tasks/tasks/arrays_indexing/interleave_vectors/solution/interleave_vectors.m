function v = interleave_vectors(a, b)
%INTERLEAVE_VECTORS Alternate elements of A and B, appending the leftover tail.
a = reshape(a, 1, []);
b = reshape(b, 1, []);
k = min(numel(a), numel(b));
v = zeros(1, 2 * k);
v(1:2:end) = a(1:k);
v(2:2:end) = b(1:k);
v = [v, a(k+1:end), b(k+1:end)];
end
