function [B, removed] = drop_nan_rows(A)
%DROP_NAN_ROWS Remove rows containing NaN; report their indices.
bad = any(isnan(A), 2);
B = A(~bad, :);
removed = find(bad);
removed = reshape(removed, [], 1);
end
