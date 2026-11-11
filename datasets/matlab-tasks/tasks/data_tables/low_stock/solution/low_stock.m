function R = low_stock(T, threshold)
%LOW_STOCK Rows with Qty below threshold, ascending by Qty (stable).
R = T(T.Qty < threshold, :);
[~, idx] = sort(R.Qty);
R = R(idx, :);
end
