function idx = insert_position(v, x)
%INSERT_POSITION Leftmost index where x can be inserted into sorted v.
lo = 1;
hi = numel(v) + 1;
while lo < hi
    mid = floor((lo + hi) / 2);
    if v(mid) < x
        lo = mid + 1;
    else
        hi = mid;
    end
end
idx = lo;
end
