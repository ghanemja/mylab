function y = fill_forward(x)
%FILL_FORWARD Replace NaN with the last preceding non-NaN value.
y = x;
last = NaN;
for k = 1:numel(y)
    if isnan(y(k))
        y(k) = last;
    else
        last = y(k);
    end
end
end
