function [mask, cleaned] = mad_outliers(x, thresh)
%MAD_OUTLIERS Flag outliers using the scaled median absolute deviation.
med = median(x(:));
smad = 1.4826 * median(abs(x(:) - med));
if smad == 0
    mask = x ~= med;
else
    mask = abs(x - med) > thresh * smad;
end
cleaned = x(~mask);
end
