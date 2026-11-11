function varargout = describe_vector(x)
%DESCRIBE_VECTOR Min, max, mean and count of non-NaN values.
x = x(~isnan(x));
count = numel(x);
if count == 0
    mn = NaN;
    mx = NaN;
    avg = NaN;
else
    mn = min(x);
    mx = max(x);
    avg = sum(x) / count;
end
if nargout <= 1
    varargout{1} = struct('min', mn, 'max', mx, 'mean', avg, 'count', count);
else
    vals = {mn, mx, avg, count};
    varargout = vals(1:nargout);
end
end
