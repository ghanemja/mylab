function s = nan_summary(x)
%NAN_SUMMARY Descriptive statistics of the non-NaN entries of x.
v = x(~isnan(x));
v = v(:);
s.count = numel(v);
if isempty(v)
    s.mean = NaN;
    s.median = NaN;
    s.std = NaN;
    s.min = NaN;
    s.max = NaN;
    return
end
s.mean = mean(v);
s.median = median(v);
if numel(v) > 1
    s.std = std(v);
else
    s.std = 0;
end
s.min = min(v);
s.max = max(v);
end
