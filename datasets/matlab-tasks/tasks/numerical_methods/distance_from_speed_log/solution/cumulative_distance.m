function d = cumulative_distance(t, v)
%CUMULATIVE_DISTANCE Running trapezoidal integral of v over t.
if numel(t) ~= numel(v)
    error('cumulative_distance:size', 't and v must have the same number of elements.');
end
d = zeros(size(v));
dt = diff(t(:));
avg = (v(1:end-1) + v(2:end)) / 2;
d(2:end) = cumsum(dt .* avg(:));
end
