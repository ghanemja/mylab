function tz = zero_crossing_times(t, x)
%ZERO_CROSSING_TIMES Times where a sampled signal changes sign.
n = numel(x);
tz = zeros(1, 0);
k = 1;
while k < n
    if x(k) ~= 0 && x(k+1) ~= 0
        if x(k) * x(k+1) < 0
            tz(end+1) = t(k) - x(k) * (t(k+1) - t(k)) / (x(k+1) - x(k)); %#ok<AGROW>
        end
        k = k + 1;
    elseif x(k) ~= 0
        i = k + 1;
        j = i;
        while j < n && x(j+1) == 0
            j = j + 1;
        end
        if j < n && x(k) * x(j+1) < 0
            tz(end+1) = t(i); %#ok<AGROW>
        end
        k = j + 1;
    else
        k = k + 1;
    end
end
end
