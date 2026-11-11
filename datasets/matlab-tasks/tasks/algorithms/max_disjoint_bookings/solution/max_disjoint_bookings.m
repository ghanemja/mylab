function k = max_disjoint_bookings(slots)
%MAX_DISJOINT_BOOKINGS Greedy earliest-finish interval scheduling.
k = 0;
if isempty(slots)
    return
end
[~, idx] = sort(slots(:, 2));
slots = slots(idx, :);
freeFrom = -Inf;
for r = 1:size(slots, 1)
    if slots(r, 1) >= freeFrom
        k = k + 1;
        freeFrom = slots(r, 2);
    end
end
end
