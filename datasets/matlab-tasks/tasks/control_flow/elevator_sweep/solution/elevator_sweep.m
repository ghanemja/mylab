function [order, distance] = elevator_sweep(start, requests)
%ELEVATOR_SWEEP Serve floor requests sweeping one way and then the other.
requests = reshape(requests, 1, []);
order = zeros(1, 0);
distance = 0;
if any(requests == start)
    order = start;
end
others = requests(requests ~= start);
if isempty(others)
    return
end
above = unique(others(others > start));          % ascending
below = fliplr(unique(others(others < start)));  % descending
if others(1) > start
    order = [order, above, below];
    first = above;
    second = below;
else
    order = [order, below, above];
    first = below;
    second = above;
end
pos = start;
if ~isempty(first)
    distance = distance + abs(first(end) - pos);
    pos = first(end);
end
if ~isempty(second)
    distance = distance + abs(second(end) - pos);
end
end
