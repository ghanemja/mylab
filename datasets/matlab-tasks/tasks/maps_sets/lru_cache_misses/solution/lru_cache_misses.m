function [misses, contents] = lru_cache_misses(requests, capacity)
%LRU_CACHE_MISSES Count misses of a least-recently-used cache.
contents = zeros(1, 0);
misses = 0;
for k = 1:numel(requests)
    r = requests(k);
    pos = find(contents == r, 1);
    if isempty(pos)
        misses = misses + 1;
        if capacity == 0
            continue
        end
        if numel(contents) >= capacity
            contents(end) = [];
        end
    else
        contents(pos) = [];
    end
    contents = [r, contents]; %#ok<AGROW>
end
end
