function [val, idx] = first_repeat(v)
%FIRST_REPEAT First element whose value occurred earlier in v.
seen = containers.Map('KeyType', 'double', 'ValueType', 'logical');
val = NaN;
idx = 0;
for k = 1:numel(v)
    if isKey(seen, v(k))
        val = v(k);
        idx = k;
        return
    end
    seen(v(k)) = true;
end
end
