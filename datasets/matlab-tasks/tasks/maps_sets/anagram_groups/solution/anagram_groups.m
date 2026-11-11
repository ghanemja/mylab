function groups = anagram_groups(words)
%ANAGRAM_GROUPS Group case-insensitive anagrams in order of first appearance.
where = containers.Map('KeyType', 'char', 'ValueType', 'double');
groups = cell(1, 0);
for k = 1:numel(words)
    key = sort(lower(words{k}));
    if isKey(where, key)
        g = where(key);
        groups{g}{end + 1} = words{k};
    else
        groups{end + 1} = words(k); %#ok<AGROW>
        where(key) = numel(groups);
    end
end
end
