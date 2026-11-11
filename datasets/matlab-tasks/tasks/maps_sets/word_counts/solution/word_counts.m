function m = word_counts(text)
%WORD_COUNTS Case-insensitive word frequencies in a containers.Map.
m = containers.Map('KeyType', 'char', 'ValueType', 'double');
words = regexp(lower(text), '[a-z]+', 'match');
for k = 1:numel(words)
    w = words{k};
    if isKey(m, w)
        m(w) = m(w) + 1;
    else
        m(w) = 1;
    end
end
end
