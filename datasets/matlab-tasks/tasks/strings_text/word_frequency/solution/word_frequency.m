function [words, counts] = word_frequency(s)
%WORD_FREQUENCY Distinct lowercase words and their counts, most frequent first.
found = regexp(lower(s), '[a-z]+(''[a-z]+)*', 'match');
if isempty(found)
    words = cell(0, 1);
    counts = zeros(0, 1);
    return
end
[u, ~, j] = unique(found);
c = accumarray(j(:), 1);
[~, order] = sort(-c);
words = reshape(u(order), [], 1);
counts = reshape(c(order), [], 1);
end
