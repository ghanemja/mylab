function g = group_words_by_length(words)
%GROUP_WORDS_BY_LENGTH Struct array of words grouped by length.
g = struct('len', {}, 'words', {});
if isempty(words)
    return
end
lens = cellfun(@numel, words);
lens = reshape(lens, 1, []);
words = reshape(words, 1, []);
u = unique(lens);
for k = 1:numel(u)
    g(k).len = u(k);
    g(k).words = words(lens == u(k));
end
end
