Write a MATLAB function

    g = group_words_by_length(words)

where `words` is a cell array (row or column) of char vectors. Group the
words by their number of characters and return a `1-by-K` struct array `g`
with fields

- `len`: the word length of the group;
- `words`: a `1-by-m` cell array with the words of that length, in their
  original order (duplicates are kept).

The groups are sorted by increasing `len`. For an empty `words`, return an
empty struct array that still has the fields `len` and `words`.

Example:

    g = group_words_by_length({'sun', 'moon', 'sky', 'a', 'star', 'sun'});
    [g.len]     % [1 3 4]
    g(2).words  % {'sun', 'sky', 'sun'}
