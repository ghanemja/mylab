Write a MATLAB function `groups = anagram_groups(words)`.

`words` is a cell array of non-empty character vectors (letters only). Two
words belong to the same group when they are anagrams of each other ignoring
letter case, i.e. they contain the same letters with the same multiplicities
after conversion to lower case.

Return a 1-by-G cell array `groups`. Each element is a 1-by-k cell array
holding the words of one group exactly as given (original case), in the order
they appear in `words`. The groups themselves are ordered by the position of
their first word in `words`. Duplicate words are kept. An empty input returns
`cell(1,0)`.

Example:

```matlab
g = anagram_groups({'listen', 'Stone', 'silent', 'notes', 'cat', 'Tinsel'});
% g = { {'listen','silent','Tinsel'}, {'Stone','notes'}, {'cat'} }
```
