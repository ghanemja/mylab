Write a MATLAB function `m = word_counts(text)` that counts how often each
word occurs in the character vector `text`.

A word is a maximal run of the letters `a`–`z` and `A`–`Z`; every other
character (digits, punctuation, spaces, apostrophes, ...) separates words.
Words are counted case-insensitively and stored in lower case.

Return a `containers.Map` with `KeyType` `'char'` whose keys are the words and
whose values are their counts as doubles. If the text contains no words,
return an empty map (with `Count` 0) of the same key type.

Example:

```matlab
m = word_counts('The cat saw the other cat.');
m('the')   % 2
m('cat')   % 2
m.Count    % 4
```
