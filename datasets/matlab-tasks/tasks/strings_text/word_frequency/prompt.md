Write a MATLAB function

    [words, counts] = word_frequency(s)

that counts how often each word occurs in the char vector `s`. A word is a
run of the ASCII letters `a-z`/`A-Z`, which may contain single apostrophes
between letters (so `cat's` is one word, and in `'tis` the leading
apostrophe is not part of the word). Words are compared case-insensitively
and returned in lowercase.

- `words` is an `m-by-1` **cell array** of char vectors with the distinct
  words.
- `counts` is an `m-by-1` double column vector with their counts.
- Sort by count, highest first; break ties by alphabetical (character code)
  order of the words.
- If `s` contains no words, return a `0-by-1` cell and a `0-by-1` double.

Example:

    [w, c] = word_frequency('The cat saw the dog; the DOG ran.')
    % w = {'the'; 'dog'; 'cat'; 'ran'; 'saw'}, c = [3; 2; 1; 1; 1]
