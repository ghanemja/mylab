Write a MATLAB function `n = count_file_words(filename)` that returns the
number of words in the text file `filename`.

A word is a maximal run of characters that are not whitespace (space, tab,
carriage return or newline). Punctuation attached to a word is part of that
word, so `don't` and `end.` are one word each, and a lone `-` surrounded by
spaces is a word too. An empty file, or one containing only whitespace, has
0 words. The result is a double.

Example: a file containing

```
The quick   brown fox
	jumps -- over it.
```

has 8 words.
