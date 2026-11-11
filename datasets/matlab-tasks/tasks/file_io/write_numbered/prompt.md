Write a MATLAB function `n = write_numbered(filename, lines)`.

`lines` is a cell array of character vectors. Create (or overwrite) the text
file `filename` and write one line per element: a line number padded with
zeros to at least three digits, a colon, a space, and then the text. Every
line, including the last, ends with a single newline character (`\n`, char
10). Return the number of lines written as a double.

If `lines` is empty, create an empty file and return 0. The text may contain
characters such as `%` and `\` that must be written literally.

Example: `write_numbered('out.txt', {'alpha', 'beta'})` returns `2` and the
file contains

```
001: alpha
002: beta
```
