Write a MATLAB function `C = parse_quoted_csv(filename)` that reads a CSV
file whose fields may be quoted, and returns its contents as a cell array of
character vectors.

Rules:

- Each non-empty line of the file is one record. Empty lines are skipped. A
  trailing carriage return (`\r`) at the end of a line is removed first.
- Fields are separated by commas.
- A field that starts with a double quote `"` is a quoted field: it continues
  up to the matching closing quote, and inside it commas are ordinary
  characters and two consecutive quotes `""` stand for one literal `"`. The
  surrounding quotes are not part of the value. A closing quote is always
  followed by a comma or the end of the line.
- Unquoted fields are taken exactly as written (no trimming).
- An empty field (e.g. between two consecutive commas) is an empty character
  vector `''`.
- All records have the same number of fields, and no field spans lines.

Return `C` as an R-by-F cell array where R is the number of records and F the
number of fields. A file with no records returns `cell(0, 0)`.

Example: the line

```
7,"Smith, Jo","said ""hi""",
```

is parsed into `{'7', 'Smith, Jo', 'said "hi"', ''}`.
