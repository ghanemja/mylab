Write a MATLAB function `s = read_settings(filename)` that reads a simple
INI-style settings file and returns a struct.

Process the file line by line. Leading and trailing whitespace is removed from
every line first. Then:

- Empty lines, and lines starting with `#` or `;`, are ignored.
- A line of the form `[name]` starts a section called `name`.
- Any other line has the form `key = value`. Split it at the **first** `=`
  and trim whitespace from both parts. The value may itself contain `=`.
  Store it in the field `key` if no section has started yet, or in the field
  `<section>_<key>` otherwise (for example `db_port`).
- Value conversion: if the value is `true` or `false` (any letter case) store
  a logical scalar. Otherwise, if `str2double` turns it into a number that is
  not `NaN`, store that double. Otherwise store the value as a character
  vector (possibly empty).
- If the same field appears more than once, the last value wins.

Section and key names consist of letters, digits and underscores and start
with a letter. A file with no settings returns a struct with no fields.

Example: the file

```
# demo
name = Lab 4
[db]
port = 5432
ssl = TRUE
```

gives `s.name = 'Lab 4'`, `s.db_port = 5432`, `s.db_ssl = true`.
