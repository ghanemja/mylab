A tool stores its settings in one line of text such as
`'name=Ada; age = 36 ;active=TRUE'`. Write a MATLAB function

    cfg = parse_key_values(s)

that turns such a char vector into a scalar struct.

- Entries are separated by semicolons. Entries that are empty or contain only
  whitespace are skipped.
- Each entry has the form `key=value`, split at the **first** `=`. Leading
  and trailing whitespace of key and value is removed. An entry without `=`
  throws an error with identifier `parse_key_values:syntax`.
- The key must be a valid MATLAB variable name (see `isvarname`); otherwise
  throw an error with identifier `parse_key_values:badKey`.
- Value conversion: `true`/`false` in any letter case become logical scalars;
  otherwise, if `str2double` turns the value into a number that is not `NaN`,
  store that number; otherwise store the trimmed char vector (possibly empty).
- Fields appear in the order their keys first appear. A repeated key keeps its
  position but takes the later value.
- A string with no entries returns a struct with no fields (`struct()`).

Example:

    cfg = parse_key_values('name=Ada; age = 36 ;active=TRUE')
    % cfg.name = 'Ada', cfg.age = 36, cfg.active = true
