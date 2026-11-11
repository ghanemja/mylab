Write a MATLAB function

    s = name_initials(name)

that returns the initials of a person's name as an uppercase char row vector.
`name` is a char vector whose parts are separated by spaces and/or hyphens
(any number of them, possibly with leading or trailing separators). The
initial of every part is its first character, converted to uppercase. If the
name has no parts, return an empty char (`''`).

Examples:

- `name_initials('ada  lovelace')` returns `'AL'`.
- `name_initials('jean-luc picard')` returns `'JLP'`.
- `name_initials('   ')` returns `''`.
