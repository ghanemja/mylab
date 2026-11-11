A puzzle club encodes messages by shifting each letter forward in the alphabet
by its position among the letters of the message: the first letter by 1, the
second by 2, and so on. Non-letters are copied unchanged and are not counted.
Shifts wrap around (`'z'` shifted by 1 is `'a'`) and case is preserved. Only
the ASCII letters `a-z` and `A-Z` count as letters.

Write a MATLAB function

    out = position_shift_cipher(s, mode)

where `s` is a char vector and `mode` is `'encode'` or `'decode'` (decoding
shifts backwards instead). `mode` is optional and defaults to `'encode'`. Any
other mode throws an error with identifier `position_shift_cipher:badMode`.
`out` is a char vector of the same size as `s`.

Examples:

- `position_shift_cipher('abc')` returns `'bdf'`.
- `position_shift_cipher('Hi, there')` returns `'Ik, wljxl'`.
- `position_shift_cipher('bdf', 'decode')` returns `'abc'`.
