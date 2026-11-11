Write a MATLAB function

    tf = phrase_palindrome(s)

that returns logical `true` if the char vector `s` reads the same forwards and
backwards after removing everything except the letters `a-z`, `A-Z` and digits
`0-9` and ignoring case, and `false` otherwise. A phrase with no letters or
digits left (including the empty char) counts as a palindrome.

Examples:

- `phrase_palindrome('No lemon, no melon')` returns `true`.
- `phrase_palindrome('ab-c')` returns `false`.
