A sign-up sheet stores participant names as a MATLAB **string array**, one
name per element, written as given name, optional middle names and family
name (e.g. `"Mara Ilse Vogt"`). Write a MATLAB function

    out = names_last_first(names)

that returns a string array of the same size with each name rewritten as
`"Family, Given M."`: the last word, a comma and a space, the first word, and
then the initial of each middle word followed by a period, separated by
spaces.

- Words are separated by one or more spaces; ignore leading/trailing spaces.
- A single-word name is returned as that word.
- An empty or all-space name becomes `""` (empty string).
- Keep the letter case of the words as given; initials are the first
  character of a middle word.

Example:

    names_last_first(["Mara Ilse Vogt", "  Tomas   Reyes ", "Okonkwo"])
    % returns ["Vogt, Mara I.", "Reyes, Tomas", "Okonkwo"]

(This task uses string arrays, which GNU Octave does not support.)
