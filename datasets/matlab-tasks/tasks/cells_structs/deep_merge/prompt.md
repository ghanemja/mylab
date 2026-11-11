Configuration files are often layered: defaults first, then user overrides.
Write a MATLAB function

    c = deep_merge(a, b)

that merges two **scalar** structs. The result `c`:

- contains every field of `a`, in `a`'s order, followed by the fields of `b`
  that `a` does not have, in `b`'s order;
- for a field present in only one input, takes that value;
- for a field present in both: if **both** values are scalar structs, the
  value is `deep_merge` of the two (recursively, to any depth); otherwise the
  value from `b` wins (this includes struct arrays, and a struct replacing a
  non-struct or the other way round).

Example:

    a = struct('name', 'app', 'opts', struct('depth', 2, 'color', 'red'));
    b = struct('opts', struct('color', 'blue', 'width', 5), 'version', 3);
    c = deep_merge(a, b);
    % c.name = 'app', c.version = 3
    % c.opts has fields depth = 2, color = 'blue', width = 5 (in that order)
