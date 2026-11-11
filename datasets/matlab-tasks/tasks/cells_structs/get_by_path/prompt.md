Write a MATLAB function

    v = get_by_path(data, path)

that reads a value from nested structs and cell arrays following a path
written in MATLAB-like syntax, without using `eval`, `getfield` with dynamic
code, or similar. The path (a char vector) is a sequence of steps:

- `name` — a field of a struct. A field name starts with a letter followed by
  letters, digits or underscores. The first step may be written without a dot
  (`users`); every later field step is written with a leading dot (`.name`).
- `(k)` — element `k` of an array (struct array, numeric array, cell array,
  ...), using linear indexing; for a cell array this gives a 1-by-1 cell.
- `{k}` — the content of element `k` of a cell array.

Indices `k` are written as positive integers with digits only. The empty path
returns `data` itself.

Errors:

- If the path is not well-formed (e.g. `'users(2'`, `'users(0)'`, `'a..b'`,
  `'.a'`, `'a(x)'`, spaces), throw `get_by_path:badPath`. The whole path is
  checked before any data is read.
- If the path is well-formed but cannot be followed (missing field, field
  access on something that is not a scalar struct, index larger than the
  number of elements, `{k}` on something that is not a cell array), throw
  `get_by_path:notFound`.

Example:

    data.users = struct('name', {'Ada', 'Lin'}, 'tags', {{'admin', 'dev'}, {'ops'}});
    data.meta.version = 3;
    get_by_path(data, 'users(2).name')     % 'Lin'
    get_by_path(data, 'users(1).tags{2}')  % 'dev'
    get_by_path(data, 'meta.version')      % 3
