Write a MATLAB function `out = all_dependencies(deps, name)`.

`deps` is a `containers.Map` with `KeyType` `'char'`: each key is a package
name and its value is a cell array of character vectors listing the packages
it depends on **directly** (possibly empty, possibly with repeats). A package
that never appears as a key has no dependencies. The dependency graph may
contain cycles.

Return a 1-by-k cell array `out` of every package that `name` depends on
directly or indirectly, each listed once and sorted alphabetically (as `sort`
orders character vectors). `name` itself is never included, even when a cycle
leads back to it. If there are no dependencies (including when `name` is not
a key of `deps`), return `cell(1,0)`.

Example:

```matlab
deps = containers.Map({'app', 'web', 'db', 'log'}, ...
                      {{'web', 'db'}, {'log', 'http'}, {'log'}, {}});
all_dependencies(deps, 'app')   % {'db', 'http', 'log', 'web'}
all_dependencies(deps, 'db')    % {'log'}
```
