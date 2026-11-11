Write a MATLAB function `inv = invert_map(m)`.

`m` is a `containers.Map` with `KeyType` `'char'` whose values are character
vectors — for example, a map from a person's name to their team. Several keys
may share the same value.

Return a new `containers.Map` `inv` with `KeyType` `'char'` and `ValueType`
`'any'`. Its keys are the distinct values of `m`, and `inv(v)` is a 1-by-k
cell array of all keys of `m` that map to `v`, sorted alphabetically (as `sort`
orders character vectors). `m` itself must not be modified. An empty `m`
gives an empty `inv`.

Example:

```matlab
m = containers.Map({'zoe', 'adam', 'mia'}, {'red', 'blue', 'red'});
inv = invert_map(m);
inv('red')    % {'mia', 'zoe'}
inv('blue')   % {'adam'}
```
