Write a MATLAB **handle** class `ValueStack` (file `ValueStack.m`) — a
last-in-first-out stack that can hold values of any type (numbers, character
vectors, structs, other arrays, ...).

API:

- `s = ValueStack()` creates an empty stack.
- `push(s, x)` puts `x` on top of the stack. `push(s, x1, x2, ...)` with more
  values pushes them in order, so the last argument ends up on top.
- `x = pop(s)` removes and returns the top value.
- `x = peek(s)` returns the top value without removing it.
- `pop` and `peek` on an empty stack throw an error with identifier
  `'ValueStack:empty'`.
- Dependent property `Size` — the number of stored values (a double).
- `tf = isEmpty(s)` — logical true when the stack holds nothing.
- `c = toCell(s)` — a 1-by-N cell array of the stored values from bottom to
  top (`cell(1,0)` when empty).

Values are stored exactly as given; for example pushing `[]` or `{}` stores
that empty value as one element.

Example:

```matlab
s = ValueStack();
push(s, 1, 'two', [3 3 3]);
pop(s)     % [3 3 3]
s.Size     % 2
```
