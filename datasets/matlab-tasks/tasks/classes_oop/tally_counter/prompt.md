Write a MATLAB **handle** class `Tally` (file `Tally.m`) that models a
hand-held tally counter.

API:

- `t = Tally()` creates a counter at 0; `t = Tally(start)` starts at the
  number `start`.
- Property `Count` — the current value. It can be read from outside but must
  **not** be settable from outside the class (`SetAccess = private`);
  assigning `t.Count = 5` outside the class must raise an error.
- `increment(t)` adds 1; `increment(t, k)` adds `k` (which may be negative).
- `reset(t)` sets the count back to 0 (not to the start value).
- `n = total(t)` returns the sum of the absolute values of every amount ever
  passed to `increment` on this counter (with no argument counting as 1),
  regardless of resets. It starts at 0 for a new counter.

Because `Tally` is a handle class, copies of a variable refer to the same
counter.

Example:

```matlab
t = Tally(10);
increment(t);
u = t;
u.increment(-4);
t.Count      % 7
total(t)     % 5
```
