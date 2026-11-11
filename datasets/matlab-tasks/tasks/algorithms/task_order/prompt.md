Write a MATLAB function `order = task_order(n, deps)`.

There are `n` tasks numbered `1..n`. `deps` is an m-by-2 matrix of
precedence rules: a row `[a b]` means task `a` must be done before task `b`.
`deps` may be empty (`zeros(0,2)`) and may contain repeated rows.

Return a 1-by-n row vector `order` listing every task exactly once so that all
rules are satisfied. When several orders are possible, return the one that is
smallest in dictionary order, i.e. at every step pick the lowest-numbered task
whose prerequisites are all already done.

If the rules are contradictory (they contain a cycle, including a rule
`[a a]`), return an empty `zeros(1,0)`.

Examples:

- `task_order(4, [3 1; 2 1])` returns `[2 3 1 4]`.
- `task_order(3, [1 2; 2 3; 3 1])` returns `zeros(1,0)`.
