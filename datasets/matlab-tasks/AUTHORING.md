# Writing tasks for the MATLAB Tasks benchmark

Each task is a small, self-contained programming problem with a reference
solution and tests. The goal is a benchmark that is **correct, original and
portable**. Any MATLAB-language implementation (MATLAB, GNU Octave, mylab)
and any code-generating model can be evaluated on it.

## Layout

```
tasks/<category>/<task_name>/
  task.json        metadata (below)
  prompt.md        the problem statement a person or model sees
  solution/        reference solution: <function>.m, plus helper files if needed
  test.m           a script that calls the function and asserts results
  data/            optional input files the test reads (CSV, text)
```

`task.json`:

```json
{
  "id": "<category>/<task_name>",
  "title": "Short human title",
  "category": "<category>",
  "difficulty": "easy | medium | hard",
  "function": "function_name",
  "signature": "[out1, out2] = function_name(in1, in2)",
  "tags": ["indexing", "cellfun"],
  "license": "MIT"
}
```

`verified` is added by `tools/run_tasks.py --verify --write-status`. Don't
edit it by hand.

## Rules

1. **Original work.** Write every prompt, solution and test yourself. Don't
   copy or closely paraphrase problems from MATLAB Cody, MathWorks
   documentation, textbooks, LeetCode, Project Euler or other sites. Classic
   algorithmic ideas (bisection, run-length encoding, ...) are fine when the
   statement and code are your own. The benchmark is MIT licensed.
2. **The prompt is complete.** It gives the exact function signature, the
   inputs and their types, the expected outputs (including orientation: row
   or column), edge-case behaviour, and 1–3 examples. A careful programmer
   must be able to solve it from the prompt alone.
3. **Tests are portable MATLAB.** Use only `assert(condition)` or
   `assert(condition, message)`, together with `isequal`, `abs`, `all`,
   `size`, `class`, `ischar`, `isempty`, etc. Compare floating-point results
   with a tolerance, e.g. `assert(abs(x - 1.2345) < 1e-9)`. Don't use
   Octave-only syntax (`!=`, `++`, `#` comments, `endif`, `printf`) or
   features that exist only in mylab.
4. **Tests are strong.** Use 4–10 checks per task. Cover the examples, edge
   cases (empty input, scalars, negative numbers, ties, a single element),
   output size and orientation, and output class where it matters. A trivial
   or hard-coded answer must fail.
5. **Deterministic.** Never assert exact values that depend on random numbers.
   Random number generators differ between implementations, even with the
   same seed. If a task uses randomness, test properties instead (ranges,
   sizes, sums).
6. **No display checks.** Tests never compare printed text, because display
   formats differ between implementations. Return values and check them
   instead.
7. **Self-contained and fast.** No network, no toolboxes beyond core MATLAB
   (plus signal/statistics functions where the category says so), and under
   a few seconds per task. Files the test reads go in `data/`. Files the test
   writes go in the current folder and should be deleted at the end.
8. **Difficulty:**
   - easy: a few lines;
   - medium: needs a real algorithm or careful handling of several cases;
   - hard: a substantial algorithm, numerical method or data-structure design.

## Verifying

```bash
python tools/run_tasks.py --verify --filter <category>/<task_name> --show-failures
```

- The reference solution must pass on **mylab**, and on **Octave** where
  Octave supports the feature. Octave has no `table` type and treats `"..."`
  as char, so tasks that depend on those may record `"octave": false`.
- If a solution passes on Octave but fails on mylab, and the code is valid
  MATLAB, keep the task. That's a mylab bug: report it.
- If it fails on both, fix the task.
