# MATLAB Tasks

An open benchmark of **157 original MATLAB programming tasks**. Each task has a
precise problem statement, a reference solution and an automated test. The
mylab project wrote it and publishes it under the MIT license. Nothing is
copied from MATLAB Cody, documentation, textbooks or puzzle sites.

Three audiences:

- **AI and code-model researchers:** a ready-made evaluation set for MATLAB
  code generation (pass@k). Very little exists for MATLAB compared with
  Python.
- **Teachers:** graded practice problems with auto-checking tests, from
  first loops to numerical methods, OOP and data tables.
- **Implementers:** a conformance suite for MATLAB-language engines. mylab
  runs it on every commit, and GNU Octave is checked weekly.

## Contents

| category | tasks | easy | medium | hard | passes on Octave |
|---|---|---|---|---|---|
| algorithms | 10 | 2 | 5 | 3 | 10/10 |
| arrays_indexing | 9 | 4 | 3 | 2 | 9/9 |
| basics | 9 | 5 | 3 | 1 | 9/9 |
| cells_structs | 9 | 3 | 4 | 2 | 9/9 |
| classes_oop | 8 | 2 | 4 | 2 | 7/8 |
| control_flow | 8 | 4 | 3 | 1 | 8/8 |
| data_tables | 8 | 3 | 4 | 1 | 0/8 (no `table` in Octave) |
| file_io | 8 | 2 | 4 | 2 | 7/8 |
| functions | 9 | 3 | 5 | 1 | 9/9 |
| linear_algebra | 9 | 2 | 5 | 2 | 9/9 |
| maps_sets | 10 | 3 | 5 | 2 | 10/10 |
| numerical_methods | 10 | 4 | 4 | 2 | 10/10 |
| optimization | 8 | 2 | 4 | 2 | 8/8 |
| polynomials_interpolation | 8 | 3 | 3 | 2 | 8/8 |
| signal_processing | 8 | 3 | 3 | 2 | 8/8 |
| simulation | 8 | 2 | 3 | 3 | 8/8 |
| statistics | 9 | 2 | 6 | 1 | 9/9 |
| strings_text | 9 | 3 | 5 | 1 | 8/9 |
| **total** | **157** | **52** | **73** | **32** | **146/157** |

Reference solutions pass on **mylab: 157/157** and **GNU Octave 8.4:
146/157**. Octave's failures are all missing features: `table`, abstract
methods, `readmatrix` and string arrays. Each task's `task.json` records
these results under `verified`.

## How the answers were checked

- Expected values in the tests were computed independently: with
  NumPy/SciPy, exact fractions, closed forms, or by hand. They were not taken
  from the reference solution.
- Every reference solution runs on two independent implementations (mylab
  and Octave) wherever Octave supports the feature.
- Tests cover edge cases (empty inputs, ties, single elements, orientation,
  output class). Deliberately wrong solutions were run to confirm the tests
  reject them.
- Tests never compare printed text and never depend on random number
  streams, so results are the same on every MATLAB-language implementation.

## Task format

```
tasks/<category>/<name>/
  task.json      id, title, category, difficulty, function, signature, tags, verified
  prompt.md      the problem statement
  solution/      reference solution (<function>.m, or class files)
  test.m         assertions; the task passes if this script runs without error
  data/          input files, if the task reads any
```

See [AUTHORING.md](AUTHORING.md) to contribute new tasks.

## Running it

From the mylab repository root, after `pip install -e .`:

```bash
cd datasets/matlab-tasks
python tools/run_tasks.py                        # reference solutions on mylab
python tools/run_tasks.py --engine octave        # ...on GNU Octave
python tools/run_tasks.py --solutions answers/   # grade your own (or a model's) solutions
```

To grade candidate solutions, put the function file for task
`basics/sum_of_multiples` at `answers/basics/sum_of_multiples/sum_of_multiples.m`.
Helper files may sit next to it.

### Evaluating a code model

1. Export the prompts:

   ```bash
   python tools/export_jsonl.py
   ```

   This writes `dist/matlab_tasks.jsonl` (prompts and signatures, no
   solutions).
2. Ask the model for each prompt, and save its code as
   `answers/<id>/<function>.m`.
3. Grade:

   ```bash
   python tools/run_tasks.py --solutions answers/ --json results.json
   ```

   The summary shows pass rates by category. For pass@k, run k samples into
   separate folders.

To check answers against real MATLAB, use the same `test.m` files: put the
answer file and `test.m` in one folder and run `test`.

`python tools/export_jsonl.py --full` also includes solutions and tests, for
uploading to a dataset hub.

## License and use

MIT. You may use the tasks for training, evaluation and teaching. If you
publish results, please cite the benchmark and the version (git commit) you
used. Avoid training a model on the test files you then evaluate it with.

## Limitations

- 157 tasks is a modest size. It's enough to compare engines and models
  coarsely, but not to rank closely matched systems.
- Only mylab and Octave have checked the reference solutions. They're written
  for real MATLAB semantics but haven't been run on MATLAB itself. Please
  report any task that behaves differently there.
- Coverage leans towards coursework topics. Toolbox-heavy domains (image
  processing, control, deep learning) and graphics output are not covered yet.
