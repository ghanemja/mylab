# mylab

**A free, open-source environment for running MATLAB-language code.**
It's for students, researchers and engineers who can't afford a MATLAB license.

mylab reads `.m` scripts and functions and runs them on top of
[NumPy](https://numpy.org), [SciPy](https://scipy.org) and
[Matplotlib](https://matplotlib.org). It is written in Python, so it is easy to
install, easy to contribute to, and easy to connect to other tools such as AI
assistants and Jupyter.

```matlab
>> A = magic(4);
>> x = A(2:3, :) \ [1; 2]
>> plot(linspace(0, 2*pi), sin(linspace(0, 2*pi))), grid on
```

> **Status: early (v0.1).** Core language features and 450+ built-in functions
> work. Many programs used in courses and tutorials run unchanged. Larger
> toolbox features do not exist yet (see [Roadmap](docs/ROADMAP.md)).

---

## Install

You need Python 3.10 or newer.

```bash
git clone <this repository>
cd mylab
pip install -e .
```

## Use it

Start the interactive prompt:

```bash
mylab
```

Run a script:

```bash
mylab examples/hello.m
```

Run a one-liner:

```bash
mylab -e "disp(sum(1:100))"
```

Use `--no-plots` on servers. Plots are drawn off-screen and you can still
save them with `saveas` or `print`.

### From Python

```python
from mylab import Interpreter

m = Interpreter()
m.execute("""
    t = linspace(0, 1, 5);
    y = t.^2;
""")
y = m.get_var("y")          # -> numpy array
m.set_var("k", 3)
m.eval_expression("k * 2")  # -> array([[6.]])
```

This Python API is what future integrations will build on: the web app, the
Jupyter kernel and the AI-assistant (MCP) server.

## What works

- **Language:** matrices and `[...]` building, ranges, 1-based column-major
  indexing, `end` inside an index, logical indexing, arrays that grow on
  assignment, deleting with `x(i) = []`, all operators and their precedence,
  complex numbers, integer types with saturation, `if/elseif/else`, `for`,
  `while`, `switch`, `try/catch`, `break/continue/return`.
- **Functions:** local functions in scripts, `.m` function files on the path,
  multiple return values, `nargin`/`nargout`, `varargin`/`varargout`,
  anonymous functions with captured variables, `@name` handles, `global`
  variables.
- **Data types:** double, single, logical, char, int8–uint64, `"string"`
  scalars, cell arrays, structs and struct arrays, function handles.
- **Output:** MATLAB-style display of results (`x = ...`), `format long/short`,
  `disp`, `fprintf` and `sprintf` with MATLAB's format-reuse rules.
- **Library:** 450+ functions covering math, statistics, linear algebra,
  strings and regular expressions, cells and structs, `cellfun`/`arrayfun`,
  polynomials, interpolation, `fzero`, `fminsearch`, `integral`,
  `ode45`/`ode23`/`ode15s`, file input/output and plotting (`plot`, `subplot`,
  `scatter`, `bar`, `histogram`, `surf`, `contour`, …). See
  [docs/FUNCTIONS.md](docs/FUNCTIONS.md) for the full list.
- **Error messages** are in the same style as MATLAB's and include the line
  number.

Known differences and missing features are listed in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Project layout

```
mylab/
  lexer.py        tokenizer (handles MATLAB's whitespace and quote rules)
  parser.py       recursive-descent parser -> syntax tree (ast.py)
  interpreter.py  evaluator: workspaces, calls, assignment, control flow
  indexing.py     1-based, column-major, auto-growing indexing
  ops.py          operators and concatenation
  values.py       runtime types (cell, struct, string, function handle)
  display.py      command-window formatting
  builtins/       the function library, one module per area
  cli.py          `mylab` command: prompt and script runner
examples/         runnable .m scripts (each one is also run as a test)
tests/            pytest suite
docs/             function list, compatibility notes, roadmap
```

## Contributing

Contributions are very welcome, especially new built-in functions, and
bug reports with a small `.m` snippet that behaves differently from MATLAB.

**Please read [CLEANROOM.md](CLEANROOM.md) before contributing.** It explains
how this project stays legally clean. In short: implement behaviour from your
own understanding and tests, and never copy MathWorks documentation text,
examples or source code.

```bash
pip install -e ".[dev]"
pytest            # run the tests
ruff check .      # lint
```

## License and trademarks

mylab is released under the [MIT License](LICENSE).

mylab is an independent project. It is not affiliated with, endorsed by or
sponsored by The MathWorks, Inc. MATLAB® is a registered trademark of The
MathWorks, Inc. The name is used here only to describe compatibility.
