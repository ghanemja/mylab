# mylab

**A free, open-source environment for running MATLAB-language code.** It's for
students, researchers and engineers who can't afford a MATLAB license.

- **Runs in your browser with nothing to install.** A full editor, command
  window, workspace and plots, running locally through WebAssembly.
- **Runs on your computer** as a command-line tool, a Python library, a Jupyter
  kernel, or a tool your AI assistant can use (MCP).
- **Has an AI helper built in.** It explains errors, fixes code and translates
  to Python, using free local models (Ollama) or your own API key. No paid
  account is needed.
- **Covers a lot of the language:** matrices, N-D arrays, cells, structs,
  strings, tables, `containers.Map`, `classdef` classes and about 750
  functions, from linear algebra to `fft`, `ode45`, `fminsearch`, `kmeans` and
  plotting.

![The mylab browser app](docs/images/web-app.png)

```matlab
>> A = magic(4);
>> x = A(2:3, :) \ [1; 2];
>> t = linspace(0, 2*pi); plot(t, sin(t)), grid on
>> T = readtable('grades.csv'); T(T.Score > 90, :)
```

mylab is built on [NumPy](https://numpy.org), [SciPy](https://scipy.org) and
[Matplotlib](https://matplotlib.org). It is an independent project with its own
implementation. See [Legal](#license-and-trademarks).

---

## Ways to use it

### 1. In the browser (no install)

Open the web app, which is hosted with GitHub Pages from this repository. Your
files stay in your browser and the code runs on your own machine. The first
visit downloads about 30–60 MB, which is cached afterwards.
See [docs/WEB.md](docs/WEB.md).

### 2. On your computer

You need Python 3.10 or newer.

```bash
pip install "git+https://github.com/ghanemja/mylab"
```

```bash
mylab                      # interactive prompt
mylab myscript.m           # run a script
mylab -e "disp(sum(1:100))"
```

New to all this? Start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

### 3. In Jupyter notebooks

```bash
pip install "mylab-lang[jupyter] @ git+https://github.com/ghanemja/mylab"
mylab-kernel install --user
```

See [docs/JUPYTER.md](docs/JUPYTER.md).

### 4. From AI assistants (Claude Desktop, Cursor, …)

```bash
pip install "mylab-lang[mcp] @ git+https://github.com/ghanemja/mylab"
```

Add `{"mcpServers": {"mylab": {"command": "mylab-mcp"}}}` to your client's
configuration. See [docs/MCP.md](docs/MCP.md).

### 5. From Python

```python
from mylab.api import Session

s = Session()
r = s.run("y = cumsum(1:5)")
print(r.output)        # y = ...
s.get("y")             # numpy array
r.figures              # PNG bytes of any plots
```

---

## Open datasets

mylab also publishes two datasets for the community (see [datasets/](datasets/)):

- **[MATLAB Tasks](datasets/matlab-tasks/):** 157 original programming tasks
  with reference solutions and tests. Use them to benchmark AI code models on
  MATLAB, for graded teaching exercises, or to check MATLAB-language engines.
- **[MATLAB Corpus](datasets/matlab-corpus/):** 29,114 real-world `.m` files
  (4.1M lines) from 200 permissively licensed repositories, pinned and
  annotated.

## What's supported

| Area | Supported |
|---|---|
| Language | All operators and precedence, ranges, 1-based column-major indexing, `end`, logical indexing, arrays that grow on assignment, deleting with `x(i) = []`, `if`/`for`/`while`/`switch`/`try`, `break`/`continue`/`return`, command syntax (`hold on`) |
| Functions | Script-local functions, `.m` files on the path, multiple outputs, `nargin`/`nargout`, `varargin`/`varargout`, anonymous functions with captured variables, `global`, `eval`/`feval` |
| Data types | double, single, logical, char, int8–uint64 (saturating), complex numbers, N-D arrays, cell arrays, structs and struct arrays, string arrays, function handles, `containers.Map`, `table` |
| Classes | `classdef` value and handle classes, properties with defaults, `Constant`/`Dependent` properties, `get.`/`set.` methods, constructors, `Static` methods, inheritance with `obj@Parent`, operator overloading, custom `disp` |
| Library | About 750 functions: math, linear algebra, statistics and distributions, signal processing (`fft`, `butter`, `filtfilt`, `findpeaks`, …), optimization (`fzero`, `fminsearch`, `fmincon`, `linprog`, …), ODEs (`ode45`, `ode15s`, …), interpolation, polynomials, strings and regex, file input/output, `readtable`/`writetable`, plotting (`plot`, `subplot`, `scatter`, `bar`, `histogram`, `surf`, `contour`, …) |

The full list is in [docs/FUNCTIONS.md](docs/FUNCTIONS.md). Type `help name` in
mylab for any function. Known differences and gaps (for example Simulink,
events, enumerations and object arrays) are in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Speed

- **Vectorized code** runs at NumPy speed (compiled BLAS/LAPACK underneath).
- **Simple numeric loops are compiled.** Loops that do scalar arithmetic and
  read or write array elements are translated to fast code automatically,
  about 20–50× faster than interpreting them: 200,000 iterations of
  `s = s + k^2` take about 0.03 s. If anything unusual happens (an index
  out of range, a complex result), mylab falls back to the interpreter for
  that loop, so results and error messages stay the same.
- **Other loops are interpreted** (ones that call your own functions, use
  cells, structs or strings, or print): about 2–10 µs per statement. That's
  fine for coursework, but slower than MATLAB.

`benchmarks/bench.py` measures common patterns.

## Project layout

```
mylab/            the engine (lexer, parser, interpreter, values, indexing, display)
  builtins/       the function library, one module per area
  classdef.py     user-defined classes;   objects.py  containers.Map / object protocol
  api.py          Session API used by every front end
  cli.py          `mylab` command          kernel.py   Jupyter kernel
  mcp_server.py   MCP server               help_data.py  help text for every function
web/              the browser app (Pyodide + CodeMirror, no build step)
examples/         runnable example scripts
tests/            pytest suite (engine, functions, objects, tables, MCP, kernel)
docs/             guides, function list, compatibility notes
```

## Contributing

Contributions are very welcome: new functions, bug reports with a small `.m`
snippet, documentation, and translations of the web app. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and **[CLEANROOM.md](CLEANROOM.md)** first.
The clean-room rules keep the project legally safe for everyone.

## License and trademarks

mylab is released under the [MIT License](LICENSE).

mylab is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc.
MATLAB® is a registered trademark of The MathWorks, Inc. The name is used here
only to describe compatibility. mylab contains no MathWorks code or
documentation.
