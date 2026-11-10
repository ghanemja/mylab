# Changelog

## 1.0.0

First public release.

### Engine
- MATLAB-language interpreter: matrices, N-D arrays, 1-based column-major
  indexing with growth and deletion, `end`, logical indexing, all operators,
  control flow, `try`/`catch`, command syntax.
- Functions: script-local functions, `.m` files on the path, multiple outputs,
  `varargin`/`varargout`, anonymous functions with closures, `global`.
- Data types: double, single, logical, char, int8–uint64, complex numbers,
  cell arrays, structs and struct arrays, string arrays, function handles.
- `classdef` classes (value and handle, properties, methods, constructors,
  inheritance, static methods, Constant/Dependent properties, get/set
  methods, operator overloading, custom display).
- `containers.Map` and `table` (with `readtable`/`writetable`).
- Command-window display and error messages with line numbers.

### Library
- About 750 built-in functions: math, linear algebra, statistics and
  probability distributions, signal processing, optimization, ODE solvers,
  interpolation, polynomials, strings and regular expressions, cells and
  structs, file input/output, and plotting through Matplotlib.
- Help text for every function (`help name`).

### Also in 1.0 (pre-release QA)
- Graphics handles for figures, axes, lines and other plot objects, with
  dot syntax and `get`/`set`. New plotting functions: `tiledlayout`,
  `nexttile`, `fplot`, `quiver`, `polarplot`, `yyaxis`, `peaks`, `shading`,
  colormaps and more.
- Tables: row names, joins, `groupsummary`, `findgroups`/`splitapply`,
  grouped `varfun`, brace assignment, `rmmissing`/`fillmissing`.
- File I/O: `textscan`, `fscanf`, `dir`, `readcell`/`writecell`,
  `readlines`/`writelines`, `importdata`, delimiter-aware `writematrix`/
  `readmatrix` that skips header lines.
- Numerics: `accumarray` with functions, `pchip`/`makima`/`ppval`, ODE events
  and accurate dense output, `polyfit` with `S` and `mu`, `gradient`,
  `conv2`, `maxk`/`mink` and more.
- Language: nested functions, per-function `persistent`, `MException`
  objects with standard identifiers, `warning('off', id)`, classdef access
  control, abstract classes and property validation, command syntax for any
  function, recursion up to 500 calls.

### Ways to run it
- `mylab` command-line prompt and script runner.
- Python API: `mylab.api.Session`.
- Browser app with no install (Pyodide), including an AI assistant panel
  (Ollama, Anthropic, OpenAI-compatible).
- MCP server for AI assistants: `mylab-mcp`.
- Jupyter kernel: `mylab-kernel install`.
