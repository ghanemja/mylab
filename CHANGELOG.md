# Changelog

## Unreleased

### Language
- Object arrays of classdef objects: `objs(3) = Point(1, 2)`, `[a, b]`,
  `[objs.x]`, `Point.empty`, loops, `arrayfun`, `numel`, deletion.
- `enumeration` blocks (with or without a constructor, or derived from a
  number type), `switch` on members, `enumeration('Name')`.
- `events` blocks with `notify`, `addlistener`, `listener` and property
  `PreSet`/`PostSet` listeners; `event.EventData` subclasses.
- Methods in separate files of an `@ClassName` folder; functions and classes
  in `+package` folders (`pkg.func(...)`, `pkg.Class(...)`, `@pkg.func`).
- `onCleanup`, `inputParser`, `RandStream`, `timer`, `message`, System-object
  `step`, `s = rng` / `rng(s)`, `rand('seed', n)`.

### Library
- Geometry: `pdist2`, `knnsearch`, `dsearchn`, `inpolygon`, `polyarea`,
  `rectint`, `convhull`, `convhulln`, `delaunay`, `delaunayn`, `griddata`,
  `scatteredInterpolant`, `sphere`, `cylinder`, `ellipsoid`.
- N-D and statistics: `fftn`, `ifftn`, `convn`, `rcond`, `interpn`,
  `interp3`, `datasample`, `mvnrnd`, `mvnpdf`, `pca`, `grp2idx`, `ksdensity`,
  `random`, more samplers (`gamrnd`, `betarnd`, …), `spectrogram`, `ldl`.
- `categorical` arrays, usable as table variables and grouping variables.
- Files: `jsondecode`/`jsonencode`, `audioread`/`audiowrite`/`audioinfo`
  (WAV), `xlsread`/`xlswrite` and Excel support in `readtable`,
  `writetable`, `readmatrix`, `writematrix`, `readcell`, `writecell`.
- Plots: `quiver3`, `surface`, `pcolor`, `surfl`, `trisurf`, `trimesh`,
  `triplot`, `stem3`, `xline`, `yline`, `alpha`, `boxplot`, `gobjects`.

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
- About 1,250 built-in functions: math, linear algebra, statistics and
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

### Added before release (launch work)
- **Speed:** a loop compiler runs simple numeric loops 15–55× faster, with an
  exact fallback to the interpreter.
- **Images:** 77 functions (`imread`, `imwrite`, `imresize`, `imfilter`,
  `edge`, `imbinarize`, `bwlabel`, `regionprops`, …) and a proper `imshow`.
- **Sparse matrices** stored sparsely (`sparse`, `speye`, `spdiags`, `\`,
  `eigs`, `pcg`, `gmres`, …).
- **Dates and times:** `datetime`, `duration`, calendar durations, `datenum`,
  `datestr`, `datevec`.
- **Files and system:** `fread`/`fwrite`/`fseek`, `genpath`, `mfilename`,
  `which`, `system`, `copyfile`, zip tools, `webread`, `evalin`/`assignin`.
- **Animations and dialogs:** `getframe`, `VideoWriter`, `animatedline`;
  `drawnow` loops come back as animated GIFs; `waitbar`, `inputdlg`,
  `questdlg`, `uigetfile` as text prompts; `axes`, `rectangle`, `findobj`.
- **Language:** `arguments` blocks, `f(Name=Value)`, command syntax with
  paths, Octave-style block ends, `import`.
- **Debugger:** `mylab --debug`, `keyboard`, `dbstop`; a tracer for
  step-through views.
- **Autograder:** `mylab-grade` (check, run, batch, export, Gradescope zip).
- **Browser app:** share links, projects with folders and data files, open a
  folder or a GitHub repository, `%%` sections, published reports,
  step-through panel, assignments with "Check my answers".
- **VS Code extension** (`editors/vscode`), conda recipe, release workflow.
- **Datasets:** MATLAB Tasks (157 tasks) and MATLAB Corpus (29,114 files).
- **Conformance suite:** 257 scripts with 5,377 checks compared against GNU
  Octave; all pass. About 40 behaviour differences found this way were fixed
  (integer and single classes, exact 64-bit integers, set functions,
  `num2str`, `regexp`, reductions on N-D arrays, hex literals and more).
- Appending to cell and struct arrays in a loop is no longer quadratic.
- The loop compiler inlines anonymous functions and one-line local
  functions: loops that call them run about 100× faster.
- **Optional accelerator** (`pip install "mylab-lang[fast]"`): large
  elementwise array expressions are evaluated in one multi-threaded pass,
  2–5× faster than NumPy on four cores.
- The PyPI package is named `mylab-lang` (the name `mylab` was taken).

### Ways to run it
- `mylab` command-line prompt and script runner.
- Python API: `mylab.api.Session`.
- Browser app with no install (Pyodide), including an AI assistant panel
  (Ollama, Anthropic, OpenAI-compatible).
- MCP server for AI assistants: `mylab-mcp`.
- Jupyter kernel: `mylab-kernel install`.
