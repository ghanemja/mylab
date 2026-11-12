# Compatibility notes

mylab aims to run everyday MATLAB-language code unchanged. This page lists
what isn't supported yet and where behaviour knowingly differs. If you find a
difference that isn't listed, please open an issue with a short `.m` snippet,
what mylab printed, and what you expected.

## Supported (summary)

- N-dimensional arrays, cell arrays, structs and struct arrays, string arrays,
  function handles, integer and single types, complex numbers.
- `classdef` classes:
  - value and handle classes, properties with defaults;
  - Constant and Dependent properties, `get.`/`set.` methods;
  - property validation (`(1,1) double {mustBePositive}`);
  - access control (`Access`, `GetAccess`, `SetAccess`: private, protected,
    immutable), abstract classes;
  - constructors, Static methods, inheritance (`obj@Parent`, `method@Parent`);
  - operator overloading, custom `disp`/`display`, `delete`/`isvalid`.
- Nested functions sharing their parent's variables, `arguments` blocks,
  `f(Name=Value)` syntax, and Octave-style `endfunction`/`endif` block ends.
- `MException` objects with MATLAB-style error identifiers;
  `warning('off', id)`.
- `containers.Map` and `table` (with `readtable`/`writetable` for delimited
  text files).
- About 750 built-in functions; see [FUNCTIONS.md](FUNCTIONS.md).

## Not supported yet

- **Object arrays:** you can't index objects into arrays (`objs(3) = Point(...)`).
  Use a cell array instead: `objs{3} = Point(...)`.
- **classdef features:** events and listeners, enumerations,
  `subsref`/`subsasgn` overloading, and methods defined in `@ClassName` folders.
- **Other data types:** datetime, duration, categorical, timetable,
  `containers.Map` with non-scalar keys, sparse matrices (`sparse(...)` builds
  an ordinary full matrix).
- **Tables:** `rowfun`, `stack`/`unstack`, and timetables. Table variables
  must be columns. (Row names, joins, `groupsummary`, `findgroups`/
  `splitapply`, `varfun` with groups, `rmmissing`/`fillmissing` are
  supported.)
- **Graphics handles:** `figure`, `gcf`, `gca`, `plot`, `bar`, `histogram`,
  `title`, `legend`, `colorbar` and the other plotting functions return handle
  objects. You can read and change their common properties with dot syntax
  (`h.LineWidth = 2`) or `get`/`set`, and figure numbers still work wherever a
  figure is expected. Limits:
  - only a subset of each object's properties exists. Unknown properties raise
    an error, and a few harmless ones (e.g. `TickLabelInterpreter`) are
    accepted but have no effect;
  - several handles come back as a handle array, which can be indexed,
    extended by one element, or concatenated. `gobjects` and arbitrary object
    arrays aren't supported;
  - there are no animations, GUIs (`uifigure`, App Designer), callbacks or
    interactive tools.
- **Input and output:** binary I/O (`fread`/`fwrite`), Excel files, images
  (`imread`), audio. `textscan` and `fscanf` handle common formats only (no
  fixed-width fields).
- **Toolboxes:** only the parts listed in FUNCTIONS.md (signal processing,
  statistics, optimization). Simulink, Symbolic Math, Control System, Image
  Processing and others are out of scope for now.
- **Language corners:**
  - `nargin('fname')`, `inputname`, `evalin`/`assignin`, `parfor` (runs as an
    ordinary `for` loop), `Repeating` and `Output` argument blocks (ordinary
    `arguments` blocks with sizes, classes, validators, defaults and
    name-value options are supported);
  - command syntax (`hold on`, `shading interp`, `clearvars -except x`) works
    when the arguments are plain words. In classdef files, it works only for
    a fixed list of common commands.
- **Browser app:** `input()` isn't available, and Stop clears the workspace.

## Behaviour differences

- **Speed:** vectorized code runs at NumPy speed, and simple numeric loops
  are compiled automatically (20–50× faster than interpreting). Loops that
  call user functions or work with cells, structs or text are interpreted, at
  roughly 2–10 µs per simple statement.
- **Display:** command-window output follows the familiar layout. Column
  widths, scale-factor headers and blank lines may differ slightly. Very wide
  matrices aren't split into column blocks.
- **Random numbers:** `rng(seed)` makes results repeatable within mylab, but
  the sequences differ from MATLAB's for the same seed.
- **Solvers** (`ode45`, `fzero`, `fminsearch`, `fmincon`, `integral`, …) use
  SciPy algorithms. Results agree to within tolerances, but step counts and
  the last digits can differ.
- **Plots** are drawn with Matplotlib and its default style.
- **Errors:** messages are written for mylab and are not word-for-word copies.
  Error identifiers you pass to `error` are kept.
