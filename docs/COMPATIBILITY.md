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
- `containers.Map`, `table` (joins, grouping, `readtable`/`writetable`),
  sparse matrices, `datetime` and `duration`.
- Image reading, writing and processing; binary file I/O; animations.
- About 1,100 built-in functions; see [FUNCTIONS.md](FUNCTIONS.md).

## Not supported yet

- **Object arrays:** you can't index classdef objects into arrays
  (`objs(3) = Point(...)`). Use a cell array instead: `objs{3} = Point(...)`.
- **classdef features:** events and listeners, enumerations,
  `subsref`/`subsasgn` overloading, and methods defined in `@ClassName` folders.
- **Other data types:** categorical, timetable, and `containers.Map` with
  non-scalar keys.
- **Tables:** `rowfun`, `stack`/`unstack`. Table variables must be columns.
- **GUIs:** `uicontrol`, `uifigure`, GUIDE and App Designer raise a clear
  error. `.fig` files can't be saved or opened.
- **Input and output:** Excel files, audio, bit-field precisions (`'ubitN'`)
  in `fread`/`fwrite`. `textscan` and `fscanf` handle common formats only.
- **Toolboxes:** only the parts listed in FUNCTIONS.md (image processing,
  signal processing, statistics, optimization). Simulink, Symbolic Math,
  Control System, deep learning and others are out of scope for now.
- **Language corners:** `nargin('fname')`, `parfor` (runs as an ordinary
  `for` loop), `Repeating` and `Output` argument blocks, the `!command` shell
  escape, compiled MEX files, `onCleanup`.

## Things that work differently

- **Sparse matrices** are supported for double, complex and logical data in
  two dimensions.
  - Indexing a single element (`S(i,j)`) returns a full scalar.
  - Functions without a sparse version work on the full matrix and return a
    full result; converting more than 2^28 elements is refused with an error.
- **Dates:** `datetime` supports the time zones `'UTC'` and `'local'` only,
  and English month and day names. Plot axes show dates as serial numbers.
- **Images:** `strel` returns a logical neighbourhood matrix, not an object.
  `edge`, `histeq`, adaptive `imbinarize` and bicubic `imrotate` agree
  closely with MATLAB but not pixel for pixel. 16-bit RGB files are handled
  at 8 bits per channel. MATLAB's sample images are not included.
- **Graphics handles:** plotting functions return handles whose common
  properties you can read and set (`h.LineWidth = 2`, `get`/`set`). Only a
  subset of properties exists. Lighting and mouse-interaction commands are
  accepted and ignored.
- **Animations:** when there is no live plot window (browser, Jupyter, MCP),
  each `drawnow`, `pause` or `getframe` records a frame, and the run returns
  an animated GIF (at most 150 frames). `VideoWriter` writes `.avi`/`.mp4`
  when ffmpeg is installed; otherwise it writes a GIF and warns.
- **Dialogs** (`inputdlg`, `questdlg`, `uigetfile`, `waitbar`, …) are text
  prompts. When nobody can answer (batch runs, the browser), they return
  their default answer and print a note.
- **Version checks:** `version` and `ver` report MATLAB 9.14 (R2023a) so
  scripts take their modern code path. `gpuArray`/`gather` pass data through
  (everything runs on the CPU), and `parpool` describes a single worker.
- **Command syntax** (`hold on`, `shading interp`, `addpath ../lib`) works
  when the first argument looks like a word, a path or quoted text. In
  classdef files and `arguments` blocks, it works only for a fixed list of
  common commands.
- **Debugging:** `mylab --debug`, `keyboard` and `dbstop` give a `K>>` prompt
  in a terminal. The browser app records a run and lets you step through it
  forwards and backwards; it can't pause a live run.
- **Browser app:** `input()`, `system` and web downloads aren't available,
  and Stop clears the workspace.

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
