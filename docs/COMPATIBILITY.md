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
  - operator overloading, custom `disp`/`display`, `delete`/`isvalid`;
  - object arrays (`objs(3) = Point(1, 2)`, `[objs.x]`, `Point.empty`);
  - `enumeration` blocks, `events` with `notify` and `addlistener`;
  - methods in separate files of an `@ClassName` folder, and functions and
    classes in `+package` folders.
- Nested functions sharing their parent's variables, `arguments` blocks,
  `f(Name=Value)` syntax, and Octave-style `endfunction`/`endif` block ends.
- `MException` objects with MATLAB-style error identifiers;
  `warning('off', id)`.
- `containers.Map`, `table` (joins, grouping, `readtable`/`writetable`),
  sparse matrices, `datetime`, `duration` and `categorical`.
- `inputParser`, `onCleanup`, `RandStream`, `timer`.
- Image reading, writing and processing; binary file I/O; animations.
- JSON (`jsondecode`/`jsonencode`), WAV audio files and Excel `.xlsx` files.
- Block-diagram models (mylab Blocks): `sim`, `new_system`, `add_block`,
  `add_line`, `set_param`, `get_param`, `find_system`, `linmod`, `trim` and a
  library of 96 blocks, with a diagram editor in the web app and import of
  simple `.slx` files. See [BLOCKS.md](BLOCKS.md) for what is covered and
  what is not (no code generation, state charts, physical modelling,
  S-functions or algebraic-loop solving; `.mdl` files are not read).
- About 1,250 built-in functions; see [FUNCTIONS.md](FUNCTIONS.md).

## Not supported yet

- **classdef features:** `subsref`/`subsasgn` overloading, old-style
  classes without `classdef`, `private` folders, and classes derived from
  built-in types such as `double` (except enumerations). Object arrays can't hold objects of different classes
  unless you put them in a cell array.
- **Other data types:** timetable, and `containers.Map` with non-scalar keys.
- **Tables:** `rowfun`, `stack`/`unstack`. Table variables must be columns.
- **GUIs:** `uicontrol`, `uifigure`, GUIDE and App Designer raise a clear
  error. `.fig` files can't be saved or opened.
- **Input and output:** old binary Excel files (`.xls`), audio formats other
  than WAV, HDF5 and XML files, bit-field precisions (`'ubitN'`) in
  `fread`/`fwrite`. `textscan` and `fscanf` handle common formats only.
- **Toolboxes:** only the parts listed in FUNCTIONS.md (image processing,
  signal processing, statistics, optimization). Simulink, Symbolic Math,
  Control System, deep learning and others are out of scope for now.
- **Language corners:** `nargin('fname')`, `parfor` (runs as an ordinary
  `for` loop), `Repeating` and `Output` argument blocks, the `!command` shell
  escape, compiled MEX files.

## Things that work differently

- **Sparse matrices** are supported for double, complex and logical data in
  two dimensions.
  - Indexing a single element (`S(i,j)`) returns a full scalar.
  - Functions without a sparse version work on the full matrix and return a
    full result; converting more than 2^28 elements is refused with an error.
- **Object arrays:** growing an array (`objs(5) = obj`) fills the gap by
  calling the constructor with no arguments, once per new element.
- **`import`** stays in effect for the whole session, not only inside the
  function that calls it.
- **Enumerations:** each member is created once, the first time the class is
  used. Classes derived from a number type (`classdef Level < uint8`) convert
  with `double(x)` or `uint8(x)`; arithmetic on members needs that conversion.
- **Events:** listeners run immediately, in the order they were added.
  Property listeners support `PreSet` and `PostSet`.
- **`onCleanup`** runs when its function returns or fails, or when the
  variable is cleared. It does not run when the variable is overwritten, and
  it is not tracked inside cells or structs.
- **`timer`** has no background thread. `start(t)` runs the callbacks at once
  without waiting for `Period`; an endless timer keeps running inside
  `wait(t)` until a callback calls `stop(t)`.
- **`inputParser`:** text that is exactly a parameter name, followed by a
  complete list of name-value pairs, is read as that name even when an
  optional positional argument could take it.
- **Random streams:** `RandStream` accepts every generator name but always
  uses the same generator. `s = rng` and `rng(s)` save and restore the state.
- **Sound:** `sound`, `soundsc` and `audioplayer` play nothing. `audioread`
  and `audiowrite` handle WAV files only.
- **Excel:** `xlsread`, `xlswrite`, `readtable`, `writetable`, `readmatrix`,
  `writematrix`, `readcell` and `writecell` work with `.xlsx` files. Install
  `openpyxl` (`pip install mylab[excel]`) to keep formatting and other sheet
  contents when writing; without it a built-in reader and writer handles
  plain numbers, text and logical values. Dates are read as Excel serial
  numbers. Formulas are read as their last saved value.
- **Categorical arrays** support construction, comparison, indexing,
  `categories`, `countcats`, category editing, sorting, tables and grouping.
  Arithmetic-style functions such as `histogram` draw a bar chart of counts;
  `min`/`max`/`median` of ordinal arrays are not available.
- **Scattered interpolation:** `'natural'` uses linear interpolation.
  `scatteredInterpolant` extrapolates with the plane of the nearest triangle,
  which is close to MATLAB's result but not identical.
- **`message`:** there is no message catalog, so `getString(message(id))`
  returns the identifier and its arguments.
- **`uistack`, `movegui`, `web`, `linkprop`, `brush`** and similar desktop
  commands are accepted and do nothing.
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
