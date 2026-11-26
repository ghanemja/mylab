# Compatibility notes

mylab aims to run everyday MATLAB-language code unchanged. This page lists
what isn't supported yet and where behaviour knowingly differs, as of 1.1.0.
Each statement here was checked against the 1.1.0 code.

If you find a difference that isn't listed, please
[report it](https://github.com/ghanemja/mylab/issues/new?template=report-a-difference.yml)
with a short `.m` snippet, what mylab printed, and what MATLAB printed. If you
have MATLAB, [validation/](../validation/) runs mylab's behaviour checks there
and lists what differs.

## How compatibility is checked

- 705 small scripts that assert documented MATLAB behaviour, and 469 more
  written afterwards and kept out of sight while gaps were closed. mylab
  passes 700 and 469 of them. Corrections to checks whose expectation was
  wrong about MATLAB are listed in
  [tests/probes/heldout/CORRECTIONS.md](../tests/probes/heldout/CORRECTIONS.md).
- 5,377 values compared with GNU Octave (see
  [tests/conformance](../tests/conformance/README.md)).
- 25 real repositories that run unchanged, 185 scripts and test suites in all
  (see [datasets/matlab-corpus](../datasets/matlab-corpus/README.md)).

None of these has been run against MATLAB itself by the project; the
expected values come from MATLAB's documented behaviour.

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
  - operator overloading, `subsref`/`subsasgn` overloading, custom
    `disp`/`display`, `delete`/`isvalid`;
  - object arrays, including arrays of different classes with a
    `matlab.mixin.Heterogeneous` base;
  - classes derived from built-in types (`classdef MyDouble < double`);
  - `enumeration` blocks, `events` with `notify` and `addlistener`;
  - methods in separate files of an `@ClassName` folder, functions and
    classes in `+package` folders, `private` folders, and old-style classes
    (`class(s, 'Name')` in an `@Name` folder).
- Nested functions sharing their parent's variables, `arguments` blocks
  (including `Repeating` and `Output`), `f(Name=Value)` syntax, `nargin('f')`.
- Octave syntax: `endfunction`/`endif` block ends, `+=` and friends, `x++`,
  `#{ }#` comments, `do`/`until`, `unwind_protect`, `printf`.
- `MException` objects; the common runtime errors have MATLAB's identifiers
  and message texts; `warning('off', id)`.
- `containers.Map`, `table` and `timetable` (joins, grouping, `rowfun`,
  `stack`/`unstack`, multi-column variables, `readtable`/`writetable`),
  sparse matrices, `datetime` (with IANA time zones), `duration`,
  calendar durations (`calmonths`, `calyears`, …) and `categorical`.
- `inputParser`, `onCleanup`, `RandStream`, `timer`, `parfor` (run as an
  ordinary loop), `gpuArray` (data stays on the CPU).
- Files: MAT-files of every version, including version 7.3 (`save -v7.3`,
  `load`, `matfile`, which reads and writes only the indexed part of a
  numeric variable); HDF5 (`h5create`, `h5write`, `h5read`, `h5info`,
  `h5disp`, `h5readatt`, `h5writeatt`, and a subset of the low-level `H5F`,
  `H5G`, `H5D`, `H5S`, `H5A` and `H5T` functions); NetCDF, classic and
  NetCDF-4 (`nccreate`, `ncwrite`, `ncread`, `ncinfo`, `ncdisp`, `ncreadatt`,
  `ncwriteatt`, with groups); XML (`xmlread`, `xmlwrite`); JSON;
  spreadsheets (`.xlsx` and `.ods`; `.xls` read-only); WAV and FLAC audio;
  images and `imageDatastore`; binary files with bit-field precisions
  (`'ubit4'`); figure files (`savefig`, `openfig`).
- Geometry: `polyshape` (with holes, several regions and the set operations
  `union`, `intersect`, `subtract`, `xor`), `nsidedpoly`, `triangulation`,
  `delaunayTriangulation`.
- Toolbox areas: image processing, signal processing, statistics and machine
  learning, optimization, control systems (`tf`, `ss`, `zpk`, `frd`, `pid`,
  `step`, `bode`, design functions), the Curve Fitting core (`fit`,
  `fittype`, `cfit`, `sfit`, including lowess, loess and interpolant
  surface fits), `griddedInterpolant` and
  `scatteredInterpolant`, ODEs including BVP, DDE and PDE solvers, and
  symbolic math with SymPy installed (see [SYMBOLIC.md](SYMBOLIC.md)).
- More than 2,700 built-in functions; see [FUNCTIONS.md](FUNCTIONS.md).
- Block-diagram models (mylab Blocks): `sim`, `new_system`, `add_block`,
  `add_line`, `set_param`, `get_param`, `find_system`, `linmod`, `trim` and a
  library of 107 blocks, with a diagram editor in the web app and import of
  `.slx` files (31 of 68 public models run). See [BLOCKS.md](BLOCKS.md) for
  what is covered and what is not (no code generation, state charts,
  physical modelling or compiled S-functions; `.mdl` files are not read).
- Apps written in code (`uifigure` and its components, `figure` +
  `uicontrol`, class files exported from App Designer); see
  [APPS.md](APPS.md).
- Rotation and transform conversions (`eul2rotm`, `rotm2quat`, `tform2eul`,
  …), `rotx`/`roty`/`rotz` and `angdiff`. The `coder.*` directives are
  accepted and do nothing; `coder.ceval` is an error.

## Not supported yet

- **Files:** Excel binary workbooks (`.xlsb`) and writing `.xls`; audio
  formats other than WAV and FLAC; HDF5 property lists other than
  `'H5P_DEFAULT'` (so `H5D.create` cannot make chunked or compressed
  datasets); NetCDF-4 user-defined types (compound, enum, vlen, opaque);
  `fileDatastore`, `augmentedImageDatastore`; `.mlapp` files and GUIDE
  layouts.
- **Toolboxes:** deep learning, computer vision, the Robotics System
  Toolbox's `se2`/`se3` objects, `timeseries`, Simscape and physical
  modelling, and the other toolboxes not listed above.
- **Language:** compiled MEX files and Java.

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
- **`inputParser`:** optional inputs are positional until text names an
  input (followed by valid name-value pairs), until the optional input's
  validator rejects the text, or at any text when the optional input has no
  validator, as MATLAB documents. Optional inputs can also be given by name,
  and partial matching covers their names. One difference: with no
  validator, a struct is expanded only when all its fields name inputs.
- **Random numbers:** after `rng(seed)`, `rand`, `randn`, `randi` and
  `randperm` return the same numbers as MATLAB's default generator (the
  Mersenne Twister). `RandStream` accepts every generator name, but every
  stream uses the Mersenne Twister. Functions that draw in a different order
  from MATLAB's (some samplers in the statistics functions) give different
  numbers for the same seed. The old `rand('state')`, `rand('seed')` and
  `rand('twister')` forms (and `randn`'s) work on the same generator: a saved
  state or seed restores the same sequence, but the numbers are not those of
  MATLAB's retired generators, and `rand` and `randn` share one stream.
- **Version 7.3 MAT-files:** `matfile` reads and writes only the indexed
  part of numeric, char and logical variables; cells, structs and objects are
  read and written whole, as are all variables of version 7 files. String
  arrays, tables, `datetime` and
  other objects are stored as structs that mylab turns back into objects;
  MATLAB sees them as structs. Like MATLAB, `matfile` creates a version 7.3
  file when the file does not exist yet.
- **Sound:** `sound`, `soundsc` and `audioplayer` play nothing.
- **Excel:** `xlsread`, `xlswrite`, `readtable`, `writetable`, `readmatrix`,
  `writematrix`, `readcell` and `writecell` work with `.xlsx` and `.ods`
  files, and read `.xls` files (values and stored formula results, no
  formatting; the optional `xlrd` package is used when installed). Install
  `openpyxl` (`pip install "mylab-lang[excel]"`) to keep formatting and
  other sheet contents when writing; without it a built-in reader and writer
  handles plain numbers, text and logical values. Dates are read as Excel
  serial numbers. Formulas are read as their last saved value.
- **Symbolic math** needs the optional SymPy package
  (`pip install "mylab-lang[symbolic]"`; the web app loads it on first use).
  Results are mathematically equal to MATLAB's but terms can be ordered or
  factored differently. `solve` returns the solutions it finds without
  parameters or conditions, assumptions are limited to simple conditions on
  one variable, and `matlabFunction` returns a function handle only. The full
  list is in [SYMBOLIC.md](SYMBOLIC.md).
- **Scattered interpolation:** `'natural'` (natural-neighbour) uses linear
  interpolation. `scatteredInterpolant` extrapolates with the plane of the
  nearest triangle, which is close to MATLAB's result but not identical.
- **Curve fitting:** `fit` solves nonlinear models with SciPy's
  `least_squares` (trust-region-reflective or Levenberg-Marquardt) from its
  own data-driven starting points, so a fit with several local optima can end
  at a different one than MATLAB's; iteration counts and `output` details
  differ. Robust fits report confidence bounds from the final weights. The
  default smoothing parameter of `'smoothingspline'` is `1/(1 + h^3/6)` (h the
  mean data spacing). Lowess and loess surface fits use Euclidean distance in
  the (optionally normalized) x-y plane, robust lowess makes five reweighting
  passes, and the linear, nearest and cubic surface interpolants are NaN
  outside the convex hull of the data.
- **`polyshape`:** the vertex order and start point of set-operation results,
  and the order of boundaries, follow mylab's own rule (each solid boundary
  followed by its holes). `rmslivers` is a heuristic, `polybuffer` draws arcs
  with 64 points per circle, `overlaps` counts touching shapes as
  overlapping, and `isinterior` counts boundary points as inside.
- **`griddedInterpolant`:** `'pchip'` and `'makima'` in more than one
  dimension interpolate along the first dimension first; `'cubic'` on a
  non-uniform grid uses `'spline'`.
- **`message`:** there is no message catalog, so `getString(message(id))`
  returns the identifier and its arguments.
- **Desktop commands:** `movegui`, `web` and similar commands are accepted
  and do nothing. `brush` records its state but nothing can be brushed
  without a mouse.
- **Dates:** `'Locale'` reads month and day names in English, French,
  German, Spanish, Italian, Portuguese, Dutch, Swedish, Polish, Russian,
  Japanese and Chinese; values are shown in English, and
  `char`/`string`/`cellstr(t, fmt, locale)` show them in another of these
  languages.
- **Images:** `edge`, `histeq`, adaptive `imbinarize` and bicubic `imrotate`
  agree closely with MATLAB but not pixel for pixel. 16-bit RGB files are
  handled at 8 bits per channel. MATLAB's sample images are not included.
- **Graphics:** figures are 560 x 420 pixels, with MATLAB's colour letters
  and default colour order, and objects return MATLAB's classes and
  properties (lines, axes and rulers, `ConstantLine`, patches, images, text,
  charts). Not every property of every object exists; lighting and
  mouse-interaction commands are accepted and ignored. Plots are drawn by
  Matplotlib, so fonts, tick placement and anti-aliasing differ from
  MATLAB's. A 2-D plot can be viewed in 3-D (`view(3)`); in that view images
  are drawn as a flat surface, contour sets and `pcolor` meshes are left out,
  and a later 2-D `plot` with `hold on` returns the axes to 2-D (MATLAB keeps
  the 3-D view).
- **Animations:** when there is no live plot window (browser, Jupyter, MCP),
  each `drawnow`, `pause` or `getframe` records a frame, and the run returns
  an animated GIF (at most 150 frames). `VideoWriter` writes `.avi`/`.mp4`
  when ffmpeg is installed; otherwise it writes a GIF and warns.
- **Dialogs** (`inputdlg`, `questdlg`, `uigetfile`, `waitbar`, …) are text
  prompts. When nobody can answer (batch runs, the browser), they return
  their default answer and print a note.
- **Version checks:** `version` and `ver` report MATLAB 9.14 (R2023a) so
  scripts take their modern code path. `parpool` describes a single worker.
- **Command syntax** (`hold on`, `shading interp`, `addpath ../lib`) works
  when the first argument looks like a word, a path or quoted text. In
  classdef files it works everywhere except inside `properties`, `events`,
  `enumeration` and `arguments` blocks, where `name value` is a declaration.
- **Debugging:** `mylab --debug`, `keyboard` and `dbstop` give a `K>>` prompt
  in a terminal. The browser app records a run and lets you step through it
  forwards and backwards; it can't pause a live run.
- **Browser app:** `input()`, `system` and web downloads aren't available,
  and Stop clears the workspace.

## Behaviour differences

- **Speed:** vectorized code runs at NumPy speed, and simple numeric loops,
  small functions and anonymous functions are compiled automatically.
  Other loops are interpreted, at roughly 2–10 µs per simple statement. On
  30 benchmark cases mylab took no longer than GNU Octave 10.3 on the same
  machine; it has not been timed against MATLAB.
- **Display:** command-window output follows MATLAB's layout (`format`
  modes, column blocks, class headers); spacing can still differ in rare
  cases.
- **Solvers** (`ode45`, `fzero`, `fminsearch`, `fmincon`, `integral`, …) are
  mylab's own implementations of the published methods, or SciPy's. Results
  agree within the tolerances, but step counts and the last digits can
  differ.
- **Errors:** the common runtime errors use MATLAB's identifiers and message
  texts; other messages are written for mylab. Error identifiers you pass to
  `error` are kept.
- **Python:** MATLAB code can call Python (`py.*`, `pyrun`). A host can turn
  this off with the environment variable `MYLAB_PYTHON=0`; see "Running code
  you did not write" in the [README](../README.md#running-code-you-did-not-write).
