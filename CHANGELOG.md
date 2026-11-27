# Changelog

## 1.1.0

1.1.0 brings mylab's behaviour much closer to MATLAB's and adds a lot:
MEX files compiled from C and C++, computer vision on OpenCV, the Curve
Fitting core, HDF5 and version 7.3 MAT-files in the default install, state
charts and physical networks in mylab Blocks, and live collaboration in the
web app. It is tested continuously on Linux, Windows and macOS and in the
engines of Chrome, Firefox and Safari. Read *Breaking changes* first if you
used 1.0.0: several of the fixes change results that scripts may have relied
on.

### Breaking changes
- **Dependencies:** NumPy 1.26, SciPy 1.13, Matplotlib 3.8 and Pillow 10
  are now the minimum versions (were 1.24, 1.10, 3.7 and 9): they are the
  oldest set the full test suite passes on. **h5py 3.15.1 or newer is now
  required** (it was optional). On Linux its wheels need glibc 2.28 or newer.
- **Unknown name-value options are an error.** 1.0.0 silently ignored an
  option it did not know (a misspelt `'LineWidht'`, an option from a newer
  MATLAB release). Built-ins now raise MATLAB's error, for graphics
  "Unrecognized property 'X' for class 'Y'."
- **Figures are 560 x 420 pixels**, MATLAB's default (were 640 x 480).
  `getframe`, `VideoWriter` frames, animations and images saved at the
  screen resolution change size accordingly.
- **Colours are MATLAB's.** The letters `'g'`, `'c'`, `'m'` and `'y'` in a
  line specification, `'Color'` option or colour name are now [0 1 0],
  [0 1 1], [1 0 1] and [1 1 0] (were Matplotlib's darker colours), and the
  default colour order is MATLAB's (`[0 0.4470 0.7410]`, ...). Plots look
  different, and code that read `h.Color` gets the new values.
- **Graphics objects as MATLAB returns them:** `pie` returns a patch and a
  text for each slice (was one handle); `xline`/`yline` return
  `ConstantLine` objects (were lines); `h.Marker` reads `'square'`,
  `'diamond'`, `'pentagram'` and `'hexagram'` (was `'s'`, `'d'`, ...);
  `h.Prop` on several objects is a comma-separated list, so `{h.Color}` is a
  cell array (use `get(h, 'Color')` for the old form); image `CData` keeps
  its class (uint8, logical, ...) instead of becoming double.
- **Function lookup:** a function file in the current folder now takes
  precedence over one of the same name on the path, as in MATLAB.
- **`|` and `&` short-circuit inside `if` and `while` conditions**, as in
  MATLAB, so the right-hand side is not evaluated when the left decides.
- **Numeric results that change:**
  - `floor`, `ceil`, `fix`, `real`, `imag` and `conj` keep the class of
    integer inputs (returned double);
  - `sum` and `prod` of integer, logical and char arrays return double
    unless `'native'` is given;
  - a rectangular `A\b` returns a basic solution (pivoted QR), as MATLAB
    does, instead of the minimum-norm one;
  - `linspace` follows MATLAB's algorithm, with the end points exactly
    `a` and `b` (`linspace(0, Inf, n)` started with NaN);
  - `rand`, `randi`, `randn` and `randperm` return MATLAB's numbers after
    `rng(seed)`, so seeded random sequences differ from 1.0.0's.
- **Text:** `split` merges consecutive delimiters by default
  (`CollapseDelimiters` is true, as in MATLAB); `sscanf`/`fscanf` with a
  size limit stop where MATLAB does.
- **Display:** the command window prints values as MATLAB does (spacing,
  `format` modes, column blocks, class headers), so captured output
  (`evalc`, `disp` in tests) can differ from 1.0.0's.
- **Files:** `matfile` on a new file creates a version 7.3 MAT-file, as
  MATLAB does (was version 7).
- **`inputParser` optional inputs:** with no validator, text in an optional
  input's position is read as a parameter name, as MATLAB documents (1.0.0
  took it as the optional value). Give the optional input a validator, or
  pass it by name, to keep the old behaviour.
- **Empty text is 1x0:** `sprintf`, `strjoin`, `compose` and `blanks(0)`
  return a 1x0 char when the result is empty (was 0x0), so `size` and
  `isequal` against `''` change.
- **`max` and `min` of complex arrays** return a real array when every
  imaginary part of the result is 0.
- **Property and argument validation convert values** as MATLAB does: `[]`
  for a `(1,:)` property becomes 1x0, a scalar fills a fixed size, a vector
  in the other orientation is turned, and logical, integer and text values
  are converted to the declared class (they were refused or kept as given).
  Size errors read "Value must be a row vector (size 1x:), but its size is
  2x3."
- **`xline` and `yline` leave the axis limits alone:** `plot(1:3); yline(5)`
  keeps the y-limits at [1 3] (were widened to [1 5]), and a constant line
  outside the limits is not drawn. `'AffectAutoLimits','on'` (a mylab option;
  MATLAB does not document it for constant lines) brings the old behaviour
  back.

### MEX files and computer vision
- **MEX files.** `mex file.c` compiles C or C++ sources with the system
  compiler (cc, gcc or clang; best effort on Windows) into a MEX file
  (`.mexa64`, `.mexmaci64`/`.mexmaca64`, `.mexw64`), and programs call it like
  a `.m` function. All of the glue is Python: the sources include mylab's own
  `mex.h`, whose mx/mex functions call through a table of ctypes callbacks
  that mylab fills in when it loads the file, so nothing is compiled when
  mylab is installed. Numeric classes, logical, char, complex (separate and
  interleaved APIs), sparse, cell and struct arrays are supported, with
  `mxMalloc` memory and temporary arrays released after each call,
  `mexErrMsgIdAndTxt` errors that `try`/`catch` sees with their identifier,
  `mexPrintf`, `mexCallMATLAB`, `mexEvalString`, workspace access,
  persistent arrays, `mexAtExit` and `mexLock`. A MEX file takes precedence
  over a `.m` file of the same name in the same folder; `exist` returns 3 for
  it, and `clear name`/`clear mex` unload it. New: `mexext`; `mex` no longer raises `mylab:mex:unsupported`. MEX files built
  for MATLAB cannot be loaded and must be rebuilt from source; MEX files do
  not run in the browser. See [docs/MEX.md](docs/MEX.md).
- **Computer vision** (optional OpenCV dependency: `pip install
  "mylab-lang[vision]"`; the web app loads Pyodide's OpenCV on first use).
  See [docs/VISION.md](docs/VISION.md).
  - Features: `detectHarrisFeatures`, `detectMinEigenFeatures`,
    `detectFASTFeatures`, `detectORBFeatures`, `detectSIFTFeatures`,
    `detectBRISKFeatures`, `detectKAZEFeatures`, `detectMSERFeatures`,
    `detectSURFFeatures` (mylab's own Hessian detector, no non-free code), the
    point objects (`cornerPoints`, `SIFTPoints`, `MSERRegions`, ...) with
    indexing, `selectStrongest`, `selectUniform` and `plot`,
    `extractFeatures`, `binaryFeatures`, `matchFeatures` and
    `showMatchedFeatures`.
  - Geometry: `estgeotform2d`, `estimateGeometricTransform2D`,
    `estimateGeometricTransform`, `estimateFundamentalMatrix`,
    `estimateEssentialMatrix`, `estrelpose`, `relativeCameraPose`,
    `triangulate`, `epipolarLine`, `isEpipoleInImage`, `rigidtform3d`,
    `simtform3d`, `affinetform3d`, `transltform3d`, `rigid3d`, `affine3d`,
    `rotvec2mat3d` and related conversions; `insertShape`, `insertMarker`,
    `insertText`, `insertObjectAnnotation`.
  - Detection: `vision.CascadeObjectDetector` (OpenCV's BSD-licensed
    cascades or your own XML), `vision.PeopleDetector`/`peopleDetector`
    (OpenCV's HOG SVM), `yolov4ObjectDetector`, `yoloxObjectDetector` and
    `yolov8ObjectDetector` for user-supplied ONNX models (no weights are
    bundled), `selectStrongestBbox`, `selectStrongestBboxMulticlass`,
    `bboxOverlapRatio`, `bboxPrecisionRecall`, `evaluateDetectionPrecision`,
    `evaluateDetectionMissRate`.
  - Cameras: `cameraIntrinsics`, `cameraParameters`, `stereoParameters`,
    `detectCheckerboardPoints`, `generateCheckerboardPoints`,
    `estimateCameraParameters` (mono and stereo), `undistortImage`,
    `undistortPoints`, `world2img`, `worldToImage`, `img2world2d`,
    `pointsToWorld`, `cameraProjection`, `cameraMatrix`,
    `estimateExtrinsics`, `extrinsics`, `estworldpose`,
    `rectifyStereoImages`, `disparitySGM`, `disparityBM`, `reconstructScene`.
  - Tracking: `vision.PointTracker`, `opticalFlow`, `opticalFlowLK`,
    `opticalFlowHS`, `opticalFlowFarneback`, `vision.KalmanFilter`,
    `configureKalmanFilter`.
  - Point clouds: `pointCloud`, `pcread`/`pcwrite` (PLY and PCD, ASCII and
    binary), `pcshow`, `pcshowpair`, `pcdownsample`, `pcdenoise`,
    `pcnormals`, `pcfitplane`, `planeModel`, `pcregistericp`, `pcmerge`,
    `pctransform`, `pcsegdist`.

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
- `|` and `&` short-circuit in `if`/`while` conditions; command syntax in
  class methods; nested functions in a method keep the method's access;
  growing `[]` through colons; optional `inputParser` inputs given by name.
- Octave syntax (`+=`, `x++`, `#{ }#`, `do`/`until`, `unwind_protect`,
  `printf`, ...), `arguments` block options, MATLAB's error identifiers and
  messages for the common runtime errors, `MException` details.
- Source files that are not UTF-8 (Windows-1252, Shift-JIS) are read.
- An empty array of any class (`[]`, `''`, `zeros(1, 0)`, `struct([])`) adds
  nothing next to a cell array, so `files = []; files = [files; {d.name}']`
  builds a cell array of names.
- `num2cell` for struct, cell, char, logical, string and object arrays;
  `round` with `'TieBreaker'`; the old `rand('state')`, `rand('seed')` and
  `rand('twister')` forms (and `randn`'s) save and restore the sequence.
- `inputParser`: optional inputs given by name (a unique leading part is
  enough), `PartialMatchPriority`, ambiguous names to `Unmatched` with
  `KeepUnmatched`.
- `input` at the end of input raises `MATLAB:input:EndOfInput`, which
  `try`/`catch` can handle.

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
- **Curve fitting:** `fit`, `fittype`, `fitoptions`, `cfit` and `sfit` with
  the library models (polynomials, exponentials, power, Gaussian, sine and
  Fourier series, rationals, Weibull, interpolants, smoothing splines,
  polynomial surfaces), custom equations, weights, exclusions, robust fits,
  `confint`, `predint`, `differentiate`, `integrate` and `plot(f, x, y)`.
  Lowess and loess surface fits (`Span`, `Robust`) and the surface
  interpolants (linear, nearest, cubic, thin-plate, biharmonic). See
  docs/COMPATIBILITY.md.
- **`polyshape`** and `nsidedpoly`: polygons with holes and several regions,
  the set operations (`union`, `intersect`, `subtract`, `xor`), `polybuffer`,
  `convhull`, `rmslivers`, `simplify`, `triangulation`, `turningdist`,
  queries (`area`, `centroid`, `isinterior`, `nearestvertex`, ...), `plot`
  and object arrays.
- **`imageDatastore`** (also `datastore(..., 'Type', 'image')`): folders,
  wildcards, subfolders, labels from folder names, `read`, `readimage`,
  `readall`, `countEachLabel`, `splitEachLabel`, `shuffle`, `subset`,
  `partition`; `transform` and `combine` for image and text datastores.
- **More spreadsheets:** `.ods` files are read and written, and `.xls` files
  read, by `readtable`, `readmatrix`, `readcell`, `xlsread`, `sheetnames` and
  the rest of the `.xlsx` functions. `.xlsb` is refused with a message.
- **NetCDF-4 files** (`'Format'` `netcdf4` and `netcdf4_classic`, groups,
  chunking, compression, several unlimited dimensions) with `nccreate`,
  `ncwrite`, `ncread` (start, count, stride), `ncinfo`, `ncdisp`,
  `ncreadatt` and `ncwriteatt`.
- **Low-level HDF5:** a subset of the `H5F`, `H5G`, `H5D`, `H5S`, `H5A`,
  `H5T` and `H5ML` functions.
- `matfile` on version 7.3 files reads and writes only the indexed part of a
  numeric variable (`m.x(1:10, :)`), and `size(m, 'x')` and `whos(m)` read no
  data.
- **Dates in other languages:** `datetime(text, ..., 'Locale', loc)` reads
  month and day names in French, German, Spanish, Italian, Portuguese, Dutch,
  Swedish, Polish, Russian, Japanese and Chinese; `char`, `string` and
  `cellstr(t, fmt, locale)` show them in those languages.
- `view(3)` (and any `view(az, el)` that is not from above) turns a 2-D plot
  into a 3-D one with the same objects and handles; `view(2)` returns it.
  Arrays of graphics handles keep MATLAB's shapes under `'`, `reshape`,
  `repmat` and concatenation.
- `interp2` takes grid vectors of either orientation; `bsxfun` treats the
  dimensions one operand lacks as 1.
- `griddedInterpolant` (all methods, extrapolation, grid and scattered
  queries), `triangulation` and `delaunayTriangulation` objects, `voronoin`.
- **HDF5 and version 7.3 MAT-files in the default install:** `h5create`,
  `h5write`, `h5read`, `h5info`, `h5disp`, attributes, `save -v7.3`, `load`
  and `matfile`. Files follow MATLAB's layout (checked against hdf5storage
  and mat73). The browser app loads HDF5 support the first time a program
  uses it.
- Toolbox-level coverage added since 1.0.0: control systems (`tf`, `ss`,
  `zpk`, `frd`, `pid`, analysis and design), statistics and machine learning
  (distributions, tests, regression, clustering, learners), signal
  processing (filter design, spectra, wavelets), image processing (colour
  spaces, morphology, regions, `strel`), ODE suite with mass matrices,
  events, BVP/DDE/PDE solvers, adaptive quadrature, a unit-testing framework,
  MAT-file reader and writer for every standard type, chart types (heatmap,
  histogram2, bar3, swarm and bubble charts, geo axes, ...), figure files
  (`savefig`/`openfig`), datetime/duration/categorical axes.
- Graphics properties that now match MATLAB: `ax.YAxis` after `yyaxis`
  (two rulers), polar lines' `ThetaData`/`RData`, `ConstantLine` `Value`
  and labels, `pie` outputs, Z properties on 2-D axes, numeric graphics
  handles (`h = zeros(1, n); h(k) = plot(...)`).
- `echo`, `pkg load` (for the Octave packages mylab has built in), `bucky`,
  `randg`, `handle`, `fields`, `dlmread` ranges, `exist` for data files on
  the path.
- `timeseries` objects: construction from data, times and names (dates as
  text or `datetime`), `TimeInfo`, `DataInfo`, `QualityInfo` and events,
  `resample`, `synchronize`, `getsampleusingtime`, `addsample`/`delsample`,
  `append`, statistics with time weighting and quality codes, arithmetic,
  `plot`, conversion to and from timetables, and save/load in MAT-files.
- `properties`, `fieldnames` and `isprop` list the public properties of
  built-in objects (`datetime`, `duration`, `calendarDuration`,
  `containers.Map`, tables, `MException`, `RandStream`, graphs, ...), all
  from one list per class; `properties('datetime')` takes a class name.
  `calendarDuration` has a `Format` property.
- `scatter` and `scatter3` take one RGB colour per point (an M-by-3 matrix).

### Speed
- In the speed comparison with GNU Octave 10.3 (benchmarks/octave_compare),
  mylab is at least 1.5x faster on 19 of the 29 cases both can run and level
  on the other 10: function calls 24-46x, struct updates 14x, plain loops
  4-8x; matrix maths, FFT, cell/string building and object methods level.
- Anonymous functions defined inside function files are compiled (they never
  were): `arrayfun`/`cellfun`, `fminsearch` and `ode45` with such handles run
  3-8x faster.
- Sparse banded systems use LAPACK's banded solver; one long FFT uses several
  cores; `regexp` on cell arrays builds only the outputs asked for.
- Start-up imports SciPy, numexpr and the app server on first use, and now
  also the archive, CSV, JSON, audio and XML modules.
- `x(i, :) = v`, `x(:, j) = v`, `x(mask) = v` and other stores into an array
  no other variable shares are made in place instead of copying the array: a
  loop that fills the rows of a large matrix is no longer quadratic.
  `regexprep` on cell arrays is about twice as fast.
- Loops over integer-class values (`int8` ... `uint64`) are compiled, with
  MATLAB's rounding and saturation: a 200,000-step `int32` loop runs in
  about 0.05 s (was 2 s). Elements of 3-D arrays and of integer arrays are
  compiled too, and hot `if`/`while` conditions in loops that stay
  interpreted.
- `s.f(i) = v` and `s(k).f(i) = v` store in place when nothing else shares
  the struct, and so do stores into global and persistent arrays. Functions
  with several outputs are compiled when called for one.
- Sparse matrices: `bsxfun` keeps them sparse, a sparse matrix times a sparse
  vector stays sparse, rows and columns are extracted directly, and products
  of integer-valued sparse matrices are exact.
- Real programs from the corpus: DGCNN's `demo_p2k` 371 s -> 39 s and
  `demo_grid` 31 s -> 11 s, WEC-Sim's bemio 413 s -> 226 s, PRMLT's
  `mrf_demo` 44 s -> 34 s. `svd`, `eig`, `expm` and the other factorisations
  of matrices up to 2000 rows run on one BLAS thread, which is faster for
  them; for matrices above about 100 rows, results can differ from 1.0.0's
  in the last bit.

### Fixed
- **Python 3.14:** `b = a; b(2) = 5` also changed `a` (Python 3.14 lends
  references without counting them, and an in-place fast path relied on the
  count). The thresholds are now measured at import, and the fast paths turn
  themselves off where they cannot be trusted (free-threaded builds).
- **Windows, Python 3.10:** deeply recursive programs crashed the process
  (Windows' 1 MB thread stack); deep calls now continue on a larger stack.
- A file written within the same clock tick as the last directory listing
  was not found (Windows CI); persistent variables could pass to a re-parsed
  function; argument names leaked from one call to the next; NaN handling in
  the fast numeric paths matches the general one.
- Block diagrams: a non-numeric block parameter is reported by its MATLAB
  class instead of a Python error.
- `regexprep` evaluates `${...}` expressions in the caller's workspace;
  `fopen` accepts an empty encoding; `exist('model.slx')` returns 4 for
  `.slx` and `.mdl` files.
- **Objects in MAT-files** (version 7 and 7.3) keep everything across
  `save` and `load`: the value of a class derived from `double` or an integer
  class, every field of an old-style `@folder` object, dynamic properties,
  timetables with their row times, all table properties, dictionaries,
  `MException` causes and stack, and handle objects (and `containers.Map`)
  shared between variables or between `save -append` calls. Loading one
  variable that refers to a handle stored in another works, also through
  `matfile` and `whos -file`. Load rebuilds objects through their property
  validation and set methods, and `loadobj` gets a struct when the saved
  properties no longer fit the class. Objects that MATLAB itself writes into
  version 7.3 files still load as empty, with a warning.
- `isequal` compares tables and timetables by value (two tables built apart
  were unequal); `isequaln` works for `calendarDuration`. Missing values follow
  MATLAB: `isequal` never finds a missing string, `NaT` or undefined category
  equal (two missing strings were equal), and `isequaln` treats them as equal,
  also inside cells, structs, tables and sparse matrices (`isequaln({NaT},
  {NaT})` was false).
- `print` accepts `-tiff` (an EPS preview; the file is written without it);
  `boxplot(X, g)` takes one group value per column of a matrix; `set` on an
  empty handle array does nothing; `annotation(fig, ...)` puts the
  annotation on the given figure, panel or tab.
- **Graphics:** after `yyaxis right`, `xlabel`, `title`, `grid`, `xlim`,
  x-tick functions and `box` act on the chart's one shared x-axis, title and
  grid, and `ax.Legend` finds the legend from either side; `cla reset`
  clears both sides. A constant line at an axis limit is drawn in full over
  the frame. `subplot` finds an existing axes again after `colorbar` (it
  made a new one and deleted the old axes and its colorbar).

### Web app
- Editor conveniences: a command palette (Ctrl+Shift+P) and file switcher
  (Ctrl+P); find and replace, and search across the project; problems
  underlined as you type (from `mylab.lint`) with a Problems list; help on
  hover, usage hints and F1 help; richer autocomplete; go to definition,
  outline and rename; auto-indent, *Format document*, folding, line and
  multi-cursor shortcuts.
- A variable viewer with a paged grid, copy as CSV and quick plots.
- "Run to the next breakpoint"; tabs, cursor positions and breakpoints are
  remembered per project.
- Light, dark and high-contrast themes that follow the system, text size,
  optional MATLAB-style keys, a shortcut list built from the command table,
  and a local version history with a diff view.
- The assistant offers "Fix this error" and "Explain selection", and shows
  its edits as a diff to accept or reject.
- The 1.1 features in the app: help, hover and suggestions for dotted names
  (`RandStream.create`) and for the properties and methods of a workspace
  object after `f.`; the variable viewer shows objects (`cfit`,
  `griddedInterpolant`, class instances) property by property, timetables
  and datetime/duration/categorical arrays; the Files panel imports and opens
  `.mat`, `.h5`, `.nc`, spreadsheet, CSV, JSON and image files with a
  double-click; figures are shown at their own size, sharp on high-resolution
  screens; examples in `whats_new/` and a *What's new in 1.1* dialog.
- Block diagrams are easier to find: the six example models are in the
  `models` folder of *My files* (added once for people who already had the
  `.m` examples), the welcome window has a *Block diagrams* card, and New file
  offers *Block diagram* next to *Script or function* (a small model to start
  from). Model files have their own icon in the Files panel.
- Every file the app loads from a CDN (CodeMirror, JSZip and Pyodide's core
  files) is checked against a SHA-384 hash before it runs, and Pyodide checks
  its packages against its lock file, so a changed CDN file is refused. The
  assistant's Anthropic SDK is bundled with the app instead of loaded from a
  CDN, with the licences of its packages (`web/vendor/`).
- Accessibility: comments, keywords and line numbers in the editor meet the
  4.5:1 contrast minimum on highlighted lines; the panel resize handles report
  their size to assistive technology.
- Live collaboration: **Live** in the top bar starts a session on the current
  project and makes a link; whoever opens it (asked for a name once) gets the
  files in a temporary project they can keep. Everyone edits the same files at
  the same time (Yjs merges the edits, so nothing typed is lost; Ctrl+Z undoes
  only one's own typing), sees the others' cursors and selections in their
  colours, and new files, renames and deletions reach everyone. A person who
  shares their session can be followed in the new *Live session* tab: their
  command window, figures and workspace, shown as text and PNG images; code
  from others runs only when you press Run.
- Live sessions are browser to browser (WebRTC, through y-webrtc) with no
  server of mylab's and no account. The link carries a random session id and
  a 256-bit key after the `#`; signalling messages are encrypted with it
  (AES-GCM) and the connections by WebRTC (DTLS), so the public signalling
  server (`wss://y-webrtc-eu.fly.dev` by default) sees only a hashed topic, IP
  addresses and sizes. Signalling, STUN and TURN servers can be set in
  Settings; `scripts/signaling_server.mjs` runs a signalling server without
  any package. Received file paths are checked like imports and sizes are
  capped (1 MB of text, 2 MB otherwise, 16 MB in all).
- `scripts/vendor.mjs` (was `vendor_anthropic_sdk.mjs`) bundles every package
  the app serves itself: the Anthropic SDK and, new, Yjs 13.6.33, y-webrtc
  10.3.0, y-protocols 1.0.7, lib0 0.2.119 and simple-peer 9.11.1
  (`web/vendor/live-collab.mjs`), with their licences.

### Block diagrams
- `.slx` import, measured again on the 82 public files: 31 of the 68 models
  run (was 30). The quadrotor model, whose dynamics are an S-function file,
  now finishes within the 400 s limit (about 330 s), helped by the faster
  interpreter.
- **Iterator, function-call, switch-case and resettable subsystems.** For
  Iterator and While Iterator subsystems run their blocks several times in
  a time step; function-call subsystems run when a Function-Call Generator
  calls them; a Switch Case block switches action subsystems like an If
  block; a Reset block inside a subsystem restarts its states. Enabled and
  action subsystems can reset their states when they are switched on again,
  and their Out blocks can go back to the initial output while off.
- **Buses carry names.** Bus Creator, Bus Selector (nested names, output as
  a bus) and the new Bus Assignment work on any bus, not only one coming
  straight from a Bus Creator. Buses pass MATLAB Function blocks as structs,
  Constant blocks with a struct value make one, To Workspace logs one as a
  struct, and `Simulink.Bus` / `Simulink.BusElement` objects name the
  elements of a Bus Creator.
- **Model blocks** run another model file (normal mode).
- **`.mdl` files are read** (the older text format), with the same coverage
  as `.slx` files, including library links to `.mdl` libraries.
- **Writing `.slx` and `.mdl`:** `save_system(model, 'name.slx')` (or
  `.mdl`) and *Save a copy as .slx / .mdl* in the diagram editor write a copy
  that mylab reads back into the same model. The files are made for mylab
  and were not tested with any other program.
- **State charts:** the new Chart block runs hierarchical state machines
  with exclusive and parallel states, default transitions, junctions,
  event-triggered transitions with conditions, condition actions and
  transition actions, entry/during/exit actions in MATLAB code, input,
  output and local data and events (function-call output events call
  function-call subsystems), and `after`/`before`/`every`/`at` with ticks or
  seconds. `chart_state`, `chart_transition`, `chart_junction`, `chart_data`,
  `chart_event` and `send` build and drive charts from code; charts are
  kept in `.slx` and `.mdl` files mylab writes.
- **Physical networks:** electrical (resistor, capacitor, inductor, diode,
  voltage and current sources, reference, sensors), translational and
  rotational mechanical blocks (mass and inertia, springs, dampers, force,
  torque and velocity sources, references, ideal sensors) joined by two-way
  connections. Connected blocks form a differential-algebraic system of
  index 1, solved with the ordinary solvers; networks that are not index 1
  are errors that name the components. PS-Simulink, Simulink-PS and Solver
  Configuration blocks, Connection Port blocks for subsystems, and
  conserving ports in the diagram editor.
- The import report on the 82 public `.slx` files was run again: the same
  results (50 imported fully, 31 of 68 models run); the blocks that became
  convertible are in libraries or in models that also use toolbox blocks.
- The `.slx` reader reads a model's description, and MATLAB Function code
  whose block path has spaces at the end of a name.
- **Selector and Assignment** take any number of dimensions, starting
  indices, and indices from input ports in every dimension; Assignment can
  start from a given size instead of a Y0 input. Signals with more than two
  dimensions keep their shape.

### Testing and continuous checking
- CI on every push: lint; the full test suite on Linux, Windows and macOS
  with Python 3.10 and 3.14; the oldest and newest NumPy/SciPy/Matplotlib;
  the browser app; 705 + 469 MATLAB-behaviour checks with a ratchet (a check
  that passes may never fail again); a speed job that fails on a slowdown of
  more than 25%.
- The browser tests check the integrity hashes (a changed CDN file must be
  refused) and run axe-core for the WCAG 2.2 A and AA rules it can test, in
  three themes, the dialogs, the diagram editor and on a phone screen.
- The browser tests check live collaboration with three people in three
  browser contexts, and a signalling server and a STUN server started by the
  test, so no public service is used (`scripts/test_web_live.mjs`).
- After each release, and every Monday, a workflow installs mylab-lang from
  PyPI on Linux, Windows and macOS and runs the examples with it, and opens
  the GitHub Pages site in Chromium, Firefox and WebKit to run code there
  (`scripts/check_live_site.mjs`).
- A MATLAB validation kit (validation/), on hold and never run: MathWorks'
  licence terms forbid using MATLAB to test mylab (see
  docs/MATLAB-TERMS-REVIEW.md).
- A regression suite of 25 real repositories from the MATLAB corpus (185
  scripts and test suites, all passing), fetched on demand:
  `python datasets/matlab-corpus/tools/run_corpus.py run`.

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
- **mylab Blocks:** block-diagram modelling and simulation. A hybrid
  simulation engine with zero-crossing location, 107 blocks, readable
  `.mlb.json` model files, and `sim`, `new_system`, `add_block`, `add_line`,
  `set_param`, `get_param`, `find_system`, `linmod`, `trim`. The web app has
  a diagram editor with live scopes. Algebraic loops are solved (Newton);
  `AlgebraicLoop = error` reports them instead.
- **`.slx` import:** model callbacks and mask initialization run before a
  simulation; If/action and variant subsystems; MATLAB-language S-functions
  (level 1 and 2); library forwarding tables; stand-ins for kinematic
  vehicle models. Of 82 public files, 14 are block libraries; 50 files
  import fully, and 30 of the 68 models simulate (24 with every block real,
  6 with a stand-in subsystem). 15 models have behaviour checks. Physical
  modelling and compiled S-functions are not supported. See docs/BLOCKS.md.
- New functions: rotation and transform conversions (`eul2rotm`,
  `rotm2quat`, `tform2eul`, …), `rotx`/`roty`/`rotz`, `angdiff`, `coder.*`
  directives (accepted, no effect), `simsizes`.
- **Apps:** `uifigure` and its components, classic `figure` + `uicontrol`,
  `guidata`, callbacks with event data, and App Designer-exported class
  files. `mylab app.m` shows the app in the browser through a local server.
  The web app has an App panel, "Open as app" share links and a visual app
  builder that generates `.m` code and reopens it. See docs/APPS.md.
- Changed: class constructors called as a statement see `nargout == 0`;
  `@obj.method` creates a handle bound to the object.
- **Symbolic math** with MATLAB's syntax, backed by SymPy (optional:
  `pip install "mylab-lang[symbolic]"`): `syms`, `sym`, `diff`, `int`,
  `limit`, `taylor`, `simplify`, `subs`, `solve`, `dsolve`, `laplace`,
  symbolic matrices, `vpa`, `matlabFunction`, `latex`. In the web app SymPy
  loads on first use. See docs/SYMBOLIC.md.
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
