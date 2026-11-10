# Compatibility notes

This page lists known differences between mylab and MATLAB. Please report a
difference that isn't listed here as an issue, with a short `.m` snippet,
what mylab printed, and what you expected.

## Not supported yet

- **N-dimensional arrays.** All arrays are 2-D. Indexing with a third index
  works only when that index is `1`.
- **String arrays.** `"text"` works as a single string (one string scalar),
  including `+` to join strings. Arrays of strings such as `["a" "b"]` do not
  work yet. Use a cell array `{'a', 'b'}` instead.
- **Classes.** `classdef`, objects, methods and packages are not supported.
- **Tables, timetables, datetime, categorical, containers.Map** are not
  supported.
- **Sparse matrices.** `sparse(...)` builds an ordinary full matrix.
- **Toolboxes.** Only some Signal Processing, Statistics, Optimization and
  Control functions exist. Simulink is out of scope.
- **Graphics handles.** `gcf` and `gca` return figure numbers. `set` and
  `get` on graphics objects are not supported. Only common plot options are
  supported (`LineWidth`, `Color`, `DisplayName`, `MarkerSize`, …).
- **Text input/output.** `fopen`, `fprintf`, `fgetl`, `fileread`,
  `readmatrix`, `writematrix`, and `save`/`load` of plain numeric `.mat` data
  work. `textscan`, `fscanf` and `readtable` do not.
- **Command syntax** (`hold on`, `format long`) works for a fixed list of
  common commands. For any other function, use call syntax: `f('arg')`.
- **`nargin('fname')`**, `inputname`, `evalin` and `assignin` are not
  available.

## Behaviour differences

- **Speed:** mylab is an interpreter written in Python. Vectorized code runs
  fast because NumPy does the work. Scalar loops are much slower than in
  MATLAB, roughly 10 µs per simple statement. Vectorize hot loops where you
  can.
- **Display:** the command-window format follows MATLAB's layout. Column
  widths and scale-factor headers may differ slightly. Very wide matrices are
  not split into "Columns 1 through N" blocks.
- **Random numbers:** `rand`, `randn` and `randi` use NumPy's generator. With
  `rng(seed)`, results repeat from run to run within mylab, but they differ
  from MATLAB's sequence for the same seed.
- **Solvers:** `ode45`, `fzero`, `fminsearch`, `integral` and the others use
  SciPy algorithms. Results agree to within tolerances, but step counts and
  the last digits can differ.
- **Plots** are drawn with Matplotlib and its default style.
