# Changelog

## 1.1.0

- Hover help, completion and call hints cover the built-in functions of
  mylab 1.1.0: more than 2,800 entries (about 770 in 1.0.0), including the
  Curve Fitting functions, HDF5 and NetCDF, `polyshape`, `timeseries` and
  the methods of `RandStream` and other objects.
- Use it with mylab 1.1.0 (`pip install --upgrade mylab-lang`): the Problems
  panel and the run commands call the `mylab` you have installed.

## 1.0.0

First release.

- Language "MATLAB (mylab)" for `.m` files: syntax colouring, comment and
  bracket handling, automatic indentation, folding of blocks and `%%`
  sections, outline of functions and sections.
- Commands: Run File, Run Selection / Current Section, Debug File, Open
  Interactive Prompt, Show Help for Word Under Cursor.
- Problems panel: syntax errors and unknown functions from `mylab --check`,
  on open, on save and while typing.
- Hover help, completion and call hints for mylab's built-in functions, from
  data bundled with the extension. Hover help for functions in the same folder.
- Snippets for `for`, `if`, `while`, `switch`, `function` and `try`.
- Settings: `mylab.executable`, `mylab.checkOnType`, `mylab.runInTerminal`.
- Also works in files that use the MathWorks extension's `matlab` language.
