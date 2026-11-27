# MEX files in mylab

A MEX file is a function written in C or C++ that is compiled into a shared
library and then called from a program like any `.m` function. mylab can
build MEX files from source with its `mex` command and run them on Linux,
macOS and Windows (Windows support is best effort). They do not run in the
browser version of mylab.

```matlab
mex timestwo.c          % writes timestwo.mexa64 (on Linux) into the current folder
y = timestwo([1 2 3])   % called like any other function
```

A worked example is in [examples/mex](../examples/mex): `colstats.c` and the
script `mex_demo.m` that builds and calls it.

## What you need

A C compiler on the system path: `cc`, `gcc` or `clang` (on Windows, MinGW-w64
`gcc`, `clang`, or `cl` from a Visual Studio developer prompt). For C++
sources, `c++`, `g++` or `clang++`. The `CC` and `CXX` environment variables
choose a different compiler. `mex -setup` shows which ones mylab found.

Nothing is compiled when mylab is installed, and mylab ships no compiled code
for MEX support.

## How it works

All of the glue between a MEX file and mylab is Python:

- mylab ships its own headers, written for mylab: `mex.h` and `matrix.h` in
  `mylab/mex/include`. They declare the usual `mx*` and `mex*` functions and
  types, so C code written for the MEX interface compiles unchanged, but every
  one of those functions is a small inline function that calls through a
  table of function pointers (a struct defined in `mylab_mex_api.h`).
- `mex` compiles your sources together with a short entry file from the same
  folder (`mylab_mex_entry.c`) into a shared library. Nothing is linked
  against a matrix library.
- When a program calls the function, mylab loads the library with `ctypes`,
  fills the table with callbacks written in Python (`mylab/mex/runtime.py`),
  converts the arguments, and calls the exported entry point
  `mylab_mex_entry`, which calls your `mexFunction`.

The table is described once, in `mylab/mex/spec.py`; the header is generated
from it with `python scripts/gen_mex_header.py`, and a test checks that the
two agree.

## Building: `mex`

```
mex [options] source.c [more.c ...] [object or library files]
```

| Option | Meaning |
| --- | --- |
| `-output name` | name of the MEX file (default: the first source file's name) |
| `-outdir dir` | folder to write it to (default: the current folder) |
| `-Idir`, `-Ldir`, `-lname` | include folder, library folder, library to link |
| `-Dname`, `-Dname=value`, `-Uname` | define or undefine a macro (`-Dname#value` also works) |
| `-O` / `-g` | optimised build (default) / debug build, which also turns on `mxAssert` |
| `-R2017b` / `-R2018a` | separate (default) / interleaved complex data for `mxGetData` |
| `-largeArrayDims`, `-compatibleArrayDims` | accepted and ignored: sizes are always `size_t` |
| `-v` | print the compiler commands and output |
| `-n` | print the commands without running them |
| `-silent` | print nothing when the build works |
| `CFLAGS="..."`, `CXXFLAGS="..."`, `LDFLAGS="..."` | replace the default flags; `$CFLAGS` etc. inside the value keeps them |

The file gets the extension `mexext` returns: `.mexa64` on Linux, `.mexmaci64`
or `.mexmaca64` on macOS (Intel, Apple silicon) and `.mexw64` on Windows.
Files ending in `.c` are compiled as C; `.cpp`, `.cc` and `.cxx` with the C++
compiler, in which case the program is linked as C++.

Building a MEX file that is loaded replaces the loaded copy (unless it is
locked with `mexLock`), so the next call runs the new code.

## Calling MEX files

- A MEX file in the current folder or on the path is found by name, just like
  a `.m` file. In the same folder, the MEX file wins over a `.m` file of the
  same name. (A `.m` file next to it is still used for `help name`.)
- `exist('name')` returns 3, `which name` shows the file, and `what` lists MEX
  files.
- Handles, `feval`, `cellfun` and the other ways of calling a function work.
  `nargin('name')` and `nargout('name')` return -1.
- `nlhs` is the number of outputs the caller asked for (`nargout`). With
  `nlhs` 0 the function may still set `plhs[0]`, which becomes `ans`. If the
  caller asked for outputs that the function did not set, the call fails with
  `MATLAB:unassignedOutputs`.
- `clear name`, `clear mex`, `clear functions` and `clear all` unload MEX
  files, except locked ones. Unloading runs the function registered with
  `mexAtExit` and frees persistent arrays and memory. mylab also runs the
  `mexAtExit` functions when Python exits.

Each loaded MEX file is a private copy of the file on disk, so the file can be
rebuilt or deleted while it is loaded. A file replaced by other means keeps
running the old code until it is cleared.

## Data and memory

- An `mxArray *` is an opaque handle. Do not dereference it.
- Data pointers (`mxGetPr`, `mxGetData`, `mxGetDoubles`, `mxGetChars`,
  `mxGetIr`, ...) point into NumPy buffers owned by mylab. Data is stored in
  column-major order: element (i, j) of an m-by-n matrix is at `i + j*m`.
  Writing through them changes the array.
- Inputs (`prhs`) are copies of the caller's values, so changing them cannot
  change a variable in the program; treat them as read-only anyway.
- Complex data is stored interleaved. The separate-complex functions
  (`mxGetPr`/`mxGetPi`, `mxGetData`/`mxGetImagData` in the default build) split
  it into a real and an imaginary buffer the first time they are used on an
  array, and the interleaved ones (`mxGetComplexDoubles`, ..., or `mxGetData`
  in a `-R2018a` build) join it again. Switching between the two styles on
  the same array moves the data, so pointers obtained before the switch
  become stale. Code that sticks to one style is not affected.
- `mxGetPr` and `mxGetData` return `NULL` for an empty array, and for cell,
  struct and opaque arrays.
- Arrays created during a call and not returned in `plhs` are destroyed when
  the call returns, and so is memory from `mxMalloc`, `mxCalloc` and
  `mxRealloc` that was not freed. `mexMakeArrayPersistent` and
  `mexMakeMemoryPersistent` keep them until the MEX file is cleared (or until
  `mxDestroyArray`/`mxFree`). Keeping a non-persistent handle in a static
  variable and using it in a later call raises an error instead of crashing.
- Memory given to an array with `mxSetPr`, `mxSetData`, `mxSetIr`, ... must
  come from `mxMalloc`, `mxCalloc` or `mxRealloc`; the array takes it over.
- `mxSetM`, `mxSetN` and `mxSetDimensions` change the shape only; give the
  array a big enough buffer with `mxSetData` if it grows.
- Cells and structs own their elements. `mxGetCell` and `mxGetField` return
  the element itself (not a copy); `mxSetCell` and `mxSetField` take over the
  new element and release the old one when the call ends. `mxSetField` with a
  field the struct does not have adds the field.
- Function handles, objects, string arrays, tables and other values that have
  no mxArray layout are passed as opaque arrays: their class name is right,
  `mxIsFunctionHandle`/`mxIsObject` work, and they can be passed back to mylab
  (as an output or through `mexCallMATLAB`), but C cannot look inside them.
- Character data is UTF-16 (`mxChar` is 16 bits). `mxArrayToString`,
  `mxArrayToUTF8String`, `mxGetString` and `mxCreateString` use UTF-8.

## Errors

A Python exception must never travel through C code, and C code must never
see a half-finished Python call. mylab handles errors like this:

1. Every Python callback catches every exception, stores it, sets a `pending`
   flag in the function table and returns a harmless value (0 or `NULL`).
2. The inline wrapper in the header checks the flag after every call. If it is
   set (or the function was `mexErrMsgTxt`/`mexErrMsgIdAndTxt`), it calls
   `mylab_mex_unwind`.
3. In C, `mylab_mex_unwind` does a `longjmp` back to the `setjmp` in the entry
   point, which returns 1 to Python. In C++ (when any source is C++, the entry
   file is compiled as C++ and C sources get `-fexceptions`), it throws a
   private exception that the entry point catches, so destructors run on the
   way out. Any other C++ exception that escapes `mexFunction` becomes an
   error with identifier `mylab:mex:cppException`.
4. Back in Python, the stored error is raised in the calling program, where
   `try`/`catch` sees it with its identifier and message
   (`MException.identifier`).

So `mexErrMsgIdAndTxt('mytool:badInput', ...)` behaves like `error` with that
identifier, and an error inside code run by `mexCallMATLAB` or `mexEvalString`
ends the MEX function and reaches the caller with its original identifier.
With `mexSetTrapFlag(1)`, those two return 1 instead. `mexCallMATLABWithTrap`
and `mexEvalStringWithTrap` return the `MException` (or `NULL`). Invalid
calls (a NULL or destroyed array, an index out of range, `mxSetCell` on a
non-cell) raise errors with identifiers that start with `mylab:mex:`.

Because the jump skips the rest of your C function, `malloc`ed memory and open
files in it are not released; memory from `mxMalloc` is.

## Supported functions

Arrays: `mxCreateDoubleMatrix`, `mxCreateDoubleScalar`,
`mxCreateNumericMatrix`, `mxCreateNumericArray`, `mxCreateUninitNumericMatrix`,
`mxCreateUninitNumericArray`, `mxCreateLogicalScalar`, `mxCreateLogicalMatrix`,
`mxCreateLogicalArray`, `mxCreateString`, `mxCreateCharArray`,
`mxCreateCharMatrixFromStrings`, `mxCreateCellMatrix`, `mxCreateCellArray`,
`mxCreateStructMatrix`, `mxCreateStructArray`, `mxCreateSparse`,
`mxCreateSparseLogicalMatrix`, `mxDuplicateArray`, `mxDestroyArray`.

Shape and class: `mxGetM`, `mxGetN`, `mxSetM`, `mxSetN`,
`mxGetNumberOfElements`, `mxGetNumberOfDimensions`, `mxGetDimensions`,
`mxSetDimensions`, `mxGetElementSize`, `mxCalcSingleSubscript`, `mxGetClassID`,
`mxGetClassName`, `mxIsClass`, and the tests `mxIsDouble`, `mxIsSingle`,
`mxIsInt8` ... `mxIsUint64`, `mxIsNumeric`, `mxIsComplex`, `mxIsChar`,
`mxIsLogical`, `mxIsLogicalScalar`, `mxIsLogicalScalarTrue`, `mxIsCell`,
`mxIsStruct`, `mxIsSparse`, `mxIsFunctionHandle`, `mxIsObject`, `mxIsOpaque`,
`mxIsEmpty`, `mxIsScalar`, `mxIsFromGlobalWS` (always false), `mxIsInf`,
`mxIsNaN`, `mxIsFinite`, `mxGetInf`, `mxGetNaN`, `mxGetEps`.

Data: `mxGetPr`, `mxGetPi`, `mxSetPr`, `mxSetPi`, `mxGetData`, `mxSetData`,
`mxGetImagData`, `mxSetImagData`, `mxGetScalar`, the typed accessors
`mxGetDoubles`/`mxSetDoubles` ... `mxGetUint64s`, `mxGetComplexDoubles`,
`mxGetComplexSingles`, `mxGetComplexInt8s` ... `mxGetComplexUint64s` (with
their `mxSet` forms), `mxGetLogicals`, `mxGetChars`; sparse `mxGetIr`,
`mxGetJc`, `mxSetIr`, `mxSetJc`, `mxGetNzmax`, `mxSetNzmax`.

Text: `mxArrayToString`, `mxArrayToUTF8String`, `mxGetString`.

Cells and structs: `mxGetCell`, `mxSetCell`, `mxGetField`, `mxSetField`,
`mxGetFieldByNumber`, `mxSetFieldByNumber`, `mxGetNumberOfFields`,
`mxGetFieldNameByNumber`, `mxGetFieldNumber`, `mxAddField`, `mxRemoveField`.

Memory: `mxMalloc`, `mxCalloc`, `mxRealloc`, `mxFree`.

MEX: `mexPrintf` (formatted in C with `vsnprintf`; the text goes to mylab's
output, so `evalc` captures it), `mexErrMsgTxt`, `mexErrMsgIdAndTxt`,
`mexWarnMsgTxt`, `mexWarnMsgIdAndTxt` (respecting `warning on/off/error`),
`mexCallMATLAB`, `mexCallMATLABWithTrap`, `mexEvalString`,
`mexEvalStringWithTrap`, `mexSetTrapFlag`, `mexGetVariable`,
`mexGetVariablePtr`, `mexPutVariable` (workspaces `'base'`, `'caller'` and
`'global'`), `mexAtExit`, `mexLock`, `mexUnlock`, `mexIsLocked`,
`mexFunctionName`, `mexMakeArrayPersistent`, `mexMakeMemoryPersistent`,
`mexIsGlobal` (always false), `mxAssert`, `mxAssertS`.

`mexCallMATLAB` calls functions by name (built-ins, `.m` files on the path,
other MEX files); `mexEvalString` and the `'caller'` workspace refer to the
workspace that called the MEX function.

## Not supported

- **MEX files compiled for MATLAB.** They link to MATLAB's own libraries, so
  they cannot be loaded. mylab recognises them (they lack mylab's entry
  point) and asks for them to be rebuilt from source with mylab's `mex`.
  Likewise, a MEX file built by another version of mylab must be rebuilt.
- **The browser version of mylab:** MEX files are native code, so calling or
  building one there raises an error that says so.
- The C++ MEX API (`mex.hpp`, `matlab::mex::Function`, MATLAB Data API) and
  Fortran MEX files. C++ files that use the C API above work.
- `mxGetProperty`/`mxSetProperty` (objects are opaque), `mexGet`/`mexSet`
  (graphics), `mxCreateSharedDataCopy` and other undocumented functions,
  `mex -c` (compile only), `-client`, option files (`-f`).
- Complex integer arrays cannot be returned (mylab has no complex integers).
- MEX files in `private` folders, class folders and packages are not looked
  up; put them on the path or in the current folder.
- Calls into mylab from threads other than the one running `mexFunction`.

## Performance

Each `mx`/`mex` call goes through a Python callback (about a microsecond). Get
data pointers once and loop over the data in C, rather than calling
`mxGetCell` or `mxGetScalar` in a tight loop. Arguments and results are
copied once on the way in and once on the way out. Calling a MEX function
costs roughly 100 microseconds.
