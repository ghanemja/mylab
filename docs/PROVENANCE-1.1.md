# Provenance review of the changes after 1.0.0

This is a clean-room review of everything added to mylab between the 1.0.0
release and the start of the 1.1 work. It checks the rules in
[CLEANROOM.md](../CLEANROOM.md): no MathWorks documentation prose, example code
or source, and no GNU Octave source or tests. It was done in one sitting of
about 75 minutes, mostly with automated scans, plus a hand review of the
riskiest items. It is a screening, not a guarantee; see
[Limits](#limits-of-this-review).

## Summary

- **No copied Octave code, tests or help text was found.** The longest prose
  run shared with Octave is 14 words: a standard definition ("cumulative
  distribution function of the beta distribution with shape parameters a and
  b"). The longest code run is 24 tokens: the Robertson chemical kinetics
  ODE, a published test problem. Every shared long numeric vector is published
  mathematics, such as Butcher tableaux, Gauss-Kronrod nodes and the Rosser
  matrix.
- **No MathWorks documentation prose was found in a 20-entry sample** checked
  against the live reference pages. One 9-word definitional phrase in the
  `condeig` help matched MathWorks' wording. It was rewritten as a precaution.
- **Two new test files reuse examples from MathWorks documentation**, and
  their own docstrings say so. Both were fixed; see [Rewritten](#rewritten).
- No copyright headers, GPL text, `$Revision` tags or "ported from" notes
  were found in the new code.

## Scope

- Commit range: `19624ea` (1.0.0) to `8dee3c2`, compared with
  `git diff 19624ea HEAD`.
- 1,507 files changed, 305,433 lines inserted. 1,435 files are new; 163 of
  them are under `mylab/` (about 157,000 inserted lines there).
- The largest areas are `datasets/matlab-tasks` (662 files),
  `tests/conformance` (286), `mylab/builtins` (98), `mylab/helpdocs` (52, all
  new) and `web/js` (39).
- Only added or changed lines were examined. Text inherited from 1.0.0 is out
  of scope.

## Method

The scripts were kept outside the repository. They can be re-run from the
description below.

### 1. Similarity scan against GNU Octave

**mylab side.** The scan covered 3,517 units with 931,511 words in total:

- every entry in the `HELP` dicts of `mylab/helpdocs/*.py` (2,144 entries,
  usage lines plus description);
- for each added or changed `.py` file, the comments and string literals on
  added lines, extracted with Python's `tokenize`;
- for every other added or changed text file (`.m`, `.md`, `.json`, `.js`,
  `.txt`, `.html` and so on), the added lines.

Altogether, 1,394 files were covered.

**Octave side.** The scan covered 12,032 units with 1,889,064 distinct
shingles:

- every `.m` file under `/opt/mm/envs/oct10/share/octave/`: Octave 10.3 core,
  plus the installed Forge packages control, datatypes, image, io, optim,
  signal, statistics, struct and symbolic. Each file was split into its
  texinfo help block, its other comments, its `%!` test blocks and its code;
- the 600 extracted test files in `10.3.0/etc/tests` (`.tst` and `.cc-tst`);
- the C++ built-in help in `10.3.0/etc/built-in-docstrings`.

**Comparison.** Text was lowercased, split into `[a-z0-9]+` words, with
texinfo `@commands` removed, and compared as 8-word shingles. Two kinds of
shingle were dropped:

- 1,813 "boilerplate" shingles that occur in more than 25 Octave files, such
  as the GPL header;
- low-information shingles with 3 or fewer distinct words, or numeric
  shingles with 4 or fewer distinct values. Without this filter the ranking
  was dominated by 0/1 image masks.

Pairs were ranked by the number of shared shingles, with Jaccard similarity
and the longest shared run recorded. Every shared run was then classed as
prose or code. A run counts as distinctive when no more than 3 Octave files
contain it.

**Numeric vectors.** Bracketed literal arrays in Octave (all files, `%!`
lines included) and in mylab's added lines were cut into windows of 6
consecutive numbers. Windows that are arithmetic progressions, have fewer
than 4 distinct values or have fewer than 2 numbers with 3 or more
significant digits were dropped. The rest were matched exactly.

**Identifier overlap.** This is a signal of line-by-line translation into
Python. The scan used the code identifiers of each Octave `.m` file that are
at least 6 characters long, occur in no more than 2 Octave files and are not
Octave function names. It counted how many of them appear in the added lines
of each `mylab/**/*.py` file.

### 2. Marker grep

The added lines were searched for: `Copyright`/`(c) <year>`, `MathWorks`,
`GPL`/`GNU General Public`, `Octave`, `$Revision`/`$Id`, `%#ok`/`%#codegen`,
`mathworks.com`, and "adapted from", "ported from", "based on",
"translated from", "derived from", "copied from", "taken from" and
"port of". Each hit was read and classified.

### 3. Comparison with MathWorks documentation

The 2,144 help entries were ranked by phrases typical of reference-manual
style ("returns", "specified as", "name-value", "one or more", "for example",
"you can" and so on). The house style turned out to avoid these: the highest
score was 4. The comparison used 20 entries:

- the 15 highest-ranked or longest entries: `detectImportOptions`,
  `inputdlg`, `fread`, `ode45`, `ode15s`, `readmatrix`, `regionprops`,
  `runtests`, `impulse`, `gmres`, `edge`, `VideoReader`, `inputParser`,
  `kalman`, `grouptransform`;
- the newest modules: `fit`, `griddedInterpolant`, `delaunayTriangulation`;
- the two entries that Octave-matched definitional phrases: `condeig` and
  `betacdf`.

Their MathWorks reference pages were downloaded from mathworks.com/help (all
20 fetched) and compared with each mylab description using 6-word shingles.

### 4. Hand review

- The top 30 or so similarity pairs, the distinctive prose runs and the
  distinctive code runs.
- The newest modules: `mylab/builtins/curvefit.py`, `gridinterp.py`,
  `triangulation.py` and `_mat73.py`.
- The goodness-of-fit tests in `mylab/builtins/stats_tests.py` (`jbtest`,
  `lillietest`, `adtest`). These are the places where MATLAB and Octave use
  tabulated critical values.
- Tests whose expected values are marked "Octave reference".
- The data files added since 1.0.0.

## Results

### Octave similarity scan

| | count |
|---|---|
| mylab units × Octave units with at least 1 shared 8-word shingle | 1,733 pairs |
| with ≥ 3 / ≥ 5 / ≥ 10 shared shingles | 345 / 99 / 10 |
| largest number of shared shingles in one pair | 15 |
| distinctive prose runs (shared with ≤ 3 Octave files) | 125 |
| longest distinctive prose run | 14 words |
| longest distinctive code run | 24 tokens |

Before the low-information filter there were 4,424 pairs, with a maximum of
54 shared shingles. Those top pairs were all 0/1 matrices, such as
cellular-automaton and flood-fill grids and image masks.

**Top matches and what they are:**

| shared | mylab | Octave | what it is |
|---|---|---|---|
| 15 | `validation/matlab/checks.json`, `tests/probes/heldout/h2_oop_numerics.txt` | `m/ode/ode15s.m` (help and `%!`) | The Robertson stiff ODE (Robertson 1966), a standard test problem. The mylab check is written differently (`[0 40]`, a sum-to-one check) |
| 14 | `tests/conformance/golden.json` | image `imfilter.m` | the integers 4…24 |
| 13 | `checks.json` | `data.cc-tst` | `sum([1 2; 3 4])` = `[4 6]` |
| 11 | `mylab/help_data.py`, VS Code `functions.json` | `polyfit.m` help | the call forms `p = polyfit(x, y, n)`, `[p, S] = …`, `[p, S, mu] = …` (signatures: functional) |
| 10 | `helpdocs/linalg_more.py: lscov` | `lscov.m` help | the call forms of `lscov` |
| 10 | `builtins/plot_objects.py` | `gui/uisetfont.m` | the list of TeX Greek letter names |
| 9 | `helpdocs/sysmore.py: randi` | `randi.m` help | the call forms of `randi` |
| 9 | `builtins/plotting.py` | statistics `bar3.m` | the colour-name table `red r green g …` |
| 7 | `helpdocs/stats_dist.py: betacdf` | statistics `betacdf.m` help | "cumulative distribution function of the beta distribution with shape parameters a and b" |
| 5 | `helpdocs/signal_more.py: ellip` | signal `ellip.m` help | "with Rp dB of passband ripple and Rs dB of stopband attenuation" (standard filter-design wording) |

The remaining distinctive prose runs (8 to 12 words) fall into two groups:

- error-message fragments such as "x and y must have the same number of
  rows" and "must be a cell array of character vectors or a string array";
- standard terminology such as "of the normal distribution with mean mu and
  standard deviation sigma", "the 1-norm condition number of a square
  matrix", "reduced row echelon form of a symbolic matrix" and "negative
  log-likelihood of the negative binomial distribution".

None of them extends past one clause. Read in full, the mylab entries are
short, in the project's own style, and structured differently from Octave's
texinfo help.

The remaining code runs are coincidental short sequences:

- integer-class name lists;
- MATLAB datestr format names and month names;
- the `pdist` metric names;
- trivial test inputs such as `[1 2; 3 4]` and `1:6`;
- standard problems: van der Pol, and the integral of `exp(-x.^2)` equal to
  `sqrt(pi)`;
- documented output headings, such as the `kruskalwallis` table columns and
  the Nyquist plot labels.

### Numeric test vectors

There were 24,705 distinct Octave windows and 1,123 mylab windows. **46
windows are shared, from 5 sources, all published mathematics:**

- `mylab/blocks/engine.py`: the Dormand-Prince 5(4) Butcher tableau (Dormand
  and Prince 1980).
- `mylab/builtins/_quad.py`: Gauss-Kronrod 7-15 nodes and weights
  (QUADPACK).
- `datasets/matlab-tasks/tasks/numerical_methods/gauss_legendre_rule/test.m`:
  Gauss-Legendre weights.
- `mylab/builtins/linalg_more.py`: the Rosser test matrix (Rosser et al.
  1951).
- `tests/test_conformance_numeric.py`: `invhilb(3)`, a mathematical fact.

### Identifier overlap

There were 8,348 rare Octave identifiers and 2,146 file pairs with any
overlap. The largest overlap is between `_gallery.py` and `gallery.m`: 35
identifiers, all of them the documented `gallery` matrix names (`cauchy`,
`chebspec`, `clement`, …). The next are 6 or fewer, and all are functional
names:

- documented output-struct fields of `stepwisefit` (`SSresid`, `SStotal`,
  `wasnan`) and of `glmfit` (`residp`, `residd`, `estdisp`);
- DOM method names (`appendChild` and similar);
- graphics property names.

There is no sign of a translated Octave implementation.

### Octave reference values in tests

`tests/test_stats_*.py` use about 35 values marked "Octave reference". The
headers say these were obtained by running Octave's statistics package as a
black box. A sample of 17 of these numbers was searched for in the Octave
sources. 14 do not occur anywhere. The other 3 are short coincidences:

- `0.4300` is the KS critical value for n = 9 (`0.43001`, Miller 1956), a
  published table entry;
- `0.0535` and `0.3155` occur only in unrelated control and image files.

The goodness-of-fit tests in mylab compute their p-values in two ways:

- by simulation (`jbtest`, `lillietest` for non-normal distributions);
- from published approximations (`adtest`: Marsaglia and Marsaglia 2004,
  D'Agostino and Stephens 1986; `lillietest` normal).

They do not use tables copied from MATLAB or Octave.

### Marker grep (added lines only)

| marker | hits | classification |
|---|---|---|
| Copyright | 4 lines in 2 files | `editors/vscode/LICENSE` is the project's own MIT licence. In `plot_objects.py`, `copyright` is a TeX symbol name. Fine. |
| GPL | 1 | `datasets/matlab-corpus/README.md`: "repositories whose license is unknown or GPL-family are excluded". Fine. |
| MathWorks | 21 lines in 12 files | Trademark notices and "not affiliated" disclaimers; VS Code text about working alongside the MathWorks extension; the conformance README stating that nothing was copied; two corpus manifest entries naming MathWorks-owned GitHub repositories. Their code is not stored in this repository; see Data. Fine. |
| mathworks.com URL | 1 | `validation/README.md` links to MATLAB Online for the user's own validation run. Fine. |
| `$Revision` / `$Id` | 0 | |
| `%#ok` and other lint pragmas | 46 lines in 27 files | All in mylab's own `.m` task solutions and `validation/matlab/run_mylab_validation.m`. They are normal MATLAB comments; none is inside a Python string. Fine. |
| derivation phrases | 48 lines in 24 files | All are "class derived from …", "format taken from the extension", "port of …" in the sense of block ports, and similar wording. One is the `_ode_steppers.py` note "coefficients … taken from SciPy's solver classes": published Butcher tableaux, from BSD-licensed SciPy. Fine. |
| Octave | 642 lines in 229 files | Mostly the conformance harness (`overrides.json` with 222 lines, `run.py`, baselines), black-box comparisons, and Octave-compatible language features: `printf`, `endfunction`, `do … until`, `pkg load`, `Octave:` error IDs. No hit says that code or tests were taken from Octave. Fine. |

### MathWorks documentation sample (20 entries)

| entry | mylab 6-grams | shared with the MathWorks page | longest shared text |
|---|---|---|---|
| `condeig` | 20 | 4 (20%) | "the reciprocals of the cosines of the angles between" (rewritten; see below) |
| `betacdf` | 30 | 3 (10%) | "beta distribution with shape parameters a and b" |
| `grouptransform` | 78 | 2 (2.6%) | the method-name list `zscore norm meancenter rescale meanfill linearfill` |
| `griddedInterpolant` | 94 | 2 (2.1%) | the method-name list `linear nearest next previous pchip cubic` |
| `gmres`, `runtests`, `fread` | 125 to 171 | 2 each (1.2 to 1.6%) | "or a function handle that returns"; the table column names `Name Passed Failed Incomplete Duration Details`; "'l' or 'ieee-le' little endian" |
| `ode45`, `ode15s` | 134 / 176 | 1 each | "from tspan(1) to tspan(end)" |
| `detectImportOptions`, `inputdlg`, `readmatrix`, `regionprops`, `impulse`, `edge`, `VideoReader`, `inputParser`, `kalman`, `fit`, `delaunayTriangulation` | 68 to 165 | 0 | none |

Apart from `condeig`, every overlap is an option or field name, a documented
output heading or a mathematical term. The wording of the sampled entries is
independent.

### The newest modules

- **`curvefit.py`** (2,072 lines). Linear models are solved with
  `numpy.linalg.lstsq`. Nonlinear models use SciPy `least_squares`, with
  starting points chosen by the module's own grid searches (`_rate_grid`,
  `_greedy_start`). Smoothing splines come from SciPy. The comments describe
  the module's own design. There is no Octave counterpart to copy from,
  since Octave has no `fit`. Nothing suspicious.
- **`gridinterp.py`** (461 lines). Tensor-product interpolation:
  `CubicSpline`/`PchipInterpolator` and mylab's `makima` for the piecewise
  polynomials, and Keys cubic convolution (1981, published) for `'cubic'`.
  It uses `MATLAB:griddedInterpolant:*` error identifiers (functional).
- **`triangulation.py`** (924 lines). SciPy/Qhull Delaunay; constraints are
  recovered by edge flips followed by Lawson flips (textbook algorithms).
  The structure has no Octave counterpart.
- **`_mat73.py`** (367 lines, new since 1.0.0). It writes the HDF5 layout of
  v7.3 MAT-files through h5py, and its docstring lists the public knowledge
  it relies on (`MATLAB_class`, `#refs#`, the 512-byte user block). This is
  format compatibility. Nothing suspicious.

### Data

- **US census counts** (`CDATE`, `POP`, 1790 to 1990) in `tests/test_fit.py`.
  These are US Census Bureau figures and in the public domain. MATLAB ships
  the same numbers as `census.mat`; see below for how the test used them.
- **`datasets/matlab-corpus`** holds metadata only. It lists repository,
  pinned commit, SPDX licence, path, features and call counts. The `.m`
  sources are not stored; `tools/fetch.py` downloads them on demand. The
  README states that GPL-family and unknown-licence repositories are
  excluded. The manifest's `description` fields are copied from GitHub
  repository descriptions. These are one-line facts, not a provenance
  concern.
- **`datasets/matlab-tasks`, `examples/assignment-intro`, `tests/fixtures`**
  contain small hand-made CSV and text files (scores, rainfall, quoted
  fields, "the quick brown fox"). None comes from a known MATLAB sample
  dataset. A grep for `fisheriris`, `carsmall`, `carbig`, `hospital`,
  `patients`, `count.dat`, `hald`, `sunspot`, `cameraman` and similar names
  found only the census test and a note that MATLAB's sample pictures are
  not shipped.

## Rewritten

1. **`mylab/helpdocs/linalg_more.py`, `condeig`.**
   - Before: "Condition numbers of the eigenvalues of A: the reciprocals of
     the cosines of the angles between left and right eigenvectors. Large
     values mean sensitive eigenvalues."
   - After: an explanation in our own words, built around the formula
     `1/|y'*x|` for unit left and right eigenvectors.
   - Why: the 9-word phrase is the textbook definition, but it is also the
     wording of MathWorks' `condeig` page. It was rewritten as a precaution.
2. **`tests/test_fit.py`.**
   - Before: the module docstring said the census data "and the quoted
     bounds of its quadratic fit are those of MATLAB's fit documentation".
     The display test asserted three literal lines described as "the values
     printed in MATLAB's documentation for this fit".
   - After: the docstring cites the data's actual source (US Census Bureau,
     public domain). The test builds the three expected lines from the
     coefficients and 95% bounds that it already computes with NumPy/SciPy,
     formatted as the documented `%.4g` layout.
   - Strength: the computed strings were checked to be byte-for-byte equal
     to the old literals, so the test is exactly as strict as before.
     `tests/test_fit.py` and `tests/test_linalg_more.py` pass (42 tests), as
     do all help-related tests (2,802 selected by `-k "help or condeig"`).
     `ruff check .` passes.
3. **`tests/test_triangulation.py`** (done after this review, in the same
   release).
   - Before: the 2-D fixture was the six-point, four-triangle example of
     MathWorks' `triangulation` reference page, which CLEANROOM.md does not
     allow in tests.
   - After: an original mesh, an open fan of four triangles around one
     vertex. The expected edges, neighbours, attachments, barycentric
     coordinates, point locations and nearest neighbours were worked out by
     hand for it; incentres and circumcentres are computed in the test with
     NumPy from their definitions. Where MATLAB does not define an order
     (the start of the boundary loop, the order of attached triangles), the
     test checks the property instead: a closed counterclockwise loop with
     exactly the boundary edges, or the set of triangles.
   - All 27 tests pass.

## Needs a decision

1. **`tests/test_fit.py` still uses the same scenario as MathWorks' `fit`
   example:** a `poly2` fit of census data. The data is public domain, and
   the call `fit(cdate, pop, 'poly2')` is a one-line functional call. After
   the rewrite, no expected value depends on MathWorks text. If you want no
   overlap at all, swap in another dataset; the four `census` tests compute
   everything from the data, so only the data needs to change.
2. **Verbatim MATLAB runtime error messages.** For example,
   `gridinterp.py` raises "Interpolation requires at least two sample points
   in each dimension." This review treats short runtime messages, like error
   identifiers, as behaviour that programs and users may match on, not as
   documentation prose. Please confirm that this is the project's policy,
   and consider adding a line about it to CLEANROOM.md.
3. **Standard-terminology phrases shared with Octave Forge help** were left
   as they are. Examples: the `betacdf` and `normcdf` definitions and the
   `ellip` "Rp dB of passband ripple" clause. They are single clauses of
   ordinary mathematical language inside entries that are otherwise
   independent.

(The review also found `editors/vscode/data/functions.json` out of date at
`8dee3c2`; it was regenerated in the release preparation, commit `7d8b89a`.)

## Limits of this review

- **Paraphrase.** An 8-word shingle scan finds verbatim and near-verbatim
  text. It cannot find a close paraphrase that changes a word every few
  words, or text that was reordered. Some help text is 1 to 3 sentences
  long, where any copying would be short anyway.
- **MathWorks documentation.** Only 20 of the 2,144 new help entries were
  compared with MathWorks pages. Docstrings in the Python code, the
  Markdown docs and the task prompts were not compared with MathWorks text
  at all. The style ranking used to pick the sample is a heuristic.
- **MathWorks source.** No MathWorks `.m` files were available, and none
  should be, so a port of MathWorks code could only be noticed through
  comments or style. None was noticed in the modules reviewed by hand.
- **Translation across languages.** A line-by-line port of an Octave `.m`
  function into Python shares few word sequences. The identifier scan and
  the sampled reading of `curvefit.py`, `gridinterp.py`, `triangulation.py`,
  `_mat73.py` and `stats_tests.py` lower this risk but cannot rule it out
  for the roughly 150 other new modules, which were not read in full.
- **Reference corpus.** Octave was compared as installed: Octave 10.3 `.m`
  files, extracted `.cc` tests, built-in docstrings, and the nine installed
  Forge packages. The comparison did not include:
  - Octave's C++ sources;
  - Forge packages that are not installed;
  - other GPL code (for example Scilab or GSL);
  - MATLAB Cody problems, for the task dataset.
- **Web and editor code.** The JavaScript in `web/` and `editors/vscode/`
  was only scanned against Octave and grepped for markers.
- **Process.** This review checks the result, not how it was produced. It
  cannot show whether an author looked at a MathWorks page or an Octave
  file while writing. Two test files say openly that they used
  documentation examples, so keeping that rule in front of contributors,
  including AI assistants, is worthwhile.
