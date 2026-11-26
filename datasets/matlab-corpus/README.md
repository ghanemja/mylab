# MATLAB Corpus

A reproducible, license-checked collection of real-world MATLAB code from
**200 open-source repositories: 29,114 `.m` files and about 4.1 million
lines**. Every repository is pinned to an exact commit. Each file comes with
metadata: its kind, the language features it uses, the functions it calls,
and whether mylab can parse it.

Built and maintained by the [mylab](../../README.md) project. It's for anyone
who builds MATLAB-language tools (interpreters, linters, translators, code
models) or studies how MATLAB is written in practice.

## What's in this folder

| File | What it is |
|---|---|
| `manifest.jsonl` | One line per repository: `repo`, `url`, `commit` (pinned SHA), `branch`, `license` (SPDX), `stars`, `description`, `pinned_on`. |
| `files.jsonl.gz` | One line per `.m` file. Fields are listed below. |
| `stats.json` | Totals, license mix, feature frequencies and mylab's parse rate. |
| `GAPS.md` | Functions the corpus calls that mylab doesn't provide yet, ranked by how many repositories use them. |
| `tools/` | Scripts to rebuild, download, index and package the corpus, and to run its code. |
| `regression.json`, `regression_expected.json` | The repositories whose tests and examples mylab runs unchanged, and which runs pass (see below). |

Fields in `files.jsonl.gz`:

- `repo`, `commit`, `license`, `path`, `bytes`, `lines`;
- `kind`: `script`, `function` or `classdef`;
- `features`: counts of language features used, such as `classdef`,
  `nested_functions`, `anonymous_functions`, `cell_arrays`, `struct_fields`,
  `string_literals`, `try_catch`, `tables`, `containers_map`, `plotting`,
  `file_io`, `ode_solvers`, `eval`, …;
- `calls`: the 50 most frequent function names the file calls (names
  assigned as variables in the file are excluded);
- `mylab_parse`: `ok`, or the parse error message.

The source code itself is not stored in this repository. You download it
with the tools, or build a redistributable archive (see below).

## Getting the code

```bash
pip install -e .                          # from the mylab repository root
cd datasets/matlab-corpus
python tools/fetch.py                     # .m files + license files at the pinned commits (~400 MB)
python tools/index.py                     # regenerate files.jsonl.gz, stats.json, GAPS.md
python tools/package.py                   # dist/matlab-corpus.tar.gz with licenses and NOTICE.md
```

`fetch.py` uses partial, sparse git clones, so only `.m` files and license
files are downloaded. Nothing in the corpus is ever executed by these tools.

### Block-diagram model files

Eight of the repositories also hold `.slx` model files: 82 files, 11 MB.
`models.jsonl` lists them (repository, pinned commit, path, license, size and
SHA-256). They are test data for the `.slx` reader of
[mylab Blocks](../../docs/BLOCKS.md) and, like the `.m` files, are not stored
in this repository:

```bash
python tools/fetch_models.py              # into ~/.cache/mylab-corpus-models (or $MYLAB_SLX_CACHE)
python tools/fetch.py                     # the .m files of the same repositories, into ~/.cache/mylab-corpus
python -m mylab.blocks.import_report ~/.cache/mylab-corpus-models --sources ~/.cache/mylab-corpus
```

`fetch_models.py` fetches only the listed files at the pinned commits and
checks each against its SHA-256. The report says, for every file, whether it
was imported fully, imported with unsupported blocks (named and counted) or
not at all, and whether the imported model then ran. With `--sources` the
folders of the model's repository go on the search path and its set-up
scripts are run when the model names variables that do not exist.
`tests/test_blocks_slx.py` runs the same survey, and sanity checks on models
with a known expected behaviour, when the files are in the cache
(`$MYLAB_SLX_CACHE`, and `$MYLAB_CORPUS_CACHE` for the `.m` files); it is
skipped otherwise.

Importing reads the files as data and executes nothing. Simulating an
imported model runs the code stored in it (callbacks, mask initialization,
function blocks) and the repository's scripts **in the mylab interpreter**,
with a time limit: none of it is run by Python or by the original product.

## Running real code: the regression suite

`regression.json` selects repositories whose tests and examples mylab runs
**unchanged**, from beginning to end: unit-test suites with known results
(matGeom's, run through mylab's own `runtests`), textbook demos (PRML),
toolbox examples and course scripts. `regression_expected.json` records which
runs pass. The code is fetched on demand and is not stored in mylab:

```bash
python tools/run_corpus.py fetch          # check out the selected repositories (once, ~1.7 GB)
python tools/run_corpus.py run            # fails if a run that passed before fails now
python tools/run_corpus.py run --only PRML/PRMLT --verbose
python tools/run_corpus.py survey REPO    # try every test/demo-looking script of a repository
```

or `MYLAB_CORPUS_REGRESSION=1 python -m pytest tests/test_corpus_regression.py`.
Each script runs in a fresh mylab process, in its own folder (or the folder a
run names), with the repository's folders on the path, figures drawn off
screen and a time limit. Only mylab executes the code.

The selection leaves out scripts that need a toolbox mylab does not have (for
example the Neural Network Toolbox's `newff`, the Robotics System Toolbox's
`se3`), compiled MEX files, data that is not in the repository, or someone at
the keyboard (`input`, `ginput`).

**Current selection: 25 repositories, 185 runs, all passing** (about 9 minutes
with three jobs). Two of the runs are unit-test suites with known results:
matGeom's 721 tests (707 pass; the run fails if fewer pass than now) and
RVC3's toolbox and book tests (69 pass).

| Repository | Runs passing | What it is |
|---|---|---|
| [Agarciafernandez/MTT](https://github.com/Agarciafernandez/MTT) | 4 of 4 | Implementation of several Bayesian multi-target tracking algorithms, i |
| [FeeLab/seqNMF](https://github.com/FeeLab/seqNMF) | 1 of 1 | An algorithm for unsupervised discovery of sequential structure |
| [HuangCongQing/Algorithms_MathModels](https://github.com/HuangCongQing/Algorithms_MathModels) | 20 of 20 | 【国赛】【美赛】数学建模相关算法 MATLAB实现（2018年初整理） |
| [MatthewPeterKelly/OptimTraj](https://github.com/MatthewPeterKelly/OptimTraj) | 13 of 13 | A trajectory optimization library for Matlab |
| [PRML/PRMLT](https://github.com/PRML/PRMLT) | 40 of 40 | Matlab code of machine learning algorithms in book PRML |
| [Pengskr/MPC](https://github.com/Pengskr/MPC) | 1 of 1 | 基于Matlab实现模型预测控制(MPC) |
| [ZitengWang/MASP](https://github.com/ZitengWang/MASP) | 2 of 2 | Microphone Array Speech Processing |
| [alecjacobson/gptoolbox](https://github.com/alecjacobson/gptoolbox) | 1 of 1 | Matlab toolbox for Geometry Processing. |
| [baddoo/piDMD](https://github.com/baddoo/piDMD) | 4 of 4 | MATLAB codes for physics-informed dynamic mode decomposition (piDMD) |
| [dlaptev/RobustPCA](https://github.com/dlaptev/RobustPCA) | 1 of 1 | Robust PCA implementation and examples (Matlab) |
| [hiroyuki-kasai/NMFLibrary](https://github.com/hiroyuki-kasai/NMFLibrary) | 5 of 5 | NMFLibrary: Non-negative Matrix Factorization (NMF) Library: Version 2 |
| [hiroyuki-kasai/SGDLibrary](https://github.com/hiroyuki-kasai/SGDLibrary) | 1 of 1 | MATLAB/Octave library for stochastic optimization algorithms: Version  |
| [ivaquero/book-control](https://github.com/ivaquero/book-control) | 2 of 2 | 简明控制理论教程 |
| [lis-epfl/swarmlab](https://github.com/lis-epfl/swarmlab) | 1 of 1 | SmarmLab: a versatile Matlab package for drone swarm simulation. |
| [masumhabib/PlotPub](https://github.com/masumhabib/PlotPub) | 12 of 12 | Publication quality plot in MATLAB.  |
| [mattools/matGeom](https://github.com/mattools/matGeom) | 43 of 43 | Matlab geometry toolbox for 2D/3D geometric computing |
| [mnakanishi/TRCA-SSVEP](https://github.com/mnakanishi/TRCA-SSVEP) | 2 of 2 | Task-related component analysis (TRCA)-based algorithm for detecting s |
| [morriswmz/doa-tools](https://github.com/morriswmz/doa-tools) | 8 of 8 | A set of MATLAB functions for direction-of-arrival (DOA) estimation in |
| [olzhas/rrt_toolbox](https://github.com/olzhas/rrt_toolbox) | 12 of 12 | RRT, RRT*, RRT*FN algorithms for MATLAB |
| [petercorke/RVC3-MATLAB](https://github.com/petercorke/RVC3-MATLAB) | 2 of 2 | New toolbox, data and examples for Robotics, Vision & Control: 3rd edi |
| [rasmusbergpalm/DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) | 2 of 2 | Matlab/Octave toolbox for deep learning. Includes Deep Belief Nets, St |
| [ravenxrz/Mathematical-Modeling](https://github.com/ravenxrz/Mathematical-Modeling) | 1 of 1 | 数学建模算法学习 |
| [s-kajita/IntroductionToHumanoidRobotics](https://github.com/s-kajita/IntroductionToHumanoidRobotics) | 1 of 1 | Matlab code for a Springer book "Introduction to Humanoid Robotics" |
| [swedishembedded/control](https://github.com/swedishembedded/control) | 2 of 2 | Embedded Firmware Control Systems Toolbox (Pure C and GNU Octave) |
| [trekhleb/machine-learning-octave](https://github.com/trekhleb/machine-learning-octave) | 4 of 4 | 🤖 MatLab/Octave examples of popular machine learning algorithms with c |


## How repositories were selected

- GitHub repositories whose main language is MATLAB, found through the GitHub
  search API.
- Not forks, with at least 60 stars, ranked by stars, up to 200 repositories.
- A license that GitHub detects as one of: MIT, BSD-2-Clause, BSD-3-Clause,
  Apache-2.0, ISC, 0BSD, Unlicense, CC0-1.0, Zlib or BSL-1.0. Repositories
  whose license is unknown or GPL-family are excluded.

Current mix: MIT 135, BSD-3-Clause 24, Apache-2.0 23, BSD-2-Clause 16,
CC0-1.0 2.

`tools/build_manifest.py` re-runs the selection. Existing pins are kept unless
you pass `--refresh`, so results stay comparable over time.

## Intended uses

- **Testing MATLAB-language implementations.** Measure the parse rate on real
  code and find unsupported syntax and functions. mylab uses it this way:
  `GAPS.md` is our compatibility to-do list.
- **Studying MATLAB usage:** which features and functions are common in
  practice.
- **Building tools:** linters, formatters, MATLAB-to-Python translators.
- **Training or evaluating code models,** provided you comply with each file's
  license. Keep the attribution: `NOTICE.md` and the license files are in the
  packaged archive.

## Licensing

- The metadata and tools in this folder are MIT licensed, like mylab.
- The code belongs to its authors and is under each repository's own license.
  Most are permissive, but all require attribution (CC0 does not). The
  packaged archive includes each repository's license file and a NOTICE
  listing every source. Repositories with no license file in the checkout are
  left out of the archive.
- License detection comes from GitHub and can be wrong. Individual files may
  carry different terms, such as third-party code vendored inside a
  repository. If you redistribute, check the files you rely on.

## Known limitations

- The selection favours popular repositories, which leans towards machine
  learning, image processing, robotics and signal processing.
- Some `.m` files are not MATLAB, e.g. Objective-C sources. A few files are
  intentionally broken (test fixtures).
- `calls` is a static approximation. Variables created by `load` or `eval`
  can appear as calls, and calls made through function handles stored in
  variables are missed.
- The code can contain personal data such as author names and email addresses
  in comments, as published by its authors.

## Current results for mylab

mylab parses **99.1%** of the files. Most of the remaining 0.9% are not
MATLAB, are broken source files, or use rare constructs (`?ClassName`
metaclass queries, overloaded `end`). See `stats.json` for the error
breakdown and `GAPS.md` for missing functions.
