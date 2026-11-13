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
| `tools/` | Scripts to rebuild, download, index and package the corpus. |

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
python -m mylab.blocks.import_report ~/.cache/mylab-corpus-models
```

`fetch_models.py` fetches only the listed files at the pinned commits and
checks each against its SHA-256. The report says, for every file, whether it
was imported fully, imported with unsupported blocks (named and counted) or
not at all, and whether the imported model then ran. `tests/test_blocks_slx.py`
runs the same survey when the files are in the cache and is skipped otherwise.
The files are read as data and nothing in them is executed.

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
