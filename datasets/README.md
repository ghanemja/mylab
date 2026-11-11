# Datasets

The mylab project publishes two open datasets for anyone working with
MATLAB-language code:

| Dataset | What it is | Size | License |
|---|---|---|---|
| [MATLAB Tasks](matlab-tasks/) | Original programming tasks with reference solutions and tests. Use it to benchmark code models, as graded practice for teaching, or as a conformance suite for MATLAB-language engines. | 157 tasks in 18 categories | MIT |
| [MATLAB Corpus](matlab-corpus/) | Real-world MATLAB code from 200 permissively licensed repositories, pinned to commits, with per-file features, function usage and parse status. Use it to test parsers and tools, study usage, or train models with attribution. | 29,114 files, 4.1M lines | Metadata MIT; code under each repository's license |

Both are refreshed and checked by the weekly `Datasets` workflow
(`.github/workflows/datasets.yml`).
