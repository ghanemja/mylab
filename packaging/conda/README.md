# conda recipe

`meta.yaml` is a recipe for [conda-forge](https://conda-forge.org). Once it is
accepted there, anyone can install mylab with:

```
conda install -c conda-forge mylab-lang
```

The recipe builds from the source package on PyPI, so **publish to PyPI
first**. Step-by-step instructions are in
[docs/RELEASING.md](../../docs/RELEASING.md#6-conda-forge-optional-about-30-minutes-then-waiting).

Two values change with each release: `version` and `sha256`. Each GitHub
release of mylab includes `conda-meta.yaml`, a copy of this recipe with both
filled in.

To try the recipe on your own computer (needs conda):

```
conda install -n base conda-build
conda build packaging/conda
```

The optional extras of the pip package (`mylab-lang[mcp]`, `mylab-lang[jupyter]`)
are not part of the conda package. Users who want them install `mcp` or
`ipykernel` next to it.
