# Contributing to mylab

Thank you for helping! mylab exists so that anyone can run MATLAB-language
code for free, and every function, fix and bug report moves that forward.

## Before you start

**Read [CLEANROOM.md](CLEANROOM.md).** In short: implement behaviour from your
own understanding, and never copy MathWorks (or GNU Octave) documentation,
examples or code. This protects the project and everyone who uses it.

## Set up

```bash
git clone https://github.com/ghanemja/mylab
cd mylab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Reporting a bug

Open an issue with:

- a **short `.m` snippet** that shows the problem;
- what mylab printed;
- what you expected, and why. "MATLAB prints X" is fine, but please type the
  expected output yourself rather than pasting from MathWorks documentation.

## Adding a built-in function

1. Pick the module in `mylab/builtins/` for its topic, or add a new module and
   import it in `mylab/builtins/__init__.py`.
2. Write the function:

   ```python
   @builtin("myfunc")
   def _myfunc(interp, args, nargout):
       x = fnum(args[0])               # numeric argument as a float array
       return x * 2                    # one output; return a tuple for several
   ```

   Values are 2-D or N-D numpy arrays (char arrays use dtype `<U1`),
   `MString`/`StringArray`, `Cell`, `Struct`, `FunctionHandle`, or objects
   (see `mylab/values.py`). Raise `MatlabError("...")` for user errors.
3. Add a help entry in `mylab/help_data.py`, written in your own words. A test
   checks that every built-in has one.
4. Add tests in `tests/` (`m.run(...)`, `m.num(...)`, `m.str(...)` from
   `tests/conftest.py`).
5. Regenerate the function list: `python scripts/gen_function_list.py`.

## Working on the browser app

The app in `web/` is plain HTML, CSS and JavaScript with no build step.

```bash
python scripts/build_web.py --serve      # http://localhost:8000
node scripts/test_web.mjs                # end-to-end test (needs Playwright + Chromium)
```

## Pull requests

- Keep changes focused, and add tests.
- `pytest -q` and `ruff check .` must pass.
- Describe behaviour changes in `CHANGELOG.md`.
