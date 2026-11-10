# Roadmap

The goal is a free tool for people who need to run MATLAB-language code but
can't buy a license. It should be easy to start (ideally nothing to install)
and work well with AI assistants.

## Phase 1: core engine (v0.1, done)

- Lexer, parser and interpreter for the core language.
- 450+ built-in functions: math, linear algebra, strings, cells/structs,
  solvers and plotting.
- `mylab` command-line prompt and script runner.
- Test suite, runnable examples and clean-room contribution rules.

## Phase 2: no-install browser app

- Run the Python engine in the browser with
  [Pyodide](https://pyodide.org) (WebAssembly). Nothing needs to run on a
  server, so hosting is free.
- Editor with syntax highlighting, command window, workspace viewer, plot
  pane and file browser. Files are stored in the browser (IndexedDB), with
  export to and import from `.m` files.
- Works on Chromebooks and low-end laptops.

## Phase 3: AI integration

- An **MCP server** (`mylab-mcp`) so that AI assistants (Claude Desktop,
  Cursor, …) can run code, read variables and return plots.
- An **assistant panel** in the web app that explains errors, writes and
  fixes code, and translates between mylab code and Python. It will work with
  free local models (Ollama) and with the user's own API key. No paid account
  will be required.
- `help` text written for both humans and AI tools.

## Phase 4: breadth and speed

- A Jupyter kernel.
- N-dimensional arrays, string arrays, `containers.Map`, `table`, and basic
  `classdef`.
- More toolbox-style functions, chosen by user requests: signal processing,
  statistics, optimization, control.
- Speed:
  - Cache the parsed syntax tree.
  - Add fast paths for scalar code.
  - Optionally, compile hot loops with Numba or another faster backend. The
    engine keeps the language front end separate from the numeric backend,
    so a faster backend can be added later without changing user code.

## Non-goals

- Simulink.
- Matching MATLAB's exact random-number streams or solver step sequences.
- Copying MATLAB's look, documentation or branding (see
  [CLEANROOM.md](../CLEANROOM.md)).
