# Roadmap

The goal is a free tool for people who need to run MATLAB-language code but
can't buy a license. It should be easy to start (nothing to install) and work
well with AI assistants.

## Done in 1.0

- **Core engine:**
  - lexer, parser and interpreter for the language;
  - N-D arrays, string arrays, cells, structs;
  - `classdef` classes, `containers.Map`, tables.
- **Library:** about 750 functions, including signal processing, statistics,
  optimization, ODE solvers and plotting. Every function has help text.
- **Ways to run it:**
  - the `mylab` command-line prompt and script runner;
  - a Python API (`mylab.api.Session`);
  - a browser app with no install (Pyodide/WebAssembly): editor, command
    window, workspace, plots, files stored in the browser, import/export;
  - an AI assistant in the browser app (Ollama, Anthropic or OpenAI-compatible
    providers, using your own key or a free local model);
  - an MCP server (`mylab-mcp`) for AI assistants;
  - a Jupyter kernel (`mylab-kernel`).
- **Speed:**
  - cached literals and parsed files;
  - fast paths for scalar code;
  - in-place updates of arrays that aren't shared.

## Next

- Object arrays, enumerations, events, and `@ClassName` folders.
- datetime/duration, categorical and timetable.
- Excel and image files, binary file I/O.
- `input()` in the browser (needs cross-origin isolation for a synchronous
  prompt).
- Faster loops: compile hot loops to bytecode, or optionally use Numba where
  it is available. The engine keeps the language front end separate from the
  numeric back end, so this won't change user code.
- Translations of the web app interface.

## Non-goals

- Simulink.
- Matching MATLAB's exact random-number streams or solver step sequences.
- Copying MATLAB's look, documentation or branding (see
  [CLEANROOM.md](../CLEANROOM.md)).
