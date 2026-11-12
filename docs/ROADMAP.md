# Roadmap

The goal is a free tool for people who need to run MATLAB-language code but
can't buy a license. It should be easy to start (nothing to install) and work
well with AI assistants.

## Done in 1.0

- **Core engine:**
  - lexer, parser and interpreter for the language;
  - N-D arrays, string arrays, cells, structs;
  - `classdef` classes, `containers.Map`, tables.
- **Library:** about 1,100 functions, including image and signal processing,
  statistics, optimization, sparse solvers, ODE solvers and plotting. Every function has help text.
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
  - a loop compiler for simple numeric loops (15–55× faster);
  - cached literals and parsed files, fast paths for scalar code;
  - in-place updates of arrays that aren't shared.
- **Data:** sparse matrices, dates and times, images, binary files,
  animations.
- **Teaching:** share links, step-through, published reports, assignments
  and the `mylab-grade` autograder.
- **Tools:** debugger, VS Code extension, conformance suite against GNU
  Octave.
- **Datasets:** MATLAB Tasks (157 tasks) and MATLAB Corpus (200 repositories).

## Next

- Object arrays, enumerations, events, and `@ClassName` folders.
- categorical and timetable; date tick labels on plots.
- Excel files and audio.
- Faster function calls, and compiling loops that call user functions.
- Real dialogs and `input()` in the browser app; a service worker for full
  offline use.
- Translations of the web app interface.
- More conformance cases, and checks against real MATLAB from volunteers.

## Non-goals

- Simulink.
- Matching MATLAB's exact random-number streams or solver step sequences.
- Copying MATLAB's look, documentation or branding (see
  [CLEANROOM.md](../CLEANROOM.md)).
