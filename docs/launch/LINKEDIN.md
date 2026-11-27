# Launch post (LinkedIn)

Attach `docs/images/demo.gif`, and `docs/images/bring-your-code.gif` as a
second image (or MP4 copies of them if LinkedIn won't animate the GIFs).

## Post

I got tired of AI agents and MATLAB not working reliably together, so I built
my own.

MATLAB is expensive, and hard for my agents to work with in a timely fashion.
A lot of people I work with (students, researchers, engineers) think in MATLAB
syntax and don't want to switch. Meanwhile, AI agents are excellent at Python.
Getting them to reliably run, check and fix MATLAB code was a constant
struggle.

So I wrote mylab: a free, open-source environment that runs MATLAB-syntax
code, with a Python engine underneath.

For people, nothing changes. Write the same .m code you already know. Open an
existing project folder and run it unchanged. Open a Simulink .slx model and
simulate it.

For agents, everything gets easier. The engine is NumPy, SciPy and
Matplotlib, and it plugs straight into Claude, Cursor and other assistants
through an MCP server. Agents can run, inspect and fix code in one loop.

What's in it:
• Runs in the browser with nothing to install (WebAssembly), or pip install mylab-lang
• A built-in AI assistant that explains errors and fixes code, with free local models or your own API key
• Live collaboration: send a link and edit the same files together, end-to-end encrypted, no account
• 2,800+ functions: ode45, fit, fft, tables, classdef, image processing, computer vision, C/C++ MEX files
• Block diagrams with state charts and physical networks, plus .slx/.mdl import
• Vectorized code at NumPy speed, and numeric loops compiled automatically (20–50× faster than interpreting them)
• A Jupyter kernel, a VS Code extension, a Python API, and an autograder for classes

Why not GNU Octave? It's the established free option and a good one, so I
measured against it. On 469 checks written from MATLAB's public
documentation that neither program had seen, mylab passed 379 and Octave 10.3
passed 333 (mylab now passes all 469). On 30 speed tests mylab was never
slower. And Octave doesn't open Simulink models.

It runs about 7,700 tests on Linux, Windows and macOS, and is tested in the
engines of Chrome, Firefox and Safari.

Try it: https://ghanemja.github.io/mylab/
Code (MIT licence): https://github.com/ghanemja/mylab

If you teach or work in MATLAB, I'd love to hear what breaks.

#opensource #matlab #python #ai #engineering #education

## Where the numbers come from

- 2,800+ functions: `docs/FUNCTIONS.md` (2,859).
- 20–50×: three loops timed with `MYLAB_JIT=0` and `=1` (a sum, a recurrence,
  a while/if loop) ran 40–52× faster compiled.
- Octave: `tests/probes/baselines/heldout_before.json` (mylab 379 of 469,
  before the gap fixes), `tests/probes/baselines/octave_10_3_and_8_4.json`
  (Octave 10.3 with its packages: 333 of 469) and
  `benchmarks/octave_compare/` (30 speed cases).
- About 7,700 tests and three browser engines: the README's "How it is
  tested".

## Before posting

1. Make the repository public.
2. Publish to PyPI, so `pip install mylab-lang` works (`docs/RELEASING.md`).
3. Turn on GitHub Pages, so the web app link works, and open it once in a
   private window: run the demo and start a live session with a second
   browser.
4. Wording: write "MATLAB", "MATLAB-syntax" or "MATLAB-compatible". Don't use
   the MATLAB or Simulink logos, and don't suggest MathWorks is involved.
   MATLAB and Simulink are registered trademarks of The MathWorks, Inc.

## First comment (optional)

mylab is a clean-room project: no MathWorks or Octave code was copied, and
MATLAB was never used to test it. Known differences are listed in
docs/COMPATIBILITY.md, and the "Report a difference" form on GitHub is the
fastest way to get one fixed.
