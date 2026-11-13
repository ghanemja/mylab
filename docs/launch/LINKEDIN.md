# Launch post (LinkedIn)

Attach `docs/images/demo.gif` (or `docs/images/web-app.png` if the GIF is
rejected). Replace the two `<…>` links before posting.

## Post

MATLAB costs more than many students, hobbyists and small labs can pay. So I
built a free alternative and I'm releasing it today.

mylab runs MATLAB-language code. It is open source (MIT) and written from
scratch in Python.

What you get:

→ It runs in the browser. Open a link and start typing. Nothing to install,
and your code stays on your machine.
→ About 1,390 functions: linear algebra, plotting, signal and image
processing, statistics, optimization, ODE solvers, tables, sparse matrices.
→ An AI assistant that can write, run and fix code. Use a free local model
or your own API key. There is also an MCP server, so tools like Claude can
run code directly.
→ Built for teaching: share code as a link, step through a program line by
line, and auto-grade assignments (Gradescope export included).
→ Also works from the terminal, Jupyter and VS Code.

Two datasets come with it, for anyone building tools for this language:
• MATLAB Tasks: 157 original programming tasks with checked answers.
• MATLAB Corpus: an index of 200 permissively licensed repositories,
29,114 files. mylab parses 99.1% of them.

What it is not: it doesn't cover everything MATLAB does. There is no
Simulink and no GUI builder, and many toolboxes are missing. Results are
checked against GNU Octave (5,377 automated checks, all passing), not
against MATLAB itself. If your code doesn't
run, please open an issue; that is the most useful thing you can do.

Try it: <web app link>
Code: <GitHub link>

If you teach a course that uses MATLAB, or know students who can't get a
license, I'd love your feedback.

#opensource #matlab #python #engineering #education #ai

## Shorter version

I built a free, open-source tool that runs MATLAB-language code, for people
who can't buy a license.

• Runs in the browser, nothing to install
• About 1,390 functions, plots included
• AI assistant built in (local model or your own key)
• Share links, step-through and autograding for classes

It doesn't cover everything (no Simulink, many toolboxes missing), and it is
tested against GNU Octave rather than MATLAB. Bug reports are welcome.

Try it: <web app link> · Code: <GitHub link>

## Before posting

1. Make the repository public.
2. Turn on GitHub Pages (see `docs/RELEASING.md`) so the web app link works.
3. Open the web app link in a private window and run the demo once.
4. Wording: say "runs MATLAB-language code" or "MATLAB-compatible". Don't
   use the MATLAB logo, and don't suggest MathWorks is involved. MATLAB is a
   registered trademark of The MathWorks, Inc.

## First comment (optional)

Details for the curious: mylab is a clean-room project. No MathWorks or
Octave code was copied. The loop compiler makes simple numeric loops 15–55×
faster than the plain interpreter. Known gaps are listed in
docs/COMPATIBILITY.md.
