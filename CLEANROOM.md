# Clean-room contribution rules

mylab aims to run programs written in the MATLAB language. That is
legitimate: courts in the US (*Google v. Oracle*, 2021) and the EU
(*SAS Institute v. World Programming*, 2012) have held that reimplementing a
programming language, or the functional interface of an API, is lawful.

What *is* protected is the expression in MathWorks' own materials: their
documentation prose, their example code, their source files and their
trademarks. To keep mylab safe for everyone who uses and distributes it,
every contribution must follow these rules.

## You may

- Implement a function so that it **behaves** the way MATLAB code expects it
  to: the same name, calling forms, inputs and outputs, edge cases and error
  conditions.
- Learn *what* a function is supposed to do from any public source. That
  includes reference pages, textbooks, course notes, Stack Overflow answers,
  and observing the behaviour of MATLAB or GNU Octave as a black box.
- Use published math (papers, textbooks, Wikipedia) and permissively licensed
  libraries (NumPy, SciPy, …) to implement the behaviour.
- Write the function signature and a short description of the behaviour
  **in your own words** before you write code. Optionally, add these notes to
  the pull request.

## You must not

- **Copy documentation text.** This includes close paraphrases and
  "translated" versions. Write all help text, docstrings and docs in mylab
  from scratch.
- **Copy example code** from MathWorks documentation into tests, examples or
  docs. Write your own examples.
- **Copy, port or translate MathWorks source code.** Many MATLAB functions
  ship as readable `.m` files. Do not open them while you work on mylab. If
  you have already read one, have someone else implement that function.
- **Copy GNU Octave source code or tests.** Octave is GPL-licensed and mylab
  is MIT-licensed. You may use Octave as a black box to compare results.
- **Decompile or reverse engineer** MathWorks binaries, or break the terms of
  any license you have agreed to.
- **Use MathWorks trademarks** as a name or logo. Writing "compatible with
  MATLAB® code" to describe the project is fine.
- **Generate code with an AI tool from pasted MathWorks documentation.** If
  you use an AI assistant, give it your own behaviour spec, not their pages.
  Review its output for anything that looks copied.

## When in doubt

Open an issue and ask before you write the code. It is much easier to plan a
clean implementation than to remove a tainted one later.
