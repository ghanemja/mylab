"""Command-line entry point: an interactive prompt and a script runner."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .errors import MatlabError
from .interpreter import Interpreter
from .lexer import tokenize

BANNER = f"""mylab {__version__} - free MATLAB-language environment (NumPy backend)
Type 'help' for help, 'exit' to quit. Statements without ';' print their result.
"""

BLOCK_OPEN = {"if", "for", "parfor", "while", "switch", "try", "function"}


def format_error(e: MatlabError) -> str:
    where = ""
    line = getattr(e, "line", None)
    file = getattr(e, "file", None)
    if file and line:
        where = f"\nError in {os.path.basename(file)} (line {line})"
    return f"Error: {e.message}{where}\n"


def needs_more_input(code: str) -> bool:
    """True if ``code`` has unclosed blocks or brackets (prompt for more lines)."""
    if code.rstrip().endswith("..."):
        return True
    try:
        tokens = tokenize(code)
    except MatlabError as e:
        return "unbalanced" in e.message and code.count("[") + code.count("(") + code.count("{") > \
            code.count("]") + code.count(")") + code.count("}") or "unterminated block comment" in e.message
    depth = 0
    for i, t in enumerate(tokens):
        if t.kind == "KW":
            if t.value in BLOCK_OPEN:
                depth += 1
            elif t.value == "end":
                # 'end' inside an index expression is not a block terminator, but the
                # lexer keeps it a keyword; the parser handles that, so approximate here.
                prev = tokens[i - 1] if i else None
                if prev is not None and prev.kind == "OP" and prev.value in ("(", ",", ":", "+", "-", "*", "/"):
                    continue
                depth -= 1
    # A file of functions without 'end' never "closes"; treat a blank line as the end.
    return depth > 0 and not code.endswith("\n\n")


def repl(interp: Interpreter) -> int:
    try:
        import readline  # noqa: F401  (line editing and history where available)
    except ImportError:  # pragma: no cover
        pass
    from .builtins.plotting import _plt
    sys.stdout.write(BANNER)
    interp.state["clc_hook"] = lambda: sys.stdout.write("\033[2J\033[H")
    interactive_plots = False
    while True:
        try:
            line = input(">> ")
        except EOFError:
            sys.stdout.write("\n")
            return 0
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            continue
        code = line
        while needs_more_input(code):
            try:
                code += "\n" + input("   ")
            except EOFError:
                break
        if not code.strip():
            continue
        try:
            interp.execute(code)
        except MatlabError as e:
            sys.stdout.write(format_error(e))
        except KeyboardInterrupt:
            sys.stdout.write("\nOperation interrupted.\n")
        except SystemExit as e:
            return int(e.code or 0)
        st = interp.state.get("plot")
        if st and st.figures and not interactive_plots:
            plt = _plt()
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                plt.ion()
                plt.show(block=False)
                interactive_plots = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mylab",
        description="Run MATLAB-language code for free. With no arguments, starts an interactive prompt.",
    )
    parser.add_argument("script", nargs="?", help="a .m script or function file to run")
    parser.add_argument("-e", "--eval", metavar="CODE", help="run CODE and exit")
    parser.add_argument("--no-plots", action="store_true", help="do not open plot windows")
    parser.add_argument("--version", action="version", version=f"mylab {__version__}")
    args = parser.parse_args(argv)
    if args.no_plots:
        os.environ["MPLBACKEND"] = "Agg"

    interp = Interpreter()
    if args.eval is None and args.script is None:
        return repl(interp)
    try:
        if args.eval is not None:
            interp.execute(args.eval)
        if args.script:
            interp.run_file(args.script)
    except MatlabError as e:
        sys.stdout.flush()
        sys.stderr.write(format_error(e))
        return 1
    except SystemExit as e:
        return int(e.code or 0)
    if not args.no_plots:
        from .builtins.plotting import show_figures
        show_figures(interp, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
