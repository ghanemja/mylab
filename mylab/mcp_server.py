"""MCP server: lets AI assistants (Claude Desktop, Cursor, ...) run code in mylab.

Run it with ``mylab-mcp`` (stdio transport). Configure your MCP client with::

    {"mcpServers": {"mylab": {"command": "mylab-mcp"}}}

Tools: run_code, run_file, list_variables, get_variable, function_help,
search_functions and reset_session. Plots created by code come back as images.
Set MYLAB_WORKDIR to choose the folder where scripts and functions are found.
"""

from __future__ import annotations

import os
from typing import Any

from .api import Session

INSTRUCTIONS = """\
mylab runs MATLAB-language code (an open-source interpreter built on NumPy/SciPy/Matplotlib).
Use run_code to execute code in a persistent workspace: variables survive between calls, statements
without a semicolon print their value, and figures are returned as images. Use function_help and
search_functions to check which functions mylab supports before relying on one. reset_session clears
the workspace."""

_session: Session | None = None


def session() -> Session:
    global _session
    if _session is None:
        workdir = os.environ.get("MYLAB_WORKDIR")
        _session = Session(cwd=workdir if workdir and os.path.isdir(workdir) else None)
    return _session


def _format_result(result, image_cls) -> list[Any]:
    parts: list[Any] = []
    text = result.output.rstrip("\n")
    if result.error:
        where = f" (line {result.error_line})" if result.error_line else ""
        text = (text + "\n" if text else "") + f"Error{where}: {result.error}"
    if not text and not result.figures:
        text = "(no output)"
    if text:
        parts.append(text)
    for png in result.figures:
        parts.append(image_cls(data=png, format="png"))
    return parts


def build_server():
    """Create the MCP server (works with the mcp package v1 and v2)."""
    try:
        from mcp.server.mcpserver import Image, MCPServer as Server
    except ImportError:  # mcp < 2
        try:
            from mcp.server.fastmcp import FastMCP as Server, Image
        except ImportError as e:  # pragma: no cover
            raise SystemExit("The MCP server needs the 'mcp' package: pip install 'mylab-lang[mcp]'") from e

    server = Server(name="mylab", instructions=INSTRUCTIONS)

    @server.tool()
    def run_code(code: str) -> list:
        """Run MATLAB-language code in the persistent mylab workspace.

        Returns the command-window output, any error message (with line number) and any figures
        as PNG images. Variables persist between calls.
        """
        return _format_result(session().run(code), Image)

    @server.tool()
    def run_file(path: str) -> list:
        """Run a .m script or function file from disk (relative paths use the working folder)."""
        s = session()
        full = os.path.abspath(path)
        if not os.path.isfile(full):
            return [f"Error: file not found: {full}"]
        quoted = full.replace("'", "''")
        return _format_result(s.run(f"run('{quoted}')"), Image)

    @server.tool()
    def list_variables() -> str:
        """List workspace variables with their size, class and a short preview of small values."""
        rows = session().variables()
        if not rows:
            return "The workspace is empty."
        lines = ["name | size | class | value", "--- | --- | --- | ---"]
        lines += [f"{v.name} | {v.size} | {v.cls} | {v.preview}" for v in rows]
        return "\n".join(lines)

    @server.tool()
    def get_variable(name: str) -> str:
        """Show the full value of one workspace variable, as the command window would print it."""
        try:
            return session().show_variable(name).rstrip("\n")
        except Exception as e:  # noqa: BLE001 - report any lookup problem to the model
            return f"Error: {getattr(e, 'message', e)}"

    @server.tool()
    def function_help(name: str) -> str:
        """Help for a function: supported calling forms and what it does in mylab."""
        return session().help(name).rstrip("\n")

    @server.tool()
    def search_functions(query: str) -> str:
        """Search built-in function names and help text for a keyword (e.g. 'fourier', 'regression')."""
        from .help import _entries, summary
        q = query.lower().strip()
        entries = _entries()
        hits = []
        for name in sorted(session().interp.builtins, key=str.lower):
            entry = entries.get(name)
            haystack = name.lower() + " " + (entry[1].lower() if entry else "")
            if q and q in haystack:
                hits.append(f"{name}: {summary(name) or ''}".rstrip(": "))
        if not hits:
            return f"No functions match '{query}'."
        more = f"\n... and {len(hits) - 40} more" if len(hits) > 40 else ""
        return "\n".join(hits[:40]) + more

    @server.tool()
    def reset_session() -> str:
        """Clear all variables, functions defined in the session and figures."""
        session().reset()
        return "Workspace cleared."

    return server


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    build_server().run("stdio")


if __name__ == "__main__":
    main()
