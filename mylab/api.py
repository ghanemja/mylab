"""High-level session API shared by the web app, the MCP server and the Jupyter kernel.

Example::

    from mylab.api import Session
    s = Session()
    r = s.run("x = magic(3); plot(1:3)")
    r.output      # text printed to the command window
    r.error       # None, or an error message
    r.figures     # list of PNG images (bytes) produced by this run
    s.variables() # summary of the workspace
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .display import format_display, size_str
from .errors import MatlabError
from .interpreter import Interpreter
from .values import Cell, MString, Struct, class_name, is_char, shape_of, to_str


@dataclass
class RunResult:
    output: str = ""
    error: str | None = None
    error_line: int | None = None
    figures: list[bytes] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def figures_base64(self) -> list[str]:
        return [base64.b64encode(f).decode("ascii") for f in self.figures]

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "error": self.error,
            "error_line": self.error_line,
            "figures": self.figures_base64(),
        }


@dataclass
class VariableInfo:
    name: str
    size: str
    cls: str
    preview: str

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "class": self.cls, "preview": self.preview}


def _preview(v: Any, limit: int = 60) -> str:
    try:
        if isinstance(v, MString):
            s = f'"{v}"'
        elif is_char(v) and v.shape[0] <= 1:
            s = f"'{to_str(v)}'"
        elif isinstance(v, np.ndarray) and v.size == 1:
            from .display import format_scalar
            s = format_scalar(v.flat[0])
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 1 and v.size <= 8:
            from .display import format_scalar
            s = "[" + " ".join(format_scalar(x) for x in v.flat) + "]"
        else:
            s = ""  # size and class are reported separately
    except Exception:  # pragma: no cover - previews must never fail
        s = class_name(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class Session:
    """A persistent interpreter that captures output and figures per run."""

    def __init__(self, figure_format: str = "png", dpi: int = 100, cwd: str | None = None):
        os.environ.setdefault("MPLBACKEND", "Agg")
        self._buffer: list[str] = []
        self.interp = Interpreter(out=self._buffer.append)
        self.figure_format = figure_format
        self.dpi = dpi
        if cwd:
            os.chdir(cwd)

    # -- running code -----------------------------------------------------------------
    def run(self, code: str, capture_figures: bool = True, close_figures: bool = True) -> RunResult:
        self._buffer.clear()
        result = RunResult()
        try:
            self.interp.execute(code)
        except MatlabError as e:
            result.error = e.message
            result.error_line = getattr(e, "line", None)
        except SystemExit:
            result.error = "exit/quit is not available in this environment."
        except RecursionError:
            result.error = "Maximum recursion limit reached."
        except KeyboardInterrupt:
            result.error = "Interrupted."
        except Exception as e:  # an engine bug: report it instead of crashing the host
            result.error = f"Internal error: {type(e).__name__}: {e}"
        result.output = "".join(self._buffer)
        self._buffer.clear()
        if capture_figures:
            result.figures = self.render_figures(close=close_figures)
        return result

    def render_figures(self, close: bool = True) -> list[bytes]:
        st = self.interp.state.get("plot")
        if not st or not st.figures:
            return []
        images = []
        for n in sorted(st.figures):
            fig = st.figures[n]
            buf = io.BytesIO()
            fig.savefig(buf, format=self.figure_format, dpi=self.dpi, bbox_inches="tight")
            images.append(buf.getvalue())
        if close:
            from .builtins.plotting import close_all
            close_all(self.interp)
        return images

    def clear_file_cache(self) -> None:
        """Forget parsed .m files (call after editing files on disk)."""
        self.interp._file_cache.clear()
        self.interp._dir_cache.clear()
        self.interp.classes.clear()

    def reset(self) -> None:
        from .builtins.plotting import close_all
        close_all(self.interp)
        self.interp.base.vars.clear()
        self.interp.base.functions = {}
        self.interp.global_vars.clear()

    # -- workspace --------------------------------------------------------------------------
    def variables(self) -> list[VariableInfo]:
        out = []
        for name in sorted(self.interp.base.vars):
            v = self.interp.base.vars[name]
            out.append(VariableInfo(name, size_str(shape_of(v)), class_name(v), _preview(v)))
        return out

    def show_variable(self, name: str) -> str:
        """The text MATLAB would print for ``name`` (without a semicolon)."""
        return format_display(name, self.interp.get_var(name))

    def get(self, name: str) -> Any:
        return to_python(self.interp.get_var(name))

    def set(self, name: str, value: Any) -> None:
        self.interp.set_var(name, value)

    # -- introspection ------------------------------------------------------------------------
    def complete(self, prefix: str) -> list[str]:
        names = set(self.interp.base.vars) | set(self.interp.builtins) | set(self.interp.base.functions)
        return sorted(n for n in names if n.startswith(prefix))

    def help(self, name: str) -> str:
        from .help import help_text
        return help_text(name, self.interp)


def to_python(v: Any) -> Any:
    """Convert a MATLAB value to plain Python / NumPy objects (for JSON or tools)."""
    if isinstance(v, MString):
        return str(v)
    if is_char(v):
        return to_str(v)
    if isinstance(v, Cell):
        return [[to_python(x) for x in row] for row in v.data] if v.data.ndim == 2 else \
            [to_python(x) for x in v.data.flatten(order="F")]
    if isinstance(v, Struct):
        items = [{k: to_python(d[k]) for k in v.fields} for d in v.data.flatten(order="F")]
        return items[0] if len(items) == 1 else items
    if isinstance(v, np.ndarray):
        if v.size == 1:
            x = v.flat[0]
            if isinstance(x, (np.bool_, bool)):
                return bool(x)
            if np.iscomplexobj(v):
                return complex(x)
            return float(x) if v.dtype.kind == "f" else int(x)
        return v
    return repr(v)
