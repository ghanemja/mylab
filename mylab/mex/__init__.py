"""MEX files: C and C++ functions compiled with mylab's ``mex`` and called like .m functions.

Everything that connects a MEX file to mylab is Python: the MEX file includes
mylab's own ``mex.h`` (in ``include/``), whose ``mx*``/``mex*`` functions call
through a table of function pointers, and :mod:`mylab.mex.runtime` fills that
table with ctypes callbacks when it loads the file.  Nothing is compiled when
mylab is installed; ``mex`` compiles the user's sources (plus the small entry
file in ``include/``) with the system C compiler.  See docs/MEX.md.

This module stays cheap to import: the interpreter uses it to recognise MEX
files on the path, and ctypes is only loaded when one is called.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

INCLUDE_DIR = Path(__file__).resolve().parent / "include"

IN_BROWSER = sys.platform == "emscripten"


def _mexext() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return "mexw64" if sys.maxsize > 2 ** 32 else "mexw32"
    if sys.platform == "darwin":
        return "mexmaca64" if machine in ("arm64", "aarch64") else "mexmaci64"
    return "mexa64"


MEXEXT = _mexext()            # without the dot, as mexext returns it
MEX_SUFFIX = "." + MEXEXT     # the file extension, with the dot

ALL_MEXEXTS = ("mexa64", "mexmaci64", "mexmaca64", "mexw64", "mexw32", "mexglx", "mexmac", "mexmaci", "dll")


def is_mex_path(path: str) -> bool:
    return str(path).endswith(MEX_SUFFIX)


def wrapper_program(path: str):
    """The program the interpreter keeps for a MEX file: one function that hands its call to mylab.

    It is a plain function ``varargout = name(varargin)``, so everything that works with a function
    found on the path (handles, feval, nargin, which, the lookup cache) works with a MEX file too.
    """
    from ..parser import parse
    name = Path(path).name[: -len(MEX_SUFFIX)]
    quoted = str(path).replace("'", "''")
    program = parse(f"function varargout = {name}(varargin)\n"
                    f"varargout = mylabMexCall__('{quoted}', nargout, varargin{{:}});\nend\n")
    for fn in program.functions.values():
        fn.file = path  # type: ignore[attr-defined]
        fn.mex_file = path  # type: ignore[attr-defined]
        fn._jit_fn = {"never": True, "calls": 0, "sigs": {}}  # type: ignore[attr-defined]
    return program


def runtime_for(interp, create: bool = True):
    """The interpreter's MEX runtime (loaded MEX files, arrays, memory); None if none was made yet."""
    rt = interp.state.get("mex_runtime")
    if rt is None and create:
        from .runtime import MexRuntime
        rt = interp.state["mex_runtime"] = MexRuntime(interp)
    return rt


def clear_loaded(interp, patterns) -> None:
    """Unload MEX files whose names match ``patterns`` (all of them for None), except locked ones."""
    rt = runtime_for(interp, create=False)
    if rt is not None:
        rt.clear(patterns)
