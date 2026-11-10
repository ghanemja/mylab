"""Built-in function library.

Each built-in is a Python function ``f(interp, args, nargout)`` that receives
MATLAB values and returns ``None`` (no output), a single value, or a tuple of
values (multiple outputs). Register new ones with :func:`builtin`.
"""

from ._registry import BUILTINS, builtin  # noqa: F401
from . import core, mathfns, linalg, strings, data, system, numerics, plotting  # noqa: F401,E402
from . import signal, stats, optim, extras  # noqa: F401,E402
from . import table, io  # noqa: F401,E402


def _make_string_aware(names):
    """Let text functions written for char/cell inputs also take string arrays.

    String-array arguments are passed as cell arrays of char vectors; when the
    first argument was a string, text results are converted back to strings.
    """
    import numpy as np

    from ..values import Cell, MString, StringArray, is_char, string_result, to_str
    from .strings import _as_cell_if_strings

    def to_strings(res):
        if isinstance(res, Cell):
            if all(is_char(x) or isinstance(x, MString) for x in res.data.flat):
                data = np.empty(res.data.shape, dtype=object)
                for idx in np.ndindex(res.data.shape):
                    data[idx] = to_str(res.data[idx])
                return string_result(data)
            return res
        if is_char(res) and (res.ndim == 2 and res.shape[0] <= 1):
            return MString(to_str(res))
        return res

    for name in names:
        fn = BUILTINS.get(name)
        if fn is None:
            continue

        def wrapped(interp, args, nargout, _fn=fn):
            was_string = bool(args) and isinstance(args[0], (MString, StringArray))
            if not any(isinstance(a, StringArray) for a in args):
                res = _fn(interp, args, nargout)
            else:
                res = _fn(interp, [_as_cell_if_strings(a) for a in args], nargout)
            if not was_string:
                return res
            if isinstance(res, tuple):
                return tuple(to_strings(r) for r in res)
            return to_strings(res)

        BUILTINS[name] = wrapped



_make_string_aware([
    "split", "join", "compose", "replace", "extractBefore", "extractAfter", "extractBetween",
    "insertAfter", "insertBefore", "erase", "pad", "strip", "strjust", "regexprep", "strtrim",
])
