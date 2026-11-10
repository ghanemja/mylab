"""Built-in function library.

Each built-in is a Python function ``f(interp, args, nargout)`` that receives
MATLAB values and returns ``None`` (no output), a single value, or a tuple of
values (multiple outputs). Register new ones with :func:`builtin`.
"""

from ._registry import BUILTINS, builtin  # noqa: F401
from . import core, mathfns, linalg, strings, data, system, numerics, plotting  # noqa: F401,E402
from . import signal, stats, optim, extras  # noqa: F401,E402
from . import table  # noqa: F401,E402
