"""mylab: a free, open-source interpreter for MATLAB-language code, built on NumPy.

Quick start from Python::

    from mylab import Interpreter
    m = Interpreter()
    m.execute("A = magic(4); x = A \\ ones(4, 1)")
    m.get_var("x")
"""

__version__ = "1.0.0"

from .errors import MatlabError  # noqa: E402
from .interpreter import Interpreter  # noqa: E402

__all__ = ["Interpreter", "MatlabError", "__version__"]
