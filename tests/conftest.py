import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from mylab import Interpreter  # noqa: E402


class Session:
    """Runs code and captures what would be printed to the command window."""

    def __init__(self):
        self.output: list[str] = []
        self.interp = Interpreter(out=self.output.append)

    def run(self, code: str) -> str:
        self.output.clear()
        self.interp.execute(code)
        return "".join(self.output)

    def val(self, name: str):
        v = self.interp.get_var(name)
        return v

    def num(self, code_or_name: str):
        """Evaluate an expression and return it as a numpy array."""
        v = self.interp.eval_expression(code_or_name)
        return np.asarray(v)

    def str(self, expr: str) -> str:
        from mylab.values import to_str
        return to_str(self.interp.eval_expression(expr))


@pytest.fixture
def m():
    return Session()
