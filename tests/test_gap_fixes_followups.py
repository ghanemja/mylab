"""Bugs found while the gap-fix work ran real code: interp2 with grid vectors of different orientation, bsxfun
across dimensions one operand lacks, and input() when there is nothing left to read."""
import numpy as np
import pytest

from mylab import Interpreter
from mylab.errors import MatlabError


@pytest.fixture
def m():
    return Interpreter(out=lambda s: None)


def num(m, expr):
    return np.asarray(m.eval_expression(expr), dtype=float)


def test_interp2_vectors_of_different_orientation_are_grid_vectors(m):
    m.execute("[X, Y] = meshgrid(1:4, 1:3); V = X + 10*Y;")
    expected = np.array([[11, 12, 13], [21, 22, 23]], dtype=float)  # V(y, x) = x + 10 y on Yq = [1 2], Xq = 1:3
    np.testing.assert_allclose(num(m, "interp2(X, Y, V, (1:3)', [1 2])"), expected)  # column Xq, row Yq
    np.testing.assert_allclose(num(m, "interp2(X, Y, V, [1 2 3], [1; 2])"), expected)  # row Xq, column Yq
    np.testing.assert_allclose(num(m, "interp2(V, (1:3)', [1 2])"), expected)
    # vectors of the same orientation are pairs of points
    np.testing.assert_allclose(num(m, "interp2(X, Y, V, [1.5 2.5], [1 2])"), [[11.5, 22.5]])


def test_bsxfun_treats_missing_trailing_dimensions_as_one(m):
    m.execute("im = reshape(1:30, 3, 2, 5); s = sum(im, 3); r = bsxfun(@times, im, 1 ./ s);"
              "r2 = bsxfun(@plus, ones(2, 1), reshape(1:6, 1, 3, 2));")
    im = np.arange(1, 31, dtype=float).reshape(3, 2, 5, order="F")
    np.testing.assert_allclose(num(m, "r"), im / im.sum(axis=2, keepdims=True))
    np.testing.assert_allclose(num(m, "r2"), 1 + np.arange(1, 7, dtype=float).reshape(1, 3, 2, order="F")
                               * np.ones((2, 1, 1)))
    with pytest.raises(MatlabError, match="non-singleton"):
        m.execute("bsxfun(@plus, ones(2, 3), ones(3, 2))")


def test_input_at_end_of_input_is_a_catchable_error(m):
    def no_more(prompt=""):
        raise EOFError
    m.input_func = no_more
    m.execute("try, x = input('n? '); msg = ''; catch e, msg = e.identifier; end")
    assert "".join(np.asarray(m.eval_expression("msg")).ravel()) == "MATLAB:input:EndOfInput"
