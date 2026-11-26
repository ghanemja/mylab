"""Speed fixes found on slow corpus scripts: the results must stay exactly what the general code gives.

* ``x(i, :) = v``, ``x(mask) = v``, ``x(:, j) = v`` ... on an array no other variable shares are
  written in place instead of copying the whole array each time (a loop filling the rows of a large
  matrix took quadratic time).
* Reading ``A(i, :)``, ``A(i, j)`` of an N-d array, ``c{k}``, and integer arithmetic on one element
  take shorter paths.
* Optional standard-library modules (archives, JSON, CSV, WAV, fractions) are imported on first use.
"""

import subprocess
import sys
import weakref

import numpy as np
import pytest

from mylab import Interpreter, MatlabError
from mylab.values import saturate_int


def run(code):
    m = Interpreter()
    out = []
    m.out = out.append
    m.execute(code)
    return m, "".join(out)


def snapshot(m, names):
    """Values as comparable data: class, shape and bits (NaN-safe)."""
    res = {}
    for n in names:
        v = m.get_var(n)
        if isinstance(v, np.ndarray):
            res[n] = (v.dtype.str, v.shape, v.tobytes(order="F"))
        else:
            res[n] = repr(v)
    return res


# Each statement runs on x twice: once while x is unshared (written in place) and once while another
# variable still holds the old value (the general copy-on-write path). Both must give the same x, and
# the other variable must keep the old value.
CASES = [
    ("x = zeros(3, 4);", "x(2, :) = 1:4;"),
    ("x = zeros(3, 4);", "x(:, 3) = [7; 8; 9];"),
    ("x = zeros(3, 4);", "x(:, 3) = [7 8 9];"),
    ("x = zeros(3, 4);", "x(:) = 5;"),
    ("x = zeros(3, 4);", "x(:) = 1:12;"),
    ("x = zeros(3, 4);", "x(logical([1 0 1]), 2) = -1;"),
    ("x = zeros(3, 4);", "x([1 3], [2 4]) = [1 2; 3 4];"),
    ("x = zeros(3, 4);", "x([3 1], 2:3) = 9;"),
    ("x = zeros(3, 4);", "x(2, [1 1 2]) = [5 6 7];"),
    ("x = zeros(3, 4);", "x(end, end) = 3;"),
    ("x = zeros(3, 4);", "x(end, :) = 4;"),
    ("x = zeros(3, 4);", "x(2, :) = true;"),
    ("x = zeros(3, 4);", "x(2, :) = [true false true false];"),
    ("x = 1:6;", "x([2 4]) = [20 40];"),
    ("x = 1:6;", "x(x > 3) = 0;"),
    ("x = 1:6;", "x(logical([0 1 0 1 0 0])) = [8 9];"),
    ("x = 1:6;", "x(:) = 6:-1:1;"),
    ("x = (1:6)';", "x(2:3) = [1 2];"),
    ("x = 1:6;", "x([]) = 5;"),
    ("x = zeros(2, 3, 2);", "x(1, :, 2) = [1 2 3];"),
    ("x = zeros(2, 3, 2);", "x(:, :, 1) = 9;"),
    ("x = zeros(2, 3, 2);", "x(2, 3, :) = [4 5];"),
    ("x = true(2, 2);", "x(1, :) = false;"),
    ("x = 'hello';", "x(1:2) = 'HE';"),
    ("x = int8([1 2 3; 4 5 6]);", "x(2, :) = int8([7 8 9]);"),
    ("x = uint16(1:4);", "x(2:3) = uint16([9 8]);"),
    ("x = single([1 2 3]);", "x(2:3) = single([5 6]);"),
    ("x = [1+2i, 3];", "x(2) = 5i;"),
    ("x = zeros(3, 1); k = 0.5;", "x(1, k * 4) = 1;"),
    ("x = zeros(2, 2);", "x(1, :) = [1 2 3];"),
    ("x = zeros(2, 2);", "x(3, :) = [1 2];"),
    ("x = zeros(2, 2);", "x(:, 4) = 1;"),
    ("x = zeros(2, 2);", "x(0, 1) = 1;"),
    ("x = zeros(2, 2);", "x(1.5, 1) = 1;"),
    ("x = zeros(2, 2);", "x(1, :) = [];"),
    ("x = zeros(2, 2);", "x(:) = [1 2 3];"),
    ("x = zeros(2, 2);", "x(5) = 1;"),
    ("x = zeros(2, 2);", "x(2, :) = 'ab';"),
    ("x = zeros(2, 3);", "x(2, :) = [1; 2; 3];"),
    ("x = magic(4); idx = [2 3];", "x(idx(1), :) = x(idx(2), :);"),
    ("x = magic(4);", "x(x(1, 1) - 15, :) = -x(2, :);"),
    ("x = reshape(1:24, 2, 3, 4);", "x(2, 5) = 0;"),
    ("x = zeros(3);", "x(1:2, :) = ones(2, 3) * 2;"),
]


@pytest.mark.parametrize("setup,stmt", CASES)
def test_assignment_in_place_matches_copy(setup, stmt):
    results = []
    for shared in (False, True):
        m = Interpreter()
        m.out = lambda s: None
        m.execute(setup)
        if shared:
            m.execute("old = x;")
        try:
            m.execute(stmt)
            err = None
        except MatlabError as e:
            err = e.message
        results.append((err, snapshot(m, ["x"])))
        if shared:
            again = Interpreter()
            again.out = lambda s: None
            again.execute(setup)
            assert snapshot(m, ["old"])["old"] == snapshot(again, ["x"])["x"]  # the alias is untouched
    assert results[0] == results[1]


def test_row_assignment_keeps_the_same_array():
    """A row store into an unshared array changes it in place (no copy of the whole array)."""
    m = Interpreter()
    m.execute("x = zeros(50000, 3); r = [1 2 3];")
    w = weakref.ref(m.base.vars["x"])
    m.execute("x(7, :) = r; x(:, 2) = 5; x(x > 4) = 4; x(9, 1:2) = [8 9];")
    assert m.base.vars["x"] is w()
    x = m.get_var("x")
    assert x[6].tolist() == [1, 4, 3] and x[8].tolist() == [8, 9, 0] and x[0].tolist() == [0, 4, 0]


def test_shared_array_is_copied_not_changed():
    m, _ = run("x = zeros(3); y = x; x(2, :) = 1; s.a = zeros(2); t = s; t.a(1, :) = 7;")
    assert m.get_var("y").sum() == 0 and m.get_var("x")[1].tolist() == [1, 1, 1]
    assert m.get_var("s").data.flat[0]["a"].sum() == 0


def test_loop_filling_rows_of_a_large_matrix():
    m, out = run("n = 20000; x = zeros(n, 2); off = 0; "
                 "for i = 1:n, d = mod(i, 7) > 3; off(d) = off(d) + 1; x(i, :) = [i, off]; end, "
                 "disp(sum(x(:)))")
    x = np.zeros((20000, 2))
    off = 0
    for i in range(1, 20001):
        if i % 7 > 3:
            off += 1
        x[i - 1] = [i, off]
    assert float(out.strip()) == pytest.approx(x.sum(), rel=1e-12)


def test_index_with_side_effects_is_evaluated_once():
    m, out = run("function k = nxt()\n persistent c; if isempty(c), c = 0; end; c = c + 1; k = c;\nend\n"
                 "x = zeros(3, 2); x(nxt(), :) = [1 2]; x(nxt(), :) = [3 4]; disp(x)")
    assert m.get_var("x").tolist() == [[1, 2], [3, 4], [0, 0]]


def test_global_and_nested_arrays_keep_their_semantics():
    _, out = run("function r = outer()\n x = zeros(2); inner(); r = x;\n"
                 " function inner()\n  x(2, :) = [5 6];\n end\nend\n"
                 "disp(outer())")
    assert out.split() == ["0", "0", "5", "6"]
    _, out = run("global G; G = zeros(1, 3); G(:) = 2; disp(G)")
    assert out.split() == ["2", "2", "2"]


def test_folded_and_padded_scalar_indices():
    _, out = run("a = reshape(1:24, 2, 3, 4); disp([a(2, 5), a(1, 12), a(2, 3, 1), a(1, 1, 4)]); "
                 "m = magic(3); disp([m(2, 3, 1), m(3, 1, 1, 1)]); u = uint16(a); v = u(2, 7) + 1; "
                 "disp(class(v)); disp(v)")
    assert out.split() == ["10", "23", "6", "19", "7", "4", "uint16", "15"]
    for code, msg in [("a = reshape(1:24, 2, 3, 4); a(3, 1)", "Index in position 1 exceeds array bounds"),
                      ("a = reshape(1:24, 2, 3, 4); a(1, 13)", "Index in position 2 exceeds array bounds"),
                      ("m = magic(3); m(1, 1, 2)", "Index in position 3 exceeds array bounds")]:
        with pytest.raises(MatlabError, match=msg):
            run(code)


def test_row_and_column_reads():
    _, out = run("z = magic(4); disp(z(2, :)); disp(z(:, 3)'); disp(size(z(2, :))); disp(size(z(:, 3))); "
                 "c = {1, 'two'; [3 4], {5}}; disp(c{2}); disp(class(c{3})); disp(c{1, 2}); "
                 "q = zeros(2, 2, 2); disp(size(q(1, :, 2)))")
    assert out.split() == ["5", "11", "10", "8", "3", "10", "6", "15", "1", "4", "4", "1", "3", "4", "char",
                           "two", "1", "2"]
    with pytest.raises(MatlabError, match="Index exceeds the number of array elements"):
        run("c = {1, 2}; c{3}")


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32"])
def test_saturate_one_element_matches_arrays(dtype):
    values = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, 2.4999999999999996, 1e300, -1e300,
              np.inf, -np.inf, np.nan, -0.0, 127.5, -128.5, 255.5, 65535.5, 2147483647.5, -2147483648.6,
              4294967295.5, 3.0]
    for x in values:
        one = saturate_int(np.array([[x]]), dtype)
        many = saturate_int(np.array([[x, x]]), dtype)[:, :1]
        assert one.dtype == many.dtype and one.shape == many.shape and one.tobytes() == many.tobytes(), x


def test_optional_modules_are_imported_on_first_use():
    code = ("import sys, mylab\n"
            "m = mylab.Interpreter(); m.execute('x = 1;')\n"
            "late = ['tarfile', 'zipfile', 'gzip', 'json', 'csv', 'wave', 'fractions', 'xml.etree.ElementTree']\n"
            "print(sorted(n for n in late if n in sys.modules))\n"
            "m.execute(\"s = jsondecode('{\\\"a\\\": [1, 2]}'); t = jsonencode(s); f = rats(0.75);\")\n"
            "print(m.get_var('t').tobytes().decode('utf-32-le') if False else ''.join(m.get_var('t').ravel()))\n"
            "print(''.join(m.get_var('f').ravel()).strip())\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    assert lines[0] == "[]"
    assert lines[1] == '{"a":[1,2]}'
    assert lines[2] == "3/4"
