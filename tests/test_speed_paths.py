"""Faster paths added for the speed cases that were slower than GNU Octave; each must give what the general
path gives.

- anonymous functions made inside function files are compiled (they were never compiled before: the closure
  carried the file name, which the compiler took for a captured variable), except where the file's folder has
  private functions;
- a zero-argument function that bails out of compiled code is not tried compiled again;
- sparse banded systems go to LAPACK's banded solver;
- regexp builds only the outputs asked for, and uses finditer for patterns that cannot match empty text;
- one long FFT uses several cores; linspace works in place; char() builds from the string's own storage.
"""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mylab import Interpreter, jit
from mylab.builtins.sparse import _banded_solve
from mylab.builtins.strings import _compile_regex, _never_empty, _regexp_light, _regexp_one
from mylab.values import Cell, char


def _write(folder, name, text):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write(text)


def test_anonymous_function_in_a_function_file_is_compiled(tmp_path):
    _write(tmp_path, "usesanon.m", "function r = usesanon()\n  f = @(x) x^2 + 1;\n"
                                   "  r = arrayfun(f, 1:50);\n  r = sum(r);\nend\n")
    m = Interpreter(out=lambda s: None, path=[str(tmp_path)])
    seen = []
    orig = jit._compile_anonymous

    def spy(interp, handle, sig):
        rec = orig(interp, handle, sig)
        seen.append(rec)
        return rec
    jit._compile_anonymous = spy
    try:
        m.execute("v = usesanon();")
    finally:
        jit._compile_anonymous = orig
    assert np.asarray(m.eval_expression("v")).item() == sum(k * k + 1 for k in range(1, 51))
    assert seen and all(r is not None for r in seen)


def test_anonymous_function_next_to_private_functions_stays_interpreted(tmp_path):
    # helper exists both on the path and as a private function of the folder: only the interpreter knows
    # which one the handle means, so the handle is not compiled
    _write(tmp_path / "pub", "helper.m", "function y = helper(x)\n  y = -1;\nend\n")
    _write(tmp_path / "pkg", "useshelper.m", "function r = useshelper()\n  f = @(x) helper(x) + 0;\n"
                                             "  r = arrayfun(f, 1:20);\nend\n")
    _write(tmp_path / "pkg" / "private", "helper.m", "function y = helper(x)\n  y = 10 * x;\nend\n")
    m = Interpreter(out=lambda s: None, path=[str(tmp_path / "pkg"), str(tmp_path / "pub")])
    m.execute("v = useshelper();")
    assert np.asarray(m.eval_expression("v")).ravel().tolist() == [10.0 * k for k in range(1, 21)]
    assert jit._has_private(str(tmp_path / "pkg" / "useshelper.m"))


def test_zero_argument_function_that_bails_is_not_retried(tmp_path):
    _write(tmp_path, "bigvec.m", "function r = bigvec()\n  x = linspace(0, 1, 1000);\n"
                                 "  y = sin(x).^2 + cos(x).^2;\n  r = sum(y);\nend\n")
    m = Interpreter(out=lambda s: None, path=[str(tmp_path)])
    calls = []
    orig = jit.call_function

    def spy(interp, fn, args, nargout, functions):
        r = orig(interp, fn, args, nargout, functions)
        calls.append((fn, r))
        return r
    jit.call_function = spy
    try:
        for _ in range(6):
            m.execute("v = bigvec();")
    finally:
        jit.call_function = orig
    assert np.isclose(np.asarray(m.eval_expression("v")).item(), 1000.0)
    fn = calls[-1][0]
    records = list(fn._jit_fn["sigs"].values())
    assert records and all(r is None for r in records)  # given up after the first bail-out
    assert sum(1 for r in records if r is not None) == 0


@pytest.mark.parametrize("lower,upper", [(1, 1), (2, 3), (0, 2), (4, 0), (0, 0)])
def test_banded_solve_matches_superlu(lower, upper):
    rng = np.random.default_rng(lower * 10 + upper)
    n = 400
    offsets = list(range(-lower, upper + 1))
    diags = [rng.standard_normal(n - abs(k)) + (6.0 if k == 0 else 0.0) for k in offsets]
    A = sp.diags(diags, offsets, format="csc")
    b = rng.standard_normal((n, 3))
    x = _banded_solve(A, b)
    assert x is not None
    np.testing.assert_allclose(x, spla.splu(A).solve(b), rtol=1e-12, atol=1e-12)


def test_banded_solve_declines_wide_sparse_and_singular_matrices():
    n = 200
    rng = np.random.default_rng(3)
    A = sp.random(n, n, density=0.01, random_state=4, format="csc") + sp.identity(n, format="csc") * 5
    assert _banded_solve(A.tocsc(), rng.standard_normal((n, 1))) is None  # entries far from the diagonal
    S = sp.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="lil")
    S[7, :] = 0
    assert _banded_solve(S.tocsc(), np.ones((n, 1))) is None


def test_sparse_backslash_on_a_banded_system():
    m = Interpreter(out=lambda s: None)
    m.execute("n = 5000; e = ones(n, 1); A = spdiags([-e 2*e -e], -1:1, n, n); b = (1:n)'; x = A \\ b;")
    n = 5000
    e = np.ones(n)
    A = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csc")
    ref = spla.splu(A).solve(np.arange(1, n + 1.0))  # the matrix is ill-conditioned (about 1e7): compare solvers
    x = np.asarray(m.eval_expression("x")).ravel()
    assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-12


def _same(a, b):
    if isinstance(a, Cell):
        return isinstance(b, Cell) and a.shape == b.shape and all(_same(x, y) for x, y in zip(a.data.flat, b.data.flat))
    return type(a) is type(b) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("pattern", [r"\d+", r"(\d)(\w)?", r"x*", r"", r"(?<n>a)b", r"b??", r"[aeiou]{2,}"])
@pytest.mark.parametrize("switches", [set(), {"emptymatch"}])
def test_light_regexp_path_matches_the_general_one(pattern, switches):
    keys = ["match", "start", "end", "tokens", "split"]
    rx = _compile_regex(pattern, switches, "regexp")
    for txt in ["", "a1b22c333", "xxaxx", "ab ab", "日本1語2", "queue aeiou"]:
        ref = _regexp_one(rx, txt, switches, keys)
        got = _regexp_light(rx, txt, switches, keys)
        for k, g in zip(keys, got):
            assert _same(ref[k], g), (pattern, switches, txt, k)


def test_never_empty_patterns():
    import re
    assert _never_empty(re.compile(r"\d+")) and _never_empty(re.compile(r"_(\d+)_v(\d)"))
    for pat in (r"\d*", r"b??", r"(?=a)", r"a|", r""):
        assert not _never_empty(re.compile(pat))


def test_regexp_on_a_cell_array_end_to_end():
    m = Interpreter(out=lambda s: None)
    m.execute("c = {'a1b22', 'none', '7'}; [t, s, e] = regexp(c, '\\d+', 'match', 'start', 'end'); "
              "n = cellfun(@numel, t); s2 = s{1}; t1 = t{1};")
    assert np.asarray(m.eval_expression("n")).ravel().tolist() == [2, 0, 1]
    assert np.asarray(m.eval_expression("s2")).ravel().tolist() == [2, 4]
    assert [str("".join(x.ravel())) for x in m.eval_expression("t1").data.ravel()] == ["1", "22"]


@pytest.mark.parametrize("shape,axis,n", [((1, 300000), 1, None), ((300000, 1), 0, None), ((1, 300000), 1, 400000),
                                          ((1, 1, 300000), 2, None), ((1, 300000), 1, 1000)])
def test_one_long_fft_matches_numpy(shape, axis, n):
    from mylab import fastnum
    x = np.random.default_rng(0).standard_normal(shape)
    for inverse in (False, True):
        got = fastnum.fft(x, n, axis, inverse=inverse)
        ref = (np.fft.ifft if inverse else np.fft.fft)(x, n=n, axis=axis)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_linspace_in_place_is_bit_identical_to_the_formula():
    from mylab.builtins.core import _linspace_values
    for a, b, n in [(0, 1, 10), (0.1, 0.7, 7), (-3.3, 1e5, 1001), (2, 1, 4), (1e-300, 1e300, 9), (0.0, 1.0, 100001)]:
        k = np.arange(n, dtype=float)
        ref = a + k * (b - a) / (n - 1)
        ref[0], ref[-1] = a, b
        assert np.array_equal(_linspace_values(a, b, n), ref), (a, b, n)


@pytest.mark.parametrize("s", ["abc", "a\x00b", "ab\x00", "\x00", "é漢😀", "x" * 1000, "\ud800x"])
def test_char_matches_one_character_per_element(s):
    c = char(s)
    ref = np.array([list(s)], dtype="<U1")
    assert c.shape == ref.shape and c.dtype == ref.dtype and (c == ref).all()
    c[0, 0] = "z"  # writable
