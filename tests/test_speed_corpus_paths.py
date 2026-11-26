"""Faster paths added for slow corpus programs; each must give exactly what the general path gives.

- bsxfun with a sparse operand hands it to the operator, which keeps it sparse (no full copy);
  a sparse matrix times a sparse row or column expands only the vector;
- x(i) = v writes in place into global and persistent variables too, and s.f(i) = v and
  s(k).f(i) = v write in place into the field when nothing else refers to it; s(k).f is read
  without building s(k);
- products of large, mostly-zero integer matrices are computed as sparse products (exact);
- 1x1 integer arithmetic in the interpreter is done in Python, rounded and saturated;
- the loop compiler reads elements of integer-class arrays and stores into 3-D double arrays;
- small factorisations run on one BLAS thread.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fuzz_fastpaths_util import INT_CLASSES, check_all, count, num, rand_double  # noqa: E402

from mylab import Interpreter, fastnum, ops  # noqa: E402
from mylab.parser import parse  # noqa: E402


def run(code: str, path=None) -> Interpreter:
    m = Interpreter(out=lambda s: None, path=list(path or []))
    m.execute(code)
    return m


def val(m: Interpreter, expr: str):
    return m.eval_expression(expr)


# -- sparse ------------------------------------------------------------------------------------------

def test_bsxfun_keeps_a_sparse_operand_sparse():
    m = run("A = sparse([1 2 3 3 4], [1 2 3 1 4], [2 -4 6 1.5 8], 4, 4); v = [1; 2; 4; 0.5];"
            "B = bsxfun(@times, A, 1 ./ v); F = full(A) .* (1 ./ v); C = bsxfun(@rdivide, A, v');"
            "G = full(A) ./ v'; rs = sum(A, 2); D = bsxfun(@times, A, 1 ./ rs); H = full(A) .* (1 ./ full(rs));"
            "E = bsxfun(@(x, y) x .* y, A, v); P = bsxfun(@plus, A, v); Q = full(A) + v;")
    for sparse, full in (("B", "F"), ("C", "G"), ("D", "H")):
        assert val(m, f"issparse({sparse})").item(), sparse
        assert val(m, f"isequal(full({sparse}), {full})").item(), sparse
    assert val(m, "isequal(full(E), full(A) .* v)").item()  # (an anonymous function: made full, as before)
    assert not val(m, "issparse(P)").item() and val(m, "isequal(P, Q)").item()


def test_sparse_times_sparse_vector_expands_the_vector_only():
    m = run("A = sparse([1 2 3], [1 3 2], [2 -4 6], 3, 3); r = sparse([1 0 3]); c = sparse([0; 2; -1]);"
            "X = A .* r; Y = c .* A; Z = sparse([1; 0; 2]) .* sparse([1 2 0]); W = A .* sparse([1; Inf; 0]);")
    a = np.array([[2, 0, 0], [0, 0, -4], [0, 6, 0]], dtype=float)
    assert np.array_equal(val(m, "full(X)"), a * np.array([[1, 0, 3]]))
    assert np.array_equal(val(m, "full(Y)"), np.array([[0], [2], [-1]]) * a)
    assert np.array_equal(val(m, "full(Z)"), np.array([[1], [0], [2]]) * np.array([[1, 2, 0]]))
    w = np.asarray(val(m, "full(W)"))
    with np.errstate(invalid="ignore"):
        assert np.array_equal(w, a * np.array([[1], [np.inf], [0]]), equal_nan=True)
    for name in "XYZW":
        assert val(m, f"issparse({name})").item(), name


# -- in-place stores ---------------------------------------------------------------------------------

ALIAS_CASES = [
    # globals and persistents
    ("global G; G = zeros(1, 3); H = G; G(2) = 1; r = H;", [[0, 0, 0]]),
    ("global G; G = zeros(1, 3); H = G; H(2) = 1; r = G;", [[0, 0, 0]]),
    ("global G; G = zeros(1, 3); c = {G}; G(1) = 4; r = c{1};", [[0, 0, 0]]),
    ("global G; G = zeros(1, 3); for k = 1:3, G(k) = k; end; r = G;", [[1, 2, 3]]),
    # s.f(i) = v and s(k).f(i) = v
    ("s.a = zeros(1, 3); t = s; s.a(2) = 5; r = t.a;", [[0, 0, 0]]),
    ("s.a = zeros(1, 3); t = s; t.a(2) = 5; r = s.a;", [[0, 0, 0]]),
    ("s.a = zeros(1, 3); v = s.a; s.a(1) = 4; r = v;", [[0, 0, 0]]),
    ("s.a = zeros(1, 3); c = {s}; s.a(3) = 1; r = c{1}.a;", [[0, 0, 0]]),
    ("s.a = zeros(1, 3); f = @() s.a; s.a(3) = 1; r = f();", [[0, 0, 0]]),
    ("q(3).a = zeros(1, 3); w = q(3); q(3).a(2) = 7; r = w.a;", [[0, 0, 0]]),
    ("q(3).a = zeros(1, 3); w = q; q(3).a(2) = 7; r = w(3).a;", [[0, 0, 0]]),
    ("q(3).a = zeros(1, 3); w = q(2:3); q(3).a(2) = 7; r = w(2).a;", [[0, 0, 0]]),
    ("q(3).a = zeros(1, 3); w = q(2:3); w(2).a(1) = 7; r = q(3).a;", [[0, 0, 0]]),
    ("e = struct('a', {zeros(1, 2), zeros(1, 2)}); e2 = e; e(1).a(2) = 9; r = e2(1).a;", [[0, 0]]),
    ("global GS; GS.a = zeros(1, 3); L = GS; GS.a(2) = 1; r = L.a;", [[0, 0, 0]]),
    # the stores themselves
    ("h.x = zeros(2, 2, 2); for k = 1:2, h.x(k, k, 2) = k; end; r = h.x(:, :, 2);", [[1, 0], [0, 2]]),
    ("p(2).m = zeros(2); p(2).m(:, 1) = [1; 2]; r = p(2).m;", [[1, 0], [2, 0]]),
    ("g.v = int8([1 2 3]); g.v(2) = 300; r = double(g.v);", [[1, 127, 3]]),
    ("g.v = [1 2 3]; g.v(2) = true; r = [g.v, isa(g.v, 'double')];", [[1, 1, 3, 1]]),
    ("g.v = [1 2 3]; g.v(5) = 4; r = g.v;", [[1, 2, 3, 0, 4]]),
    ("s.t = {1, 2}; s.t{2} = 5; r = s.t{2};", [[5]]),
    ("q(2).a = [1 2]; k = 1; q(end).a(end) = 9; r = q(2).a;", [[1, 9]]),
]


@pytest.mark.parametrize("code, expected", ALIAS_CASES)
def test_in_place_stores_keep_value_semantics(code, expected):
    m = run(code)
    assert np.asarray(val(m, "r")).tolist() == expected


def test_unshared_fields_and_globals_are_written_in_place():
    import weakref  # (a weak reference does not count as a second holder)
    m = run("s.a = zeros(1, 5); q(3).b = zeros(2); global G; G = zeros(1, 4);")
    a = weakref.ref(m.base.vars["s"].data[0, 0]["a"])
    b = weakref.ref(m.base.vars["q"].data[0, 2]["b"])
    g = weakref.ref(m.global_vars["G"])
    m.execute("s.a(2) = 3; q(3).b(2, 1) = 4; q(end).b(:, 2) = [5; 6]; G(3) = 7; s.a(s.a == 3) = 8;")
    assert m.base.vars["s"].data[0, 0]["a"] is a() and a().tolist() == [[0, 8, 0, 0, 0]]
    assert m.base.vars["q"].data[0, 2]["b"] is b() and b().tolist() == [[0, 5], [4, 6]]
    assert m.global_vars["G"] is g() and g().tolist() == [[0, 0, 7, 0]]
    m.execute("t = s; s.a(1) = 1;")  # shared now: copied first
    assert m.base.vars["s"].data[0, 0]["a"] is not a() and m.base.vars["t"].data[0, 0]["a"] is a()


def test_stores_through_functions_keep_value_semantics(tmp_path):
    (tmp_path / "setf_.m").write_text("function setf_(s)\n  s.a(1) = 99;\n  s.a(2, 2) = 1;\nend\n")
    (tmp_path / "pers_.m").write_text("function r = pers_(cmd)\n  persistent P\n  r = [];\n  switch cmd\n"
                                      "    case 'reset', P = zeros(1, 3);\n    case 'set', P(1) = P(1) + 1;\n"
                                      "    case 'get', r = P;\n  end\nend\n")
    m = run("s.a = zeros(2); setf_(s); r1 = s.a; pers_('reset'); a = pers_('get'); pers_('set'); b = pers_('get');"
            "pers_('set'); c = pers_('get'); for k = 1:5, pers_('set'); end; d = pers_('get');", [str(tmp_path)])
    assert np.asarray(val(m, "r1")).tolist() == [[0, 0], [0, 0]]
    assert np.asarray(val(m, "a")).tolist() == [[0, 0, 0]]
    assert np.asarray(val(m, "b")).tolist() == [[1, 0, 0]]
    assert np.asarray(val(m, "c")).tolist() == [[2, 0, 0]]
    assert np.asarray(val(m, "d")).tolist() == [[7, 0, 0]]


def test_struct_element_field_reads_match_the_general_path():
    check_all([
        "q(2).a = 5; q(1).a = [1 2]; x = q(2).a; y = q(end).a; z = q(1).a(2); k = 2; w = q(k).a;",
        "q(2).a = 5; x = q(3).a;",
        "q(2).a = 5; x = q(2).b;",
        "q(2).a = 5; x = q(0).a;",
        "q(2).a = 5; x = q(1.5).a;",
        "q(2).a = 5; x = q(true).a;",
        "q(2).a = 5; x = q([1 2]).a;",
        "q = struct('a', {1, 2; 3, 4}); x = q(2, 1).a; y = q(4).a;",
    ])


# -- sparse products of integer matrices ---------------------------------------------------------------

def _counts(rng, m, k, density=0.004, lo=-3, hi=9):
    return (rng.random((m, k)) < density) * rng.integers(lo, hi, (m, k)).astype(float)


def test_exact_sparse_products_are_bit_identical():
    rng = np.random.default_rng(1)
    cases = []
    a = _counts(rng, 300, 3000)
    cases += [(a, a.T.copy()), (a, a.T), (np.asfortranarray(a), a.T.copy())]
    cases.append((a, rng.integers(-2, 3, (3000, 200)).astype(float)))       # one operand dense
    cases.append((rng.integers(-2, 3, (200, 3000)).astype(float), a.T.copy()))
    b = _counts(rng, 80, 120000)
    cases.append((b, b.T.copy()))
    for x, y in cases:
        got = fastnum.matmul_exact(x, y)
        assert got is not None
        want = x @ y
        assert got.shape == want.shape and got.dtype == want.dtype and got.tobytes() == want.tobytes()
        assert got.flags.writeable and got.flags.c_contiguous


def test_exact_sparse_products_decline_when_order_could_matter():
    rng = np.random.default_rng(2)
    a = _counts(rng, 300, 3000)
    frac = a.copy()
    frac[np.nonzero(frac)[0][0], np.nonzero(frac)[1][0]] = 0.5   # one non-integer
    big = a * 2.0 ** 40                                         # products and sums beyond 2^52
    nan = a.copy()
    nan[0, 0] = np.nan
    dense = rng.integers(-2, 3, (300, 3000)).astype(float)       # no zeros to skip
    small = _counts(rng, 30, 30)
    for x, y in ((frac, a.T), (a, frac.T), (big, big.T), (nan, a.T), (dense, dense.T), (small, small.T)):
        assert fastnum.matmul_exact(x, y) is None
    assert ops.matmul(nan, a.T.copy()).shape == (300, 300)


def test_matrix_product_through_the_interpreter_uses_it():
    m = run("rng(3); c = double(rand(200, 4000) < 0.003) .* round(10 * rand(200, 4000)); K = c * c';"
            "K2 = full(sparse(c) * sparse(c)');")
    assert val(m, "isequal(K, K2)").item()


# -- integer scalars in the interpreter -------------------------------------------------------------

def test_integer_scalar_arithmetic_matches_the_general_path():
    rng = random.Random(99)
    small = [c for c in INT_CLASSES if c not in ("int64", "uint64")]
    ops_ = ["+", "-", "*", ".*", "/", "./", "==", "~=", "<", "<=", ">", ">=", "^", "\\", "&"]
    progs = []
    for _ in range(count(60)):
        stmts = []
        for k in range(10):
            cls = rng.choice(small)
            info = np.iinfo(cls)
            a = rng.choice([int(info.min), int(info.max), 0, 1, -1, 7, rng.randint(int(info.min), int(info.max))])
            a = f"{cls}({min(max(a, int(info.min)), int(info.max))})"
            b = rng.choice([num(rand_double(rng, 0.6)), "true", "false", f"{cls}({rng.randint(-5, 300)})",
                            f"{rng.choice(small)}(3)", "single(2.5)"])
            x, y = (a, b) if rng.random() < 0.6 else (b, a)
            stmts.append(f"try, r{k} = {x} {rng.choice(ops_)} {y}; catch err, e{k} = err.message; end")
        progs.append("\n".join(stmts))
    check_all(progs)


# -- loop compiler: integer-class arrays and 3-D arrays ---------------------------------------------------

JIT_CASES = [
    "im = uint16([1 2; 3 4; 0 1]); h = zeros(3, 2, 5); for j = 1:3, for k = 1:2, h(j, k, im(j, k) + 1) = 1; end, end",
    "im = uint16(magic(4)); h = zeros(4, 4, 20); for j = 1:4, for k = 1:4, h(j, k, im(j, k) + 1) = 1; end, end",
    "im = uint16([1 2 4 3 0]); h = zeros(1, 5, 5); for k = 1:5, h(1, k, im(k) + 1) = k; end",
    "im = uint8(cat(3, [1 2; 3 4], [5 6; 7 8])); s = 0; for k = 1:8, s = s + double(im(k)); end",
    "im = uint8(cat(3, [1 2; 3 4], [5 6; 7 8])); s = 0; for k = 1:4, s = s + double(im(2, k)); end",
    "im = uint8(cat(3, [1 2; 3 4], [5 6; 7 8])); s = 0; for k = 1:6, s = s + double(im(1, k)); end",
    "im = int8([100 120 -128 127]); s = 0; t = int8(0); for k = 1:4, s = s + double(im(k)); t = t + im(k); end",
    "im = int8([100 120 -128 127]); s = 0; for k = 1:6, s = s + double(im(k)); end",
    "im = uint8([1 2; 3 4]); s = 0; for k = 1:4, s = s + double(im(k)); u = im(k) * 100; end",
    "im = int16([1 2 3 4]); for k = 1:4, im = double(im); end, s = 0; for k = 1:4, s = s + im(k); end",
    "A = zeros(2, 3, 4); for k = 1:4, A(1, 2, k) = k; A(2, 3, k) = -k; end, s = 0;"
    " for k = 1:4, s = s + A(1, 2, k); end",
    "A = zeros(2, 3, 4); for k = 1:6, A(1, 2, k) = k; end",
    "A = zeros(2, 3, 4); for k = 1:4, A(1, k) = k; end",
    "A = zeros(2, 3, 4); for k = 1:4, A(k) = k; end",
    "A = zeros(2, 3, 4); n = 0; for k = 1:4, n = n + numel(A) + length(A) + size(A, 1) + size(A, 2); end",
    "A = ones(2, 3, 4); B = A; for k = 1:4, A(1, 1, k) = 0; end",
    "A = zeros(2, 2, 2); for k = 1:4, A(1, 1, 1) = A(1, 1, 1) + k; end",
    "A = rand(3, 3, 3) > 2; for k = 1:4, A(1, 1, 1) = k; end",
    "A = zeros(2, 2, 2); for k = 1:4, A(1, 1, 3) = k; end",
    "A = zeros(2, 2, 2); for k = 1:4, A(1, 1, 2) = 1i; end",
]


@pytest.mark.parametrize("code", JIT_CASES)
def test_compiled_integer_and_3d_array_loops_match(code):
    check_all([code])


def test_3d_store_with_integer_index_is_compiled():
    m = Interpreter(out=lambda s: None)
    m.execute("im = uint16(magic(6)); h = zeros(6, 6, 40); j = 2;")
    program = parse("for k = 1:6\n  h(j, k, im(j, k) + 1) = 1;\nend")
    m.run_program(program, m.base)
    info = program.body[0]._jit_static
    assert any(v is not None for v in info["compiled"].values()) and info["bails"] == 0
    assert np.asarray(val(m, "nnz(h)")).item() == 6


# -- BLAS threads -----------------------------------------------------------------------------------

def test_small_factorisations_restore_the_thread_count():
    m = run("a = magic(4) + eye(4); [u, s, v] = svd(a); e = expm(a / 40); d = det(a); x = a \\ ones(4, 1);"
            "r = norm(u * s * v' - a);")
    assert np.asarray(val(m, "r")).item() < 1e-12
    for _set, count_ in fastnum._THREAD_CONTROLS:  # (each library's thread count, read once)
        assert count_ > 1
    saved = fastnum.one_thread()
    fastnum.restore_threads(saved)
    be = fastnum.lapack_backend()
    if be is not None and be.get_threads is not None and fastnum._THREAD_CONTROLS:
        assert be.get_threads() == fastnum._THREAD_CONTROLS[0][1]  # restored after the calls


def test_one_thread_wrapper_skips_large_arguments():
    calls = []

    def slow(interp, args, nargout):
        calls.append(len(fastnum._THREAD_CONTROLS))
        return args[0]

    f = fastnum._on_one_thread(slow, 10)
    assert f(None, [np.zeros((3, 3))], 1).shape == (3, 3)
    assert f(None, [np.zeros((30, 3))], 1).shape == (30, 3)
    assert f.__wrapped__ is slow and f.one_thread


# -- compiled conditions in loops the compiler does not take --------------------------------------------

class _CondGen:
    def __init__(self, rng):
        self.rng = rng

    def atom(self):
        rng = self.rng
        return rng.choice([
            "k", "a", "b", "v(k)", "v(end)", "w(mod(k, 3) + 1)", "numel(v)", "c{mod(k, 2) + 1}", "s",
            num(rand_double(rng, 0.4)), "NaN", "true", "false", "int8(k)", "uint8(200)", "t", "str(k)",
            "isempty(strfind(str, 'b'))", "numel(strfind(str, 'a'))", "any(v > k)", "all(w)", "sum(v)",
            "mod(k, 3)", "abs(a - k)", "max(k, b)", "x", "f(k)", "[k 1]", "w",
        ])

    def expr(self, depth):
        rng = self.rng
        if depth == 0 or rng.random() < 0.3:
            return self.atom()
        r = rng.random()
        if r < 0.6:
            op = rng.choice(["==", "~=", "<", ">", "<=", ">=", "&", "|", "&&", "||", "+", "-", "*", "/"])
            return f"({self.expr(depth - 1)} {op} {self.expr(depth - 1)})"
        if r < 0.75:
            return f"~({self.expr(depth - 1)})"
        return f"{rng.choice(['isempty', 'abs', 'any', 'numel', 'double', 'logical'])}({self.expr(depth - 1)})"

    def program(self):
        rng = self.rng
        x = rng.choice(["3", "NaN", "[1 2]", "'a'"])
        init = (f"a = {num(rand_double(rng, 0.3))}; b = {rng.choice(['2', 'int16(3)', 'true', '[]', '0.5'])};"
                f" v = {rng.choice(['[3 1 4 1 5 9 2 6]', 'int8([3 1 4 1 5 9 2 6])', '[1 NaN 3 4 5 6 7 8]'])};"
                f" w = {rng.choice(['[1 0 1]', 'logical([1 1 1])', '[2 3 4]'])}; c = {{'ab', [1 2]}};"
                f" s = 0; t = int32(2); str = 'abcab'; x = {x};"
                f" f = @(z) z * 2;")
        body = [
            "j = 0; while j < 1, j = j + 1; end",  # (a loop inside: the loop itself stays interpreted)
            f"if {self.expr(3)}, s = s + 1; elseif {self.expr(2)}, s = s - 1; end",
            f"q = 0; while q < 2 && ({self.expr(2)}), q = q + 1; end",
            f"if {self.expr(2)} | {self.expr(1)}, s = s + 10; end",
        ]
        return init + f"\nfor k = 1:{rng.randint(17, 40)}\n" + "\n".join(body) + "\nend"


def test_compiled_conditions_match_the_interpreter():
    rng = random.Random(2468)
    gen = _CondGen(rng)
    check_all(gen.program() for _ in range(count(150)))


def test_conditions_are_compiled():
    m = Interpreter(out=lambda s: None)
    program = parse("raw = {'xa', 'b'}; n = 0; for k = 1:40, j = 0; while j < 1, j = j + 1; end\n"
                    "if isempty(strfind(raw{mod(k, 2) + 1}, 'a')) == 0, n = n + 1; end\nend")
    m.run_program(program, m.base)
    cond = program.body[2].body[2].clauses[0][0]
    info = cond._jit_cond
    assert info and info["bails"] == 0 and any(c is not None for c in info["compiled"].values())
    assert np.asarray(val(m, "n")).item() == 20
