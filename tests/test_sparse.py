"""Sparse matrices: results are checked against dense NumPy / SciPy computations."""

import time

import numpy as np
import pytest

from mylab.builtins.sparse import SparseMatrix
from mylab.errors import MatlabError

SETUP = """
A = sparse([1 2 3 1 4 2],[1 2 3 3 4 1],[4 5 6 7 8 -2],4,5);
B = sparse([1 3 4 2],[2 2 5 1],[1 2 3 9],4,5);
D = reshape(1:20,4,5);
Q = sparse([4 1 0 0;1 4 1 0;0 1 4 1;0 0 1 4]);
"""
A = np.zeros((4, 5))
for _i, _j, _v in zip([1, 2, 3, 1, 4, 2], [1, 2, 3, 3, 4, 1], [4, 5, 6, 7, 8, -2]):
    A[_i - 1, _j - 1] += _v
B = np.zeros((4, 5))
for _i, _j, _v in zip([1, 3, 4, 2], [2, 2, 5, 1], [1, 2, 3, 9]):
    B[_i - 1, _j - 1] += _v
D = np.arange(1, 21, dtype=float).reshape(5, 4).T
Q = np.array([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]], dtype=float)


@pytest.fixture
def s(m):
    m.run(SETUP)
    return m


def full(s, expr):
    return np.asarray(s.num(f"full({expr})"), dtype=float)


def is_sparse(s, expr):
    return bool(s.num(f"issparse({expr})").item())


# -- construction ---------------------------------------------------------------------------------

def test_sparse_is_a_real_sparse_object(s):
    v = s.val("A")
    assert isinstance(v, SparseMatrix)
    assert v.m.nnz == 6
    assert s.str("class(A)") == "double"
    assert is_sparse(s, "A") and not is_sparse(s, "full(A)")
    np.testing.assert_array_equal(s.num("size(A)"), [[4, 5]])
    assert s.num("nnz(A)").item() == 6
    assert s.num("numel(A)").item() == 20


def test_constructors(s):
    np.testing.assert_array_equal(full(s, "A"), A)
    np.testing.assert_array_equal(full(s, "sparse(D)"), D)
    np.testing.assert_array_equal(full(s, "sparse(3,2)"), np.zeros((3, 2)))
    # duplicates are added up
    np.testing.assert_array_equal(full(s, "sparse([1 1 2],[1 1 2],[1 2 3])"), [[3, 0], [0, 3]])
    np.testing.assert_array_equal(full(s, "sparse([2 3],[1 1],5)"), [[0], [5], [5]])
    np.testing.assert_array_equal(full(s, "speye(3)"), np.eye(3))
    np.testing.assert_array_equal(full(s, "speye(2,3)"), np.eye(2, 3))
    np.testing.assert_array_equal(full(s, "spones(A)"), (A != 0).astype(float))
    np.testing.assert_array_equal(full(s, "spalloc(2,3,10)"), np.zeros((2, 3)))
    np.testing.assert_array_equal(full(s, "spconvert([1 1 5; 2 3 6])"), [[5, 0, 0], [0, 0, 6]])
    assert s.str("class(sparse([1 2],[1 2],[true true]))") == "logical"


def test_constructor_errors(s):
    for code in ("sparse([1 2],[1 2],[1 2 3])", "sparse(0,1,1)", "sparse(1.5,1,1)", "sparse(3,3,1,2,2)"):
        with pytest.raises(MatlabError):
            s.run(code + ";")


def test_spdiags_build_and_extract(s):
    np.testing.assert_array_equal(
        full(s, "spdiags([1 2 3;4 5 6;7 8 9],[-1 0 1],3,3)"), [[2, 6, 0], [1, 5, 9], [0, 4, 8]])
    # more columns than rows: the row index of B follows the row of A
    np.testing.assert_array_equal(
        full(s, "spdiags([1 2 3;4 5 6;7 8 9],[-1 0 2],3,5)"),
        [[2, 0, 3, 0, 0], [4, 5, 0, 6, 0], [0, 7, 8, 0, 9]])
    np.testing.assert_array_equal(s.num("spdiags(Q)"), [[1, 4, 0], [1, 4, 1], [1, 4, 1], [0, 4, 1]])
    np.testing.assert_array_equal(s.num("spdiags(Q, [0 1])"), [[4, 0], [4, 1], [4, 1], [4, 1]])
    s.run("[Bd, d] = spdiags(Q);")
    np.testing.assert_array_equal(s.num("d"), [[-1], [0], [1]])
    np.testing.assert_array_equal(full(s, "spdiags(Bd, d, 4, 4)"), Q)
    np.testing.assert_array_equal(full(s, "spdiags([9;9;9;9], 0, Q)"), Q - 4 * np.eye(4) + 9 * np.eye(4))


def test_random_sparse(s):
    s.run("rng(1); R = sprand(50, 40, 0.1); N = sprandn(R); Y = sprandsym(30, 0.2);")
    assert s.num("nnz(R)").item() == 200
    r = full(s, "R")
    assert r.min() >= 0 and r.max() < 1
    np.testing.assert_array_equal(full(s, "N") != 0, r != 0)
    y = full(s, "Y")
    np.testing.assert_array_equal(y, y.T)


# -- arithmetic -------------------------------------------------------------------------------------

@pytest.mark.parametrize("expr, expected, sparse_result", [
    ("A+B", A + B, True), ("A-B", A - B, True), ("A.*B", A * B, True), ("A*B'", A @ B.T, True),
    ("A'*B", A.T @ B, True), ("A+D", A + D, False), ("D-A", D - A, False), ("A+1", A + 1, False),
    ("A.*D", A * D, True), ("2*A", 2 * A, True), ("A*2", 2 * A, True), ("A/2", A / 2, True),
    ("A.^2", A ** 2, True), ("-A", -A, True), ("A'", A.T, True), ("A.'", A.T, True),
    ("A*ones(5,1)", A @ np.ones((5, 1)), False), ("ones(1,4)*A", np.ones((1, 4)) @ A, False),
    ("Q*Q", Q @ Q, True), ("Q^3", Q @ Q @ Q, True), ("A./D", A / D, True), ("abs(A)", np.abs(A), True),
    ("kron(Q, speye(2))", np.kron(Q, np.eye(2)), True), ("kron(eye(2), Q)", np.kron(np.eye(2), Q), True),
    ("plus(A, B)", A + B, True), ("times(A, 3)", 3 * A, True), ("mtimes(Q, Q)", Q @ Q, True),
])
def test_arithmetic(s, expr, expected, sparse_result):
    np.testing.assert_allclose(full(s, expr), expected, atol=1e-12)
    assert is_sparse(s, expr) == sparse_result


@pytest.mark.parametrize("expr, expected", [
    ("A>0", A > 0), ("A>=0", A >= 0), ("A==0", A == 0), ("A~=B", A != B), ("A<B", A < B), ("A<=B", A <= B),
    ("A&B", (A != 0) & (B != 0)), ("A|B", (A != 0) | (B != 0)), ("~A", A == 0), ("A>D", A > D), ("5>A", 5 > A),
])
def test_comparisons_give_logical(s, expr, expected):
    np.testing.assert_array_equal(full(s, expr) != 0, expected)
    assert s.str(f"class({expr})") == "logical"


def test_comparison_stays_sparse_and_cheap(s):
    s.run("Z = speye(200000); P = Z > 0.5; n = nnz(P); M = Z * 3 - Z;")
    assert s.num("n").item() == 200000
    assert is_sparse(s, "P") and is_sparse(s, "M")
    assert s.num("M(77,77)").item() == 2


def test_zero_preserving_functions_stay_sparse(s):
    for fn, ref in (("sin", np.sin), ("sqrt", np.sqrt), ("sign", np.sign), ("floor", np.floor)):
        np.testing.assert_allclose(full(s, f"{fn}(Q)"), ref(Q))
        assert is_sparse(s, f"{fn}(Q)")
    np.testing.assert_allclose(full(s, "spfun(@exp, Q)"), np.where(Q != 0, np.exp(Q), 0))


def test_other_functions_get_the_full_matrix(s):
    np.testing.assert_allclose(s.num("exp(Q)"), np.exp(Q))
    np.testing.assert_allclose(s.num("cumsum(A)"), np.cumsum(A, axis=0))
    np.testing.assert_allclose(s.num("sort(A)"), np.sort(A, axis=0))
    np.testing.assert_allclose(s.num("det(Q)"), np.linalg.det(Q))
    np.testing.assert_allclose(np.sort(s.num("eig(Q)").ravel()), np.linalg.eigvalsh(Q))
    assert s.num("rank(Q)").item() == 4
    assert s.run("fprintf('%g ', Q(1,:));") == "4 1 0 0 "


def test_huge_matrix_is_not_silently_densified(s):
    s.run("Z = sparse(1e5, 1e5); Z(5,7) = 2; y = sin(Z); k = nnz(y);")
    assert s.num("k").item() == 1
    with pytest.raises(MatlabError, match="sparse"):
        s.run("exp(Z);")


# -- reductions and queries ---------------------------------------------------------------------------

def test_reductions(s):
    np.testing.assert_array_equal(full(s, "sum(A)"), A.sum(axis=0, keepdims=True))
    np.testing.assert_array_equal(full(s, "sum(A,2)"), A.sum(axis=1, keepdims=True))
    assert s.num("sum(A(:))").item() == A.sum()
    assert s.num("sum(A,'all')").item() == A.sum()
    assert s.num("sum(sum(A))").item() == A.sum()
    np.testing.assert_allclose(full(s, "mean(A)"), A.mean(axis=0, keepdims=True))
    np.testing.assert_array_equal(full(s, "max(A)"), A.max(axis=0, keepdims=True))
    np.testing.assert_array_equal(full(s, "min(A,[],2)"), A.min(axis=1, keepdims=True))
    np.testing.assert_array_equal(full(s, "max(A, 3)"), np.maximum(A, 3))
    np.testing.assert_array_equal(full(s, "any(A)") != 0, A.any(axis=0, keepdims=True))
    np.testing.assert_array_equal(full(s, "all(A,2)") != 0, A.all(axis=1, keepdims=True))
    s.run("[mx, ix] = max(Q);")
    np.testing.assert_array_equal(full(s, "mx"), [[4, 4, 4, 4]])
    np.testing.assert_array_equal(s.num("ix"), [[1, 2, 3, 4]])
    assert s.num("trace(Q)").item() == 16


def test_find_nonzeros(s):
    s.run("[i, j, v] = find(A); k = find(A); k2 = find(A, 2);")
    rows, cols = np.nonzero(A.T)[::-1]
    np.testing.assert_array_equal(s.num("i").ravel(), rows + 1)
    np.testing.assert_array_equal(s.num("j").ravel(), cols + 1)
    np.testing.assert_array_equal(s.num("v").ravel(), A[rows, cols])
    np.testing.assert_array_equal(s.num("k").ravel(), np.nonzero(A.flatten(order="F"))[0] + 1)
    assert s.num("k2").shape == (2, 1)
    np.testing.assert_array_equal(s.num("nonzeros(A)").ravel(), A.flatten(order="F")[A.flatten(order="F") != 0])
    np.testing.assert_array_equal(s.num("nonzeros([0 3; 4 0])").ravel(), [4, 3])
    assert s.num("nzmax(A)").item() == 6


def test_norms(s):
    for arg, ref in (("1", 1), ("inf", np.inf), ("'fro'", "fro")):
        np.testing.assert_allclose(s.num(f"norm(Q, {arg})"), np.linalg.norm(Q, ref))
    np.testing.assert_allclose(s.num("norm(Q)"), np.linalg.norm(Q, 2))
    np.testing.assert_allclose(s.num("normest(Q)"), np.linalg.norm(Q, 2), rtol=1e-6)
    np.testing.assert_allclose(s.num("norm(Q(:,1))"), np.linalg.norm(Q[:, 0]))
    np.testing.assert_allclose(s.num("norm(Q(:,1), 1)"), np.abs(Q[:, 0]).sum())
    np.testing.assert_allclose(s.num("condest(Q)"), np.linalg.cond(Q, 1), rtol=1e-8)


def test_predicates(s):
    for expr, want in (("isnumeric(A)", 1), ("isfloat(A)", 1), ("isreal(A)", 1), ("islogical(A)", 0),
                       ("islogical(A>0)", 1), ("isa(A,'double')", 1), ("isa(A,'numeric')", 1), ("iscell(A)", 0),
                       ("isempty(sparse(0,3))", 1), ("isempty(A)", 0), ("isvector(A(:,1))", 1),
                       ("isscalar(sparse(5))", 1), ("issymmetric(Q)", 1), ("issymmetric(A)", 0),
                       ("isdiag(speye(3))", 1),
                       ("istril(tril(Q))", 1), ("istriu(Q)", 0), ("isequal(A, full(A))", 1), ("isequal(A, B)", 0),
                       ("isequal(speye(2), eye(2), sparse([1 0;0 1]))", 1), ("isbanded(Q,1,1)", 1),
                       ("isbanded(Q,0,1)", 0)):
        assert bool(s.num(expr).item()) == bool(want), expr
    np.testing.assert_array_equal(s.num("bandwidth(triu(Q))"), [[0]])
    s.run("[lo, up] = bandwidth(Q);")
    assert s.num("lo").item() == 1 and s.num("up").item() == 1
    assert s.num("sprank(Q)").item() == 4


def test_truth_value(s):
    assert s.run("if speye(2), disp('y'), else, disp('n'), end") == "n\n"
    assert s.run("if sparse([1 2;3 4]), disp('y'), else, disp('n'), end") == "y\n"


# -- indexing ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("expr, expected", [
    ("A(2,:)", A[1:2, :]), ("A(:,3)", A[:, 2:3]), ("A([1 3],[1 3])", A[np.ix_([0, 2], [0, 2])]),
    ("A(:)", A.reshape(-1, 1, order="F")), ("A([1 5 9])", A.flatten(order="F")[[0, 4, 8]].reshape(1, -1)),
    ("A(end,:)", A[3:4, :]), ("A(2:end, end)", A[1:, 4:5]), ("A(:, [true false true false true])", A[:, [0, 2, 4]]),
    ("A(A>4)", A.flatten(order="F")[A.flatten(order="F") > 4].reshape(-1, 1)),
    ("A([1 2; 3 4])", A.flatten(order="F")[[0, 1, 2, 3]].reshape(2, 2)),
    ("Q(logical(eye(4)))", np.diag(Q).reshape(-1, 1)), ("A(:,:)", A),
])
def test_index_get(s, expr, expected):
    np.testing.assert_array_equal(full(s, expr), expected)
    assert is_sparse(s, expr)


def test_scalar_index_is_a_plain_number(s):
    assert s.num("A(1,3)").item() == 7
    assert s.num("A(end,end)").item() == 0
    assert s.num("A(5)").item() == A.flatten(order="F")[4]
    s.run("x = zeros(1,3); x(2) = A(1,3); y = zeros(4,2); y(:,2) = Q(:,2);")
    np.testing.assert_array_equal(s.num("x"), [[0, 7, 0]])
    np.testing.assert_array_equal(s.num("y")[:, 1], Q[:, 1])


def test_index_errors(s):
    for code in ("A(5,1)", "A(0,1)", "A(1,2,3)", "A(21)", "A{1}", "A.x"):
        with pytest.raises(MatlabError):
            s.run(code + ";")


def test_index_set(s):
    ref = A.copy()
    s.run("C = A; C(2,3) = 9; C(1,:) = 1:5; C(:,2) = 0; C([3 4],[4 5]) = [1 2;3 4]; C(4,1) = 0;")
    ref[1, 2] = 9
    ref[0, :] = np.arange(1, 6)
    ref[:, 1] = 0
    ref[np.ix_([2, 3], [3, 4])] = [[1, 2], [3, 4]]
    ref[3, 0] = 0
    np.testing.assert_array_equal(full(s, "C"), ref)
    assert is_sparse(s, "C")
    assert s.num("nnz(C)").item() == np.count_nonzero(ref)
    np.testing.assert_array_equal(full(s, "A"), A)  # the original is untouched


def test_index_set_variants(s):
    s.run("C = A; C(C>4) = -1; E = A; E(2) = 11; E(7,7) = 1; F = A; F(1:2,1:2) = sparse([1 2;3 4]); "
          "G = sparse(2,2); G(:) = [1 2 3 4]; H = Q; H(:, end+1) = [1;2;3;4];")
    np.testing.assert_array_equal(full(s, "C"), np.where(A > 4, -1, A))
    e = np.zeros((7, 7))
    e[:4, :5] = A
    e[1, 0] = 11
    e[6, 6] = 1
    np.testing.assert_array_equal(full(s, "E"), e)
    f = A.copy()
    f[:2, :2] = [[1, 2], [3, 4]]
    np.testing.assert_array_equal(full(s, "F"), f)
    np.testing.assert_array_equal(full(s, "G"), [[1, 3], [2, 4]])
    np.testing.assert_array_equal(full(s, "H"), np.column_stack([Q, [1, 2, 3, 4]]))
    with pytest.raises(MatlabError):
        s.run("C(1,:) = [1 2 3];")


def test_value_semantics_of_assignment(s):
    s.run("K = speye(3); K2 = K; K2(1,1) = 7; a = [K(1,1) K2(1,1)];")
    np.testing.assert_array_equal(s.num("a"), [[1, 7]])
    s.run("st.K = K; t = st; t.K(1,1) = 5; b = [st.K(1,1) t.K(1,1)];")
    np.testing.assert_array_equal(s.num("b"), [[1, 5]])
    s.run("c = {K}; K(1,1) = 9; d = [c{1}(1,1) K(1,1)];")
    np.testing.assert_array_equal(s.num("d"), [[1, 9]])
    s.run("f = @() K(1,1); K(1,1) = 11; e = [f() K(1,1)];")
    np.testing.assert_array_equal(s.num("e"), [[9, 11]])
    s.run("function B = bump(M)\n M(1,1) = 42; B = M;\nend\nK = speye(3); B = bump(K); g = [K(1,1) B(1,1)];")
    np.testing.assert_array_equal(s.num("g"), [[1, 42]])


def test_delete(s):
    np.testing.assert_array_equal(full(s, "A"), A)
    s.run("C = A; C(2,:) = []; E = A; E(:,[1 5]) = []; F = A(:,1); F(2) = [];")
    np.testing.assert_array_equal(full(s, "C"), np.delete(A, 1, axis=0))
    np.testing.assert_array_equal(full(s, "E"), np.delete(A, [0, 4], axis=1))
    np.testing.assert_array_equal(full(s, "F"), np.delete(A[:, 0:1], 1, axis=0))
    with pytest.raises(MatlabError):
        s.run("C(1,1) = [];")


def test_concatenation(s):
    np.testing.assert_array_equal(full(s, "[A B]"), np.hstack([A, B]))
    np.testing.assert_array_equal(full(s, "[A; B]"), np.vstack([A, B]))
    np.testing.assert_array_equal(full(s, "[A; D]"), np.vstack([A, D]))
    np.testing.assert_array_equal(full(s, "[Q speye(4); Q Q]"), np.block([[Q, np.eye(4)], [Q, Q]]))
    np.testing.assert_array_equal(full(s, "[Q, []]"), Q)
    np.testing.assert_array_equal(full(s, "cat(1, Q, Q)"), np.vstack([Q, Q]))
    np.testing.assert_array_equal(full(s, "horzcat(Q, Q)"), np.hstack([Q, Q]))
    assert is_sparse(s, "[A; D]") and is_sparse(s, "[1 2 3 4; Q]")
    with pytest.raises(MatlabError):
        s.run("[A; Q];")
    np.testing.assert_array_equal(full(s, "blkdiag(Q, 7)"), np.block([[Q, np.zeros((4, 1))], [np.zeros((1, 4)), 7]]))
    np.testing.assert_array_equal(s.num("blkdiag([1 2], 3)"), [[1, 2, 0], [0, 0, 3]])
    assert is_sparse(s, "blkdiag(Q, 7)") and not is_sparse(s, "blkdiag([1 2], 3)")


def test_shape_functions(s):
    np.testing.assert_array_equal(full(s, "reshape(A, 5, 4)"), A.reshape(5, 4, order="F"))
    np.testing.assert_array_equal(full(s, "reshape(A, 2, [])"), A.reshape(2, 10, order="F"))
    np.testing.assert_array_equal(full(s, "repmat(Q, 2, 3)"), np.tile(Q, (2, 3)))
    np.testing.assert_array_equal(full(s, "diag(Q)"), np.diag(Q).reshape(-1, 1))
    np.testing.assert_array_equal(full(s, "diag(Q, 1)"), np.diag(Q, 1).reshape(-1, 1))
    np.testing.assert_array_equal(full(s, "diag(diag(Q))"), np.diag(np.diag(Q)))
    np.testing.assert_array_equal(full(s, "tril(Q)"), np.tril(Q))
    np.testing.assert_array_equal(full(s, "triu(Q, 1)"), np.triu(Q, 1))
    assert is_sparse(s, "reshape(A, 5, 4)") and is_sparse(s, "diag(Q)") and is_sparse(s, "tril(Q)")
    assert s.run("for c = Q, fprintf('%d ', nnz(c)); end") == "2 3 3 2 "


def test_complex_sparse(s):
    s.run("C = sparse([1 2],[1 2],[1+2i 3]); H = C'; T = C.';")
    np.testing.assert_array_equal(np.asarray(s.num("full(H)")), [[1 - 2j, 0], [0, 3]])
    np.testing.assert_array_equal(np.asarray(s.num("full(T)")), [[1 + 2j, 0], [0, 3]])
    assert not bool(s.num("isreal(C)").item())
    assert bool(s.num("isreal(real(C))").item())


# -- solving ---------------------------------------------------------------------------------------------

def test_backslash(s):
    b = np.array([[1.0], [2], [3], [4]])
    np.testing.assert_allclose(s.num("Q\\[1;2;3;4]"), np.linalg.solve(Q, b))
    np.testing.assert_allclose(s.num("Q\\magic(4)"), np.linalg.solve(Q, s.num("magic(4)")))
    np.testing.assert_allclose(s.num("magic(4)/Q"), s.num("magic(4)") @ np.linalg.inv(Q))
    np.testing.assert_allclose(full(s, "Q\\speye(4)"), np.linalg.inv(Q), atol=1e-14)
    np.testing.assert_allclose(full(s, "inv(Q)"), np.linalg.inv(Q), atol=1e-14)
    assert not is_sparse(s, "Q\\[1;2;3;4]") and is_sparse(s, "Q\\speye(4)") and is_sparse(s, "inv(Q)")
    np.testing.assert_allclose(s.num("mldivide(Q, [1;2;3;4])"), np.linalg.solve(Q, b))
    # rectangular: least squares
    r = np.array([[1.0, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(s.num("sparse([1 2;3 4;5 6])\\[1;2;3]"),
                               np.linalg.lstsq(r, np.array([[1.0], [2], [3]]), rcond=None)[0], atol=1e-12)
    # complex
    np.testing.assert_allclose(np.asarray(s.num("(Q + 1i*speye(4))\\[1;2;3;4]")),
                               np.linalg.solve(Q + 1j * np.eye(4), b))


def test_singular_solve_warns(s):
    out = s.run("x = sparse([1 1;1 1])\\[1;2];")
    assert "singular" in out.lower()


def test_poisson_2d_is_solved_sparsely(s):
    """A 22 500-unknown finite-difference Poisson problem: only feasible with real sparse storage."""
    start = time.perf_counter()
    s.run("""
        n = 150; N = n*n; h = 1/(n+1); e = ones(n,1);
        T = spdiags([-e 2*e -e], -1:1, n, n);
        L = (kron(speye(n), T) + kron(T, speye(n))) / h^2;
        [X, Y] = meshgrid((1:n)*h);
        uex = sin(pi*X) .* sin(pi*Y);
        f = 2*pi^2*uex(:);
        u = L \\ f;
        err = max(abs(u - uex(:)));
        res = norm(L*u - f) / norm(f);
        nz = nnz(L);
    """)
    elapsed = time.perf_counter() - start
    assert isinstance(s.val("L"), SparseMatrix)
    assert s.num("N").item() == 22500
    assert s.num("nz").item() == 5 * 22500 - 4 * 150
    assert s.num("res").item() < 1e-10
    assert s.num("err").item() < 1e-4  # second-order discretization error
    assert elapsed < 30, f"sparse solve took {elapsed:.1f}s"


def test_poisson_1d_large(s):
    s.run("""
        n = 200000; e = ones(n,1);
        A = spdiags([-e 2*e -e], -1:1, n, n);
        b = A * (1:n)';
        x = A \\ b;
        err = norm(x - (1:n)') / norm(x);
    """)
    assert s.num("err").item() < 1e-8


def test_assembly_loop_is_fast(s):
    start = time.perf_counter()
    s.run("""
        N = 3000; K = sparse(N, N);
        for k = 1:N-1
            K([k k+1],[k k+1]) = K([k k+1],[k k+1]) + [1 -1; -1 1];
        end
        d = full(diag(K)); nz = nnz(K);
    """)
    assert time.perf_counter() - start < 30
    assert s.num("nz").item() == 3 * 3000 - 2
    d = s.num("d").ravel()
    assert d[0] == 1 and d[-1] == 1 and np.all(d[1:-1] == 2)


def test_triplet_assembly(s):
    s.run("""
        n = 50000; i = [1:n, 2:n, 1:n-1]; j = [1:n, 1:n-1, 2:n];
        v = [2*ones(1,n), -ones(1,n-1), -ones(1,n-1)];
        A = sparse(i, j, v, n, n); u = A \\ ones(n,1); r = norm(A*u - 1) / norm(u);
    """)
    assert s.num("r").item() < 1e-12


def test_factorizations(s):
    s.run("R = chol(Q); Lc = chol(Q, 'lower'); [L, U, P] = lu(Q); [L2, U2] = lu(Q); [L4, U4, P4, Q4] = lu(Q);")
    np.testing.assert_allclose(full(s, "R'*R"), Q, atol=1e-13)
    np.testing.assert_allclose(full(s, "R"), np.linalg.cholesky(Q).T, atol=1e-13)
    np.testing.assert_allclose(full(s, "Lc"), np.linalg.cholesky(Q), atol=1e-13)
    np.testing.assert_allclose(full(s, "L*U"), full(s, "P*Q"), atol=1e-13)
    np.testing.assert_allclose(full(s, "L2*U2"), Q, atol=1e-13)
    np.testing.assert_allclose(full(s, "L4*U4"), full(s, "P4*Q*Q4"), atol=1e-13)
    assert is_sparse(s, "R") and is_sparse(s, "L")
    with pytest.raises(MatlabError, match="positive definite"):
        s.run("chol(sparse([1 2;2 1]));")
    s.run("[~, p] = chol(sparse([1 2;2 1]));")
    assert s.num("p").item() > 0
    s.run("[q, r] = qr(Q);")
    np.testing.assert_allclose(s.num("q*r"), Q, atol=1e-13)


def test_orderings(s):
    for fn in ("symrcm", "amd", "colamd"):
        p = s.num(f"{fn}(Q)").ravel()
        assert sorted(p) == [1, 2, 3, 4]


# -- eigenvalues, singular values, iterative solvers ---------------------------------------------------------

def _lap(s, n=40):
    s.run(f"n = {n}; e = ones(n,1); T = spdiags([-e 2*e -e], -1:1, n, n); "
          "L = kron(speye(n), T) + kron(T, speye(n)); b = ones(n*n, 1);")
    t = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return np.kron(np.eye(n), t) + np.kron(t, np.eye(n))


def test_eigs(s):
    lap = _lap(s)
    w = np.linalg.eigvalsh(lap)
    np.testing.assert_allclose(s.num("eigs(L, 4)").ravel(), w[::-1][:4], rtol=1e-8)
    np.testing.assert_allclose(s.num("eigs(L, 3, 'smallestabs')").ravel(), w[:3], rtol=1e-8)
    np.testing.assert_allclose(s.num("eigs(L, 3, 'sm')").ravel(), w[:3], rtol=1e-8)
    np.testing.assert_allclose(s.num("eigs(L, 2, 'largestreal')").ravel(), w[::-1][:2], rtol=1e-8)
    near = w[np.argsort(np.abs(w - 4.1))[:3]]
    np.testing.assert_allclose(np.sort(s.num("eigs(L, 3, 4.1)").ravel()), np.sort(near), rtol=1e-8)
    assert s.num("eigs(L)").shape == (6, 1)
    s.run("[V, Dg] = eigs(L, 3); r = norm(L*V - V*Dg);")
    assert s.num("r").item() < 1e-8
    assert s.num("Dg").shape == (3, 3)
    # small matrices use the dense path
    np.testing.assert_allclose(s.num("eigs(Q, 2)").ravel(), np.linalg.eigvalsh(Q)[::-1][:2])
    np.testing.assert_allclose(s.num("eigs(Q, 2, 'sm')").ravel(), np.linalg.eigvalsh(Q)[:2])


def test_eigs_nonsymmetric(s):
    s.run("rng(3); n = 500; M = sprandn(n, n, 0.01) + 10*speye(n); d = eigs(M, 3);")
    ref = np.linalg.eigvals(full(s, "M"))
    ref = ref[np.argsort(-np.abs(ref))][:3]
    np.testing.assert_allclose(np.sort(np.abs(np.asarray(s.num("d")).ravel())), np.sort(np.abs(ref)), rtol=1e-6)


def test_svds(s):
    lap = _lap(s, 25)
    sv = np.linalg.svd(lap, compute_uv=False)
    np.testing.assert_allclose(s.num("svds(L, 3)").ravel(), sv[:3], rtol=1e-8)
    np.testing.assert_allclose(s.num("svds(Q, 2)").ravel(), np.linalg.svd(Q, compute_uv=False)[:2])
    np.testing.assert_allclose(s.num("svds(Q, 2, 'smallest')").ravel(), np.linalg.svd(Q, compute_uv=False)[2:])
    s.run("[U, S, V] = svds(L, 4); r = norm(L*V - U*S);")
    assert s.num("r").item() < 1e-8


@pytest.mark.parametrize("call", [
    "pcg(L, b, 1e-10, 500)", "bicgstab(L, b, 1e-10, 500)", "gmres(L, b, 30, 1e-10, 100)", "cgs(L, b, 1e-10, 500)",
    "minres(L, b, 1e-10, 500)", "bicg(L, b, 1e-10, 500)", "pcg(@(x) L*x, b, 1e-10, 500)",
])
def test_krylov_solvers(s, call):
    lap = _lap(s, 20)
    s.run(f"[x, flag, relres, iter] = {call};")
    assert s.num("flag").item() == 0
    assert s.num("relres").item() < 1e-6
    np.testing.assert_allclose(s.num("x").ravel(), np.linalg.solve(lap, np.ones(400)), rtol=1e-5)
    assert s.num("iter").ravel()[0] >= 1


def test_preconditioned_cg(s):
    _lap(s, 40)
    s.run("[x0, f0, r0, it0] = pcg(L, b, 1e-8, 1000); Lc = ichol(L); "
          "[x1, f1, r1, it1, rv] = pcg(L, b, 1e-8, 1000, Lc, Lc'); "
          "[Li, Ui] = ilu(L); [x2, f2] = gmres(L, b, 20, 1e-8, 50, Li, Ui); "
          "[x3, f3] = pcg(L, b, 1e-8, 1000, @(r) Lc'\\(Lc\\r));")
    assert s.num("f0").item() == 0 and s.num("f1").item() == 0 and s.num("f2").item() == 0
    assert s.num("f3").item() == 0
    assert s.num("it1").item() < s.num("it0").item()  # the preconditioner helps
    assert s.num("rv").size >= 1
    np.testing.assert_allclose(s.num("x1"), s.num("x0"), rtol=1e-5)
    np.testing.assert_allclose(s.num("x2"), s.num("x0"), rtol=1e-5)
    assert bool(s.num("istril(Lc)").item())


def test_iteration_limit_and_report(s):
    _lap(s, 30)
    s.run("[x, flag] = pcg(L, b, 1e-12, 3);")
    assert s.num("flag").item() == 1
    assert "converged" in s.run("x = pcg(L, b, 1e-6, 500);")
    assert "without converging" in s.run("x = pcg(L, b, 1e-12, 2);")
    s.run("[x, flag] = lsqr(sparse([1 2;3 4;5 6]), [1;2;3], 1e-12, 100);")
    np.testing.assert_allclose(s.num("x").ravel(), [0, 0.5], atol=1e-8)


# -- display ----------------------------------------------------------------------------------------------

def test_display(s):
    out = s.run("S = sparse([1 2],[1 3],[1.5 -2])")
    assert out == ("S =\n\n  2×3 sparse double matrix (2 nonzeros)\n\n"
                   "   (1,1)      1.5000\n   (2,3)     -2.0000\n\n")
    assert s.run("Z = sparse(2,3)") == "Z =\n\n   All zero sparse: 2×3\n\n"
    assert "sparse logical matrix (1 nonzero)" in s.run("S > 1")
    assert s.run("disp(sparse([0 3]))") == "   (1,2)       3\n"
    assert "{2×3 sparse double}" in s.run("c = {S}")
    assert "[2×3 sparse double]" in s.run("st.S = S")
    assert "2x3" in s.run("whos S")


def test_spy_draws_the_pattern(s):
    s.run("spy(Q); ax = gca; n = numel(ax.Children); close all")
    assert s.num("n").item() >= 1


def test_objects_keep_sparse_values(s):
    s.run("st = struct('K', Q); c = {Q, 1}; f = cellfun(@issparse, c); g = issparse(st.K); "
          "function y = twice(x)\n y = 2*x;\nend\nh = issparse(twice(Q));")
    np.testing.assert_array_equal(s.num("f"), [[1, 0]])
    assert s.num("g").item() == 1 and s.num("h").item() == 1


def test_sparse_logical_index_into_full_array(s):
    np.testing.assert_array_equal(s.num("D(A > 4)").ravel(), D.flatten(order="F")[A.flatten(order="F") > 4])
    s.run("v = 1:3; w = v(sparse([true false true]));")
    np.testing.assert_array_equal(s.num("w"), [[1, 3]])
