"""Linear algebra built-ins: output classes of reductions, options of the factorizations and solvers.

Expected values are closed forms or identities that must hold for any correct implementation.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    """Run MATLAB code made of assert(...) statements."""
    m.run(code)


# -- output class of sum and prod -----------------------------------------------------------------

def test_sum_of_integers_is_double_unless_native(m):
    ok(m, """
        assert(sum(uint8([200 200 200])) == 600 && isa(sum(uint8([200 200 200])), 'double'));
        assert(isa(sum(int16([1 2])), 'double') && isa(sum(true(1, 3)), 'double') && sum('ab') == 195);
        assert(isa(sum(single([1 2])), 'single') && isa(sum(single([1 2]), 'double'), 'double'));
        assert(isequal(sum(uint8([200 200 200]), 'native'), uint8(255)));
        assert(isequal(sum(int8([-100 -100]), 'native'), int8(-128)));
        assert(isequal(sum([true true], 'native'), true) && islogical(sum([true false], 'native')));
        assert(isequal(sum(uint8([1 2; 3 4]), 2), [3; 7]) && isa(sum(uint8([1 2; 3 4]), 2), 'double'));
        assert(isequal(sum(int8([1 2; 3 4]), 'all', 'native'), int8(10)));
    """)


def test_prod_of_integers_is_double_unless_native(m):
    ok(m, """
        assert(prod(uint8([20 20])) == 400 && isa(prod(uint8([20 20])), 'double'));
        assert(isequal(prod(uint8([20 20]), 'native'), uint8(255)));
        assert(isa(prod(single([2 3])), 'single') && prod(int8([])) == 1);
    """)


def test_other_reductions_keep_their_documented_class(m):
    ok(m, """
        assert(isequal(cumsum(int8([100 100])), int8([100 127])));
        assert(isequal(cumprod(uint8([20 20])), uint8([20 255])));
        assert(isa(mean(int8([1 2])), 'double') && mean(int8([1 2])) == 1.5);
        assert(isequal(median(int8([1 2 4 5])), int8(3)) && isequal(max(uint8([3 9 2])), uint8(9)));
        assert(isequal(diff(uint8([5 3 9])), uint8([0 6])) && isa(mode(int8([1 1 2])), 'int8'));
    """)


def test_sum_rejects_conflicting_or_unknown_class_options(m):
    with pytest.raises(MatlabError):
        m.run("sum([1 2], 'native', 'double')")
    with pytest.raises(MatlabError):
        m.run("sum('ab', 'native')")


# -- cond, norm, rank, pinv -----------------------------------------------------------------------

def test_cond_honours_the_norm_type(m):
    # A = [1 2; 3 4], inv(A) = [-2 1; 1.5 -0.5]
    ok(m, """
        A = [1 2; 3 4];
        assert(abs(cond(A, 1) - 6 * 3.5) < 1e-12);
        assert(abs(cond(A, inf) - 7 * 3) < 1e-12);
        assert(abs(cond(A, 'fro') - sqrt(30) * sqrt(7.5)) < 1e-12);
        s = svd(A); assert(abs(cond(A) - s(1) / s(2)) < 1e-12 && abs(cond(A, 2) - cond(A)) < 1e-14);
        assert(cond([1 2; 2 4]) > 1e15 && isinf(cond([1 2; 2 4], 1)));
        assert(abs(cond([1 0; 0 2; 0 0]) - 2) < 1e-14);
    """)
    with pytest.raises(MatlabError):
        m.run("cond([1 2; 3 4], 3)")
    with pytest.raises(MatlabError):
        m.run("cond([1 2 3; 4 5 6], 1)")


def test_norm_types(m):
    ok(m, """
        assert(norm([3 4]) == 5 && norm([1 -2 3], 1) == 6 && norm([1 -2 3], inf) == 3 && norm([1 -2 3], -inf) == 1);
        assert(abs(norm([1 2 3], 3) - 36^(1/3)) < 1e-14 && norm([1 2; 3 4], 1) == 6 && norm([1 2; 3 4], inf) == 7);
        assert(abs(norm([1 2; 3 4], 'fro') - sqrt(30)) < 1e-14 && isa(norm(single([3 4])), 'single'));
    """)
    with pytest.raises(MatlabError):
        m.run("norm([1 2; 3 4], 3)")
    with pytest.raises(MatlabError):
        m.run("norm([1 2; 3 4], 'bad')")


def test_rank_and_pinv_tolerances(m):
    ok(m, """
        A = [1 0; 0 1e-6];
        assert(rank(A) == 2 && rank(A, 1e-3) == 1);
        assert(isequal(pinv(A, 1e-3), [1 0; 0 0]) && norm(pinv(A) - diag([1 1e6])) < 1e-6);
        B = [1 2; 2 4]; assert(norm(pinv(B) - B / 25) < 1e-14 && rank(magic(4)) == 3);
    """)


# -- eig ---------------------------------------------------------------------------------------------

def test_eig_left_vectors_and_output_forms(m):
    ok(m, """
        A = [2 1; 0 3];
        [V, D, W] = eig(A);
        assert(norm(A * V - V * D) < 1e-12 && norm(W' * A - D * W') < 1e-12);
        e = eig(A, 'matrix'); assert(isequal(size(e), [2 2]) && isdiag(e));
        [V, d] = eig(A, 'vector'); assert(isequal(size(d), [2 1]) && norm(A * V - V * diag(d)) < 1e-12);
        assert(isequal(eig([2 0; 0 1]), [1; 2]));
    """)


def test_eig_nobalance_and_exact_symmetry_test(m):
    ok(m, """
        A = [1 100; 1e-4 1];
        [V1, D1] = eig(A); [V2, D2] = eig(A, 'nobalance');
        assert(norm(sort(diag(D1)) - sort(diag(D2))) < 1e-10 && norm(A * V2 - V2 * D2) < 1e-10);
        assert(abs(norm(V2(:, 1)) - 1) < 1e-12);
        % a nearly (not exactly) symmetric matrix must not be treated as symmetric
        B = [1 1e-9; 0 1]; [V, D] = eig(B); assert(norm(B * V - V * D) < 1e-15);
    """)


def test_generalized_eig(m):
    ok(m, """
        A = [2 1; 1 3]; B = [2 0; 0 1];
        [V, D] = eig(A, B);
        assert(norm(A * V - B * V * D) < 1e-12 && norm(V' * B * V - eye(2)) < 1e-12);
        assert(issorted(diag(D)));
        [V2, D2] = eig(A, B, 'qz'); assert(norm(A * V2 - B * V2 * D2) < 1e-12);
        assert(norm(sort(diag(D2)) - diag(D)) < 1e-12);
        [V3, D3, W3] = eig([1 2; 3 4], [2 0; 0 1]);
        assert(norm(W3' * [1 2; 3 4] - D3 * W3' * [2 0; 0 1]) < 1e-12);
        e = eig([1 0; 0 1], [1 0; 0 0]); assert(any(isinf(e)) && any(abs(e - 1) < 1e-14));
    """)
    with pytest.raises(MatlabError):
        m.run("eig([1 2; 3 4], 'chol')")
    with pytest.raises(MatlabError):
        m.run("eig([1 2; 3 4], 'vector', 'matrix')")
    with pytest.raises(MatlabError):
        m.run("eig([1 NaN; 3 4])")


# -- svd, qr, lu, chol ----------------------------------------------------------------------------------

def test_svd_forms(m):
    ok(m, """
        A = [1 2; 3 4; 5 6];
        [U, S, V] = svd(A); assert(isequal(size(U), [3 3]) && isequal(size(S), [3 2]) && norm(U * S * V' - A) < 1e-12);
        [U, S, V] = svd(A, 'econ'); assert(isequal(size(U), [3 2]) && isequal(size(S), [2 2]));
        [U, S, V] = svd(A, 0); assert(isequal(size(U), [3 2]));
        [U, S, V] = svd(A', 0); assert(isequal(size(U), [2 2]) && isequal(size(S), [2 3]));
        [U, s, V] = svd(A, 'econ', 'vector'); assert(isequal(size(s), [2 1]) && norm(U * diag(s) * V' - A) < 1e-12);
        S = svd(A, 'matrix'); assert(isequal(size(S), [3 2]));
        assert(abs(norm(svd([3 0; 0 4]) - [4; 3])) < 1e-14);
    """)
    with pytest.raises(MatlabError):
        m.run("svd([1 2; 3 4], 'bogus')")


def test_qr_forms(m):
    ok(m, """
        A = [1 2; 3 4; 5 6];
        [Q, R] = qr(A); assert(isequal(size(Q), [3 3]) && norm(Q * R - A) < 1e-12 && istriu(R));
        [Q, R] = qr(A, 'econ'); assert(isequal(size(Q), [3 2]) && isequal(size(R), [2 2]));
        [Q, R, P] = qr(A); assert(norm(Q * R - A * P) < 1e-12 && isequal(sort(P * [1; 2])', [1 2]));
        [Q, R, p] = qr(A, 0); assert(isequal(size(Q), [3 2]) && norm(Q * R - A(:, p)) < 1e-12 && isrow(p));
        [Q, R, p] = qr(A, 'vector'); assert(norm(Q * R - A(:, p)) < 1e-12);
        d = abs(diag(R)); assert(d(1) >= d(2));
        R1 = qr(A); [~, R2] = qr(A); assert(istriu(R1) && norm(abs(R1) - abs(R2)) < 1e-12);
    """)


def test_lu_forms(m):
    ok(m, """
        A = [1 2 3; 4 5 6; 7 8 10];
        [L, U, P] = lu(A); assert(norm(P * A - L * U) < 1e-12 && istril(L) && istriu(U));
        [L2, U2] = lu(A); assert(norm(L2 * U2 - A) < 1e-12 && norm(L2 - P' * L) < 1e-14);
        [L3, U3, p] = lu(A, 'vector'); assert(isrow(p) && norm(A(p, :) - L3 * U3) < 1e-12);
        Y = lu(A); assert(norm(Y - (U + L - eye(3))) < 1e-14);
        B = [1 2; 3 4; 5 6]; [L, U, P] = lu(B); assert(isequal(size(L), [3 2]) && norm(P * B - L * U) < 1e-12);
    """)


def test_chol_forms_and_failure_flag(m):
    ok(m, """
        S = [4 2; 2 3];
        R = chol(S); assert(norm(R' * R - S) < 1e-14 && istriu(R));
        L = chol(S, 'lower'); assert(norm(L * L' - S) < 1e-14 && istril(L));
        % only the upper triangle is read
        assert(isequal(chol([4 2; 99 3]), R));
        [R, p] = chol([1 2; 2 1]); assert(p == 2 && isequal(R, 1));
        [R, p] = chol([4 2 1; 2 3 1; 1 1 -5]); assert(p == 3 && norm(R' * R - [4 2; 2 3]) < 1e-14);
        [R, p] = chol(S); assert(p == 0);
    """)
    with pytest.raises(MatlabError):
        m.run("chol([1 2; 2 1])")


# -- solving ---------------------------------------------------------------------------------------------

def test_backslash_gives_a_basic_solution_for_rectangular_systems(m):
    # an underdetermined system: the solution has at most rank(A) nonzeros (not the minimum-norm one)
    ok(m, """
        x = [1 2] \\ 3; assert(isequal(size(x), [2 1]) && nnz(x) == 1 && abs([1 2] * x - 3) < 1e-14);
        assert(norm(x - [0; 1.5]) < 1e-14);
        A = [1 1; 1 2; 1 3]; y = [1; 2; 2]; c = A \\ y; assert(norm(c - [2/3; 1/2]) < 1e-12);
        assert(norm(lsqminnorm([1 2], 3) - [0.6; 1.2]) < 1e-14);
    """)


def test_linsolve_options(m):
    ok(m, """
        x = linsolve([2 0; 1 3], [2; 5], struct('LT', true)); assert(norm(x - [1; 4/3]) < 1e-14);
        x = linsolve([2 1; 0 3], [4; 3], struct('UT', true)); assert(norm(x - [1.5; 1]) < 1e-14);
        x = linsolve([2 1; 0 3], [2; 5], struct('UT', true, 'TRANSA', true));
        assert(norm([2 1; 0 3]' * x - [2; 5]) < 1e-14);
        x = linsolve([4 1; 1 3], [1; 2], struct('SYM', true, 'POSDEF', true));
        assert(norm([4 1; 1 3] * x - [1; 2]) < 1e-14);
        [x, r] = linsolve([1 2; 3 4], [1; 2]); assert(norm([1 2; 3 4] * x - [1; 2]) < 1e-14 && abs(r - 1 / 21) < 1e-12);
        x = linsolve([1 2; 3 4], [5; 6], struct('TRANSA', true)); assert(norm([1 3; 2 4] * x - [5; 6]) < 1e-13);
    """)
    with pytest.raises(MatlabError):
        m.run("linsolve([1 2; 3 4], [1; 2], struct('BOGUS', true))")


def test_null_and_orth_options(m):
    ok(m, """
        N = null([1 1; 1 1]); assert(isequal(size(N), [2 1]) && abs(norm(N) - 1) < 1e-14 && norm([1 1] * N) < 1e-14);
        Z = null([1 2 3; 4 5 6], 'rational'); assert(norm(Z - [1; -2; 1]) < 1e-12);
        assert(isequal(size(null([1 0; 0 1e-6], 1e-3)), [2 1]) && isempty(null(eye(2))));
        Q = orth([1 1; 1 1]); assert(isequal(size(Q), [2 1]) && abs(abs(Q(1)) - 1 / sqrt(2)) < 1e-14);
        assert(isequal(size(orth([1 0; 0 1e-6], 1e-3)), [2 1]));
    """)


def test_matrix_function_outputs(m):
    ok(m, """
        [X, res] = sqrtm([4 0; 0 9]); assert(isequal(X, [2 0; 0 3]) && res < 1e-14);
        [X, alpha, condx] = sqrtm([4 1; 1 3]); assert(norm(X * X - [4 1; 1 3]) < 1e-13 && alpha > 0 && condx > 0);
        % for a scalar a the stability factor is 1 and the relative condition of sqrt is 1/2
        [x, alpha, condx] = sqrtm(4); assert(x == 2 && abs(alpha - 1) < 1e-14 && abs(condx - 0.5) < 1e-14);
        [L, flag] = logm([1 2; 0 1]); assert(norm(L - [0 2; 0 0]) < 1e-12 && flag == 0);
        assert(norm(expm([0 1; 0 0]) - [1 1; 0 1]) < 1e-14);
    """)


def test_special_matrices(m):
    ok(m, """
        assert(isequal(toeplitz([1 2 3]), [1 2 3; 2 1 2; 3 2 1]));
        assert(isequal(toeplitz([1 2 3], [1 5 6]), [1 5 6; 2 1 5; 3 2 1]));
        T = toeplitz([1 2+1i]); assert(isequal(T, [1 2+1i; 2-1i 1]));
        assert(isequal(pascal(4, 1) * pascal(4, 1), eye(4)));
        P2 = pascal(3, 2); assert(norm(P2^3 - eye(3)) < 1e-12);
        H = hilb(3); assert(isa(hilb(3, 'single'), 'single') && abs(H(2, 3) - 0.25) < 1e-15);
        assert(isequal(kron([1 2], [1; 1], [1 2]), [1 2 2 4; 1 2 2 4]));
    """)


def test_num_helper_still_reads_plain_values(m):
    np.testing.assert_allclose(m.num("cond([1 0; 0 2])"), [[2.0]])
