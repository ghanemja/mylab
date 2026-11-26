"""Functions of mylab.builtins.linalg_more: factorizations, updates, least squares, matrix functions,
page-wise operations, test matrices and iterative solvers.

Expected values are defining identities (for example U*T*U' = A) or closed forms.
"""

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


def test_schur_real_and_complex_forms(m):
    ok(m, """
        A = [1 2; -2 1];                      % eigenvalues 1 +- 2i
        [U, T] = schur(A); assert(isreal(T) && norm(U * T * U' - A) < 1e-12 && norm(U' * U - eye(2)) < 1e-12);
        assert(T(2, 1) ~= 0);                 % the real form keeps a 2-by-2 block
        [U, T] = schur(A, 'complex'); assert(istriu(T) && norm(U * T * U' - A) < 1e-12);
        assert(norm(sort(imag(diag(T))) - [-2; 2]) < 1e-12);
        T1 = schur(magic(4)); assert(isequal(size(T1), [4 4]) && abs(trace(T1) - 34) < 1e-10);
        [U, T] = schur([1 2i; -2i 3]); assert(norm(U * T * U' - [1 2i; -2i 3]) < 1e-12);
    """)
    with pytest.raises(MatlabError):
        m.run("schur([1 2 3; 4 5 6])")


def test_ordschur_and_ordeig(m):
    ok(m, """
        A = [1 5 2; 0 -2 3; 0 0 4];
        [U, T] = schur(A); e = ordeig(T); assert(norm(e - [1; -2; 4]) < 1e-12);
        [US, TS] = ordschur(U, T, 'lhp');
        assert(norm(US * TS * US' - A) < 1e-11 && abs(TS(1, 1) + 2) < 1e-12);
        [US, TS] = ordschur(U, T, [false false true]); assert(abs(TS(1, 1) - 4) < 1e-12);
        [US, TS] = ordschur(U, T, [1 3 2]);   % the largest cluster number comes first
        assert(norm(diag(TS) - [-2; 4; 1]) < 1e-11 && norm(US * TS * US' - A) < 1e-11);
    """)


def test_qz_and_ordqz(m):
    ok(m, """
        A = [1 2 3; 4 5 6; 7 8 10]; B = [2 0 1; 1 3 0; 0 1 4];
        [AA, BB, Q, Z] = qz(A, B);
        assert(norm(Q * A * Z - AA) < 1e-11 && norm(Q * B * Z - BB) < 1e-11 && istriu(AA) && istriu(BB));
        assert(norm(Q * Q' - eye(3)) < 1e-12 && norm(Z * Z' - eye(3)) < 1e-12);
        e = sort(real(ordeig(AA, BB))); assert(norm(e - sort(real(eig(A, B)))) < 1e-10);
        [AA, BB, Q, Z] = qz(A, B, 'real'); assert(isreal(AA) && norm(Q * A * Z - AA) < 1e-11);
        [AS, BS, QS, ZS] = ordqz(AA, BB, Q, Z, 'udo');
        assert(norm(QS * A * ZS - AS) < 1e-10 && norm(QS * B * ZS - BS) < 1e-10);
        r = abs(ordeig(AS, BS)); assert(r(1) > 1 && all(diff(r > 1) <= 0));
        [AA, BB, Q, Z, V, W] = qz(A, B); lam = ordeig(AA, BB);
        assert(min(svd(A - lam(1) * B)) < 1e-10 && isequal(size(V), [3 3]) && isequal(size(W), [3 3]));
    """)


def test_hess_and_balance(m):
    ok(m, """
        A = magic(4);
        [P, H] = hess(A); assert(norm(P * H * P' - A) < 1e-10 && norm(tril(H, -2)) < 1e-12);
        assert(norm(hess(A) - H) < 1e-12);
        B = [2 0 1 3; 1 3 0 1; 0 1 4 2; 5 1 2 7];
        [AA, BB, Q, Z] = hess(A, B);
        assert(norm(Q * A * Z - AA) < 1e-10 && norm(Q * B * Z - BB) < 1e-10);
        assert(norm(tril(AA, -2)) < 1e-12 && norm(tril(BB, -1)) < 1e-12 && norm(Q * Q' - eye(4)) < 1e-12);
        C = [1 100 1e4; 0.01 1 100; 1e-4 0.01 1];
        [T, Bal] = balance(C); assert(norm(T \\ C * T - Bal) < 1e-10 && cond(Bal) <= cond(C) * (1 + 1e-12));
        [S, Pv, B3] = balance(C); assert(norm(B3 - Bal) < 1e-12 && isequal(sort(Pv)', 1:3));
        assert(norm(B3(Pv, Pv) - diag(1 ./ S) * C * diag(S)) < 1e-10);
        assert(norm(sort(eig(Bal)) - sort(eig(C))) < 1e-8 && isequal(size(balance(C, 'noperm')), [3 3]));
    """)


def test_cholupdate_update_and_downdate(m):
    ok(m, """
        A = [4 2; 2 3]; R = chol(A); x = [1; 1];
        R1 = cholupdate(R, x); assert(istriu(R1) && norm(R1' * R1 - (A + x * x')) < 1e-12);
        R2 = cholupdate(R1, x, '-'); assert(norm(R2 - R) < 1e-12);
        [R3, p] = cholupdate(R, [5; 5], '-'); assert(p == 1);
        H = [4 1+1i; 1-1i 3]; Rh = chol(H); z = [1i; 2];
        Rz = cholupdate(Rh, z); assert(norm(Rz' * Rz - (H + z * z')) < 1e-12);
    """)
    with pytest.raises(MatlabError):
        m.run("R = chol([4 2; 2 3]); cholupdate(R, [5; 5], '-')")


def test_qr_updates(m):
    ok(m, """
        A = magic(4); [Q, R] = qr(A); u = [1; 0; 2; 0]; v = [0; 1; 0; 3];
        [Q1, R1] = qrupdate(Q, R, u, v); assert(norm(Q1 * R1 - (A + u * v')) < 1e-11 && norm(tril(R1, -1)) < 1e-12);
        c = [1; 2; 3; 4];
        [Q1, R1] = qrinsert(Q, R, 2, c); assert(norm(Q1 * R1 - [A(:, 1) c A(:, 2:4)]) < 1e-11);
        [Q1, R1] = qrinsert(Q, R, 3, [9 8 7 6], 'row'); assert(norm(Q1 * R1 - [A(1:2, :); 9 8 7 6; A(3:4, :)]) < 1e-11);
        [Q1, R1] = qrdelete(Q, R, 3); assert(norm(Q1 * R1 - A(:, [1 2 4])) < 1e-11);
        [Q1, R1] = qrdelete(Q, R, 1, 'row'); assert(norm(Q1 * R1 - A(2:4, :)) < 1e-11);
    """)


def test_rotations_and_subspace(m):
    ok(m, """
        [G, y] = planerot([3; 4]); assert(norm(G - [0.6 0.8; -0.8 0.6]) < 1e-15 && norm(y - [5; 0]) < 1e-15);
        [G, y] = planerot([2; 0]); assert(isequal(G, eye(2)) && isequal(y, [2; 0]));
        G = givens(1, 1); assert(norm(G * [1; 1] - [sqrt(2); 0]) < 1e-15);
        [c, s] = givens(3, 4); assert(abs(c - 0.6) < 1e-15 && abs(s - 0.8) < 1e-15);
        assert(abs(subspace([1; 0], [1; 1]) - pi / 4) < 1e-14 && abs(subspace([1; 0], [0; 1]) - pi / 2) < 1e-14);
        assert(subspace([1 0; 0 1; 0 0], [1; 1; 0]) < 1e-14);
    """)


def test_lscov_weights_and_covariance(m):
    # the weighted problem is checked against its normal equations
    ok(m, """
        A = [1 0; 0 1; 1 1; 1 2]; b = [1; 2; 3; 5.5]; w = [1; 2; 1; 1];
        x = lscov([1 0; 0 1; 1 1], [1; 2; 3]); assert(norm(x - [1; 2]) < 1e-12);
        [x, stdx, mse, S] = lscov(A, b, w);
        W = diag(w); xr = (A' * W * A) \\ (A' * W * b); r = b - A * xr;
        assert(norm(x - xr) < 1e-12 && abs(mse - r' * W * r / 2) < 1e-12);
        assert(norm(S - inv(A' * W * A) * mse) < 1e-12 && norm(stdx - sqrt(diag(S))) < 1e-12);
        V = [2 1 0 0; 1 3 1 0; 0 1 4 1; 0 0 1 2];
        xv = lscov(A, b, V); assert(norm(xv - (A' / V * A) \\ (A' / V * b)) < 1e-12);
        assert(norm(lscov(A, b, V, 'orth') - xv) < 1e-8);
    """)


def test_lsqminnorm(m):
    ok(m, """
        assert(norm(lsqminnorm([1 2; 2 4], [1; 2]) - [0.2; 0.4]) < 1e-14);
        A = [1 2 3; 4 5 6]; b = [1; 2]; x = lsqminnorm(A, b);
        assert(norm(A * x - b) < 1e-13 && norm(x - pinv(A) * b) < 1e-13);
        assert(norm(lsqminnorm([1 0; 0 1e-8], [1; 1], 1e-4) - [1; 0]) < 1e-12);
    """)


def test_decomposition_objects(m):
    ok(m, """
        A = [4 3 1; 6 3 2; 1 1 1]; b = [1; 2; 3];
        dA = decomposition(A); assert(strcmp(dA.Type, 'lu') && isequal(dA.MatrixSize, [3 3]));
        assert(norm(A * (dA \\ b) - b) < 1e-12 && norm((b' / dA) * A - b') < 1e-12);
        assert(norm(A' * (dA' \\ b) - b) < 1e-12 && ~isIllConditioned(dA));
        d1 = decomposition([4 1; 1 3]); d2 = decomposition([2 1; 1 -3]);
        assert(strcmp(d1.Type, 'chol') && strcmp(d2.Type, 'ldl'));
        d3 = decomposition([1 2; 0 3]); d4 = decomposition([1 2; 3 4; 5 6]);
        assert(strcmp(d3.Type, 'triangular') && strcmp(d4.Type, 'qr'));
        dQ = decomposition([1 2; 3 4; 5 6]); assert(norm(dQ \\ [1; 2; 4] - [1 2; 3 4; 5 6] \\ [1; 2; 4]) < 1e-12);
        dC = decomposition(magic(4), 'cod', 'CheckCondition', false);
        assert(rank(dC) == 3 && norm(dC \\ (1:4)' - pinv(magic(4)) * (1:4)') < 1e-10);
        d2 = 2 * dA; assert(norm((d2 \\ b) - (dA \\ b) / 2) < 1e-14 && d2.ScaleFactor == 2);
        assert(isa(dA, 'decomposition') && abs(rcond(dA) - rcond(A)) < 1e-14);
    """)
    with pytest.raises(MatlabError):
        m.run("decomposition([1 2; 3 4], 'chol')")
    with pytest.raises(MatlabError):
        m.run("decomposition([1 2; 3 4], 'NoSuchOption', 1)")


def test_funm_matches_closed_forms(m):
    ok(m, """
        assert(norm(funm([1 0; 0 2], @exp) - diag(exp([1 2]))) < 1e-14);
        J = [2 1; 0 2];                       % a Jordan block: f(J) = [f(2) f'(2); 0 f(2)]
        assert(norm(funm(J, @exp) - exp(2) * [1 1; 0 1]) < 1e-12);
        assert(norm(funm(J, @log) - [log(2) 0.5; 0 log(2)]) < 1e-12);
        assert(norm(funm(J, @sin) - [sin(2) cos(2); 0 sin(2)]) < 1e-12);
        A = [1 2; 3 4];
        assert(norm(funm(A, @cos)^2 + funm(A, @sin)^2 - eye(2)) < 1e-10);
        assert(norm(funm(A, @(x, k) exp(x)) - expm(A)) < 1e-10 * norm(expm(A)));
        [F, flag, out] = funm(J, @exp); assert(flag == 0 && isfield(out, 'terms') && isreal(F));
    """)


def test_tensorprod(m):
    ok(m, """
        assert(isequal(tensorprod([1 2; 3 4], [1; 1], 2, 1), [3; 7]));
        assert(isequal(size(tensorprod([1 2], [3 4])), [1 2 1 2]));
        assert(tensorprod([1 2; 3 4], [1 2; 3 4], 'all') == 30);
        A = reshape(1:24, 2, 3, 4); B = reshape(1:60, 3, 5, 4);
        C = tensorprod(A, B, [2 3], [1 3]); D = zeros(2, 5);
        for k = 1:4, D = D + A(:, :, k) * B(:, :, k); end
        assert(isequal(C, D) && isequal(tensorprod([1 2 3], [4 5 6], 2), 32));
    """)


def test_page_functions(m):
    ok(m, """
        A = reshape(1:12, 2, 3, 2) + 0; B = reshape(1:12, 3, 2, 2) + 0;
        C = pagemtimes(A, B); assert(isequal(size(C), [2 2 2]) && isequal(C(:, :, 2), A(:, :, 2) * B(:, :, 2)));
        C2 = pagemtimes(A, 'transpose', A, 'none'); assert(isequal(C2(:, :, 1), A(:, :, 1)' * A(:, :, 1)));
        C3 = pagemtimes(A, B(:, :, 1)); assert(isequal(C3(:, :, 2), A(:, :, 2) * B(:, :, 1)));
        assert(isequal(pagetranspose(A), permute(A, [2 1 3])));
        Z = A + 1i * A; assert(isequal(pagectranspose(Z), conj(permute(Z, [2 1 3]))));
        S = cat(3, [2 0; 0 4], [1 1; 0 1]);
        X = pageinv(S); assert(isequal(X(:, :, 1), [0.5 0; 0 0.25]) && isequal(X(:, :, 2), [1 -1; 0 1]));
        Y = pagemldivide(S, [2; 4]); assert(isequal(Y(:, :, 1), [1; 1]) && isequal(Y(:, :, 2), [-2; 4]));
        Y = pagemrdivide([2 4], S); assert(isequal(Y(:, :, 1), [1 1]));
        n = pagenorm(S, 'fro'); assert(isequal(size(n), [1 1 2]) && abs(n(2) - sqrt(3)) < 1e-14);
        n1 = pagenorm(S, 1); assert(n1(1) == 4 && n1(2) == 2);
        sv = pagesvd(S); assert(isequal(size(sv), [2 1 2]) && isequal(sv(:, :, 1), [4; 2]));
        [U, Sg, V] = pagesvd(S); assert(norm(U(:, :, 2) * Sg(:, :, 2) * V(:, :, 2)' - S(:, :, 2)) < 1e-14);
        e = pageeig(S); assert(isequal(size(e), [2 1 2]) && isequal(sort(e(:, :, 1)), [2; 4]));
    """)
    with pytest.raises(MatlabError):
        m.run("pagemtimes(ones(2, 3), 'sideways', ones(3, 2), 'none')")


def test_gallery_families_from_their_definitions(m):
    ok(m, """
        assert(isequal(gallery('minij', 3), [1 1 1; 1 2 2; 1 2 3]));
        T = gallery('tridiag', 4); assert(issparse(T) && isequal(full(T), toeplitz([2 -1 0 0])));
        P = gallery('poisson', 3); assert(issparse(P) && isequal(size(P), [9 9]) && P(5, 5) == 4 && nnz(P) == 33);
        L = gallery('lehmer', 3); assert(abs(L(1, 3) - 1/3) < 1e-15 && abs(L(3, 2) - 2/3) < 1e-15);
        assert(isequal(gallery('moler', 3), [1 -1 -1; -1 2 0; -1 0 3]));
        F = gallery('frank', 4); assert(isequal(F, [4 3 2 1; 3 3 2 1; 0 2 2 1; 0 0 1 1]) && abs(det(F) - 1) < 1e-10);
        assert(isequal(gallery('circul', [1 2 3]), [1 2 3; 3 1 2; 2 3 1]));
        assert(isequal(gallery('jordbloc', 3, 2), [2 1 0; 0 2 1; 0 0 2]));
        assert(isequal(gallery('clement', 4), [0 1 0 0; 3 0 2 0; 0 2 0 3; 0 0 1 0]));
        assert(isequal(gallery('fiedler', [1 4 2]), [0 3 1; 3 0 2; 1 2 0]) && ...
            isequal(gallery('pei', 2, 3), [4 1; 1 4]));
        Rs = gallery('ris', 3); assert(abs(Rs(1, 1) - 0.5 / 2.5) < 1e-15 && abs(Rs(3, 3) + 1 / 3) < 1e-15);
        C = gallery('cauchy', [1 2], [3 4]); assert(isequal(C, [1/4 1/5; 1/5 1/6]));
        R = gallery('parter', 3); assert(abs(R(2, 1) - 1 / 1.5) < 1e-15);
        [v, beta, s] = gallery('house', [3; 4]); assert(norm((eye(2) - beta * (v * v')) * [3; 4] - [s; 0]) < 1e-14);
        for k = [1 2 3 4 5 6]
            Q = gallery('orthog', 5, k); assert(norm(Q' * Q - eye(5)) < 1e-12);
        end
        K = gallery('kahan', 4); assert(istriu(K) && abs(K(1, 2) + cos(1.2)) < 1e-12);
        B = gallery('binomial', 4); assert(isequal(B * B, 8 * eye(4)));
        assert(isequal(gallery(3), [-149 -50 -154; 537 180 546; -27 -9 -25]) && isequal(size(gallery(5)), [5 5]));
        Gm = gallery('gcdmat', 4); assert(Gm(4, 2) == 2 && Gm(3, 2) == 1 && Gm(4, 4) == 4);
    """)
    with pytest.raises(MatlabError):
        m.run("gallery('nosuchfamily', 3)")


def test_gallery_randsvd_has_the_requested_condition_number(m):
    ok(m, """
        rng(3); A = gallery('randsvd', 6, 100); assert(abs(cond(A) - 100) < 1e-8);
        A = gallery('randsvd', [6 4], 50, 1); s = svd(A); assert(abs(s(1) / s(end) - 50) < 1e-9);
        A = gallery('randsvd', 5, -10); assert(norm(A - A') < 1e-14 && all(eig(A) > 0) && abs(cond(A) - 10) < 1e-9);
        A = gallery('randsvd', 6, 100, 3, 1, 1); assert(abs(cond(A) - 100) < 1e-7);
        [lo, up] = bandwidth(A .* (abs(A) > 1e-13)); assert(lo <= 1 && up <= 1);
    """)


def test_classic_test_matrices(m):
    ok(m, """
        assert(isequal(hankel([1 2 3]), [1 2 3; 2 3 0; 3 0 0]) && ...
            isequal(hankel([1 2 3], [3 4 5]), [1 2 3; 2 3 4; 3 4 5]));
        R = rosser; assert(isequal(R, R') && isequal(size(R), [8 8]) && trace(R) == 4040);
        assert(abs(max(eig(R)) - 1020.049018429997) < 1e-8);
        assert(isequal(wilkinson(3), [1 1 0; 1 0 1; 0 1 1]));
        for n = [1 2 4 8 12 20 24 40]
            H = hadamard(n); assert(isequal(H' * H, n * eye(n)) && all(H(1, :) == 1) && all(H(:, 1) == 1));
        end
    """)
    with pytest.raises(MatlabError):
        m.run("hadamard(6)")


def test_condeig_and_normest1(m):
    ok(m, """
        assert(norm(condeig([2 0; 0 3]) - [1; 1]) < 1e-14);
        c = condeig([1 1000; 0 2]); assert(all(abs(c - sqrt(1 + 1000^2)) < 1e-6));
        [V, D, s] = condeig(magic(3)); assert(norm(magic(3) * V - V * D) < 1e-11 && abs(s(1) - 1) < 1e-12);
        assert(normest1(magic(6)) == 111);
        A = inv(gallery('frank', 10)); [c, v, w, it] = normest1(A);
        assert(abs(c - norm(A, 1)) < 1e-6 * norm(A, 1) && norm(w - A * v) < 1e-8 * c && numel(it) == 2);
    """)


def test_normest1_with_a_function_handle(m):
    m.run("""
        function r = af(flag, x, M)
            switch flag
                case 'dim', r = size(M, 1);
                case 'real', r = isreal(M);
                case 'notransp', r = M * x;
                case 'transp', r = M' * x;
            end
        end
        M = magic(8) + diag(1:8);
        c = normest1(@af, 2, [], M);
        assert(abs(c - norm(M, 1)) < 1e-10);
    """)


def test_more_iterative_solvers(m):
    ok(m, """
        e = ones(16, 1); A = spdiags([-e 2 * e -e], -1:1, 16, 16) + speye(16); b = ones(16, 1);
        [x, flag, relres] = bicgstabl(A, b, 1e-10, 100); assert(flag == 0 && norm(A * x - b) < 1e-8 && relres < 1e-9);
        [x, flag] = qmr(A, b, 1e-10, 100); assert(flag == 0 && norm(A * x - b) < 1e-8);
        [x, flag] = tfqmr(A, b, 1e-10, 100); assert(flag == 0 && norm(A * x - b) < 1e-8);
        [x, flag, relres, iter, resvec] = symmlq(A, b, 1e-10, 100);
        assert(flag == 0 && norm(A * x - b) < 1e-8 && iter <= 16 && numel(resvec) == iter + 1);
        [x, flag] = symmlq(A, b, 1e-10, 100, diag(diag(A))); assert(flag == 0 && norm(A * x - b) < 1e-8);
        [x, flag] = bicgstabl(@(v) A * v, b, 1e-10, 100); assert(flag == 0 && norm(A * x - b) < 1e-8);
        [x, flag] = bicgstabl(A, b, 1e-10, 1); assert(flag ~= 0);
    """)


def test_rref_and_adjoint_for_numeric_matrices(m):
    ok(m, """
        R = rref([1 2 3; 4 5 6; 7 8 9]); assert(isa(R, 'double') && norm(R - [1 0 -1; 0 1 2; 0 0 0]) < 1e-12);
        [R, jb] = rref(magic(4)); assert(isequal(jb, [1 2 3]) && isequal(rref([1 2; 3 4]), eye(2)));
        % with a tolerance the negligible pivot candidates are set to zero and the column is skipped
        assert(isequal(rref([1 1e-9; 0 1e-9], 1e-6), [1 1e-9; 0 0]) && isequal(rref([1 2; 0 1e-9]), eye(2)));
        assert(norm(adjoint([1 2; 3 4]) - [4 -2; -3 1]) < 1e-13);
        A = magic(3); assert(norm(adjoint(A) - det(A) * inv(A)) < 1e-10);
        assert(norm(adjoint([1 2; 2 4]) - [4 -2; -2 1]) < 1e-12);
    """)
