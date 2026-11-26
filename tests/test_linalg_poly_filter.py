"""Polynomial tools and filtering: residue, residuez, polyvalm, polyeig, mpoles, filter, filtic, filter2,
conv, deconv and the Fourier transforms. Expected values are worked out by hand."""

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


def test_residue_simple_poles(m):
    # 1/((s+1)(s+2)) = 1/(s+1) - 1/(s+2)
    ok(m, """
        [r, p, k] = residue(1, [1 3 2]);
        assert(isempty(k) && norm(sort(p) - [-2; -1]) < 1e-12);
        [ps, i] = sort(p); assert(norm(r(i) - [-1; 1]) < 1e-12);
        % (s^3 + 2 s^2 + 3 s + 4)/(s^2 + 3 s + 2) = s - 1 + 2/(s+1) + 2/(s+2)
        [r, p, k] = residue([1 2 3 4], [1 3 2]);
        assert(isequal(k, [1 -1]) && norm(r - [2; 2]) < 1e-12 && norm(sort(p) - [-2; -1]) < 1e-12);
    """)


def test_residue_repeated_and_complex_poles(m):
    # (s^3 + 1)/(s - 1)^3 = 1 + 3/(s-1) + 3/(s-1)^2 + 2/(s-1)^3
    ok(m, """
        [r, p, k] = residue([1 0 0 1], [1 -3 3 -1]);
        assert(isreal(r) && isreal(p) && norm(r - [3; 3; 2]) < 1e-8 && norm(p - 1) < 1e-8 && isequal(k, 1));
        % (s + 2)/(s^2 + 2 s + 5): poles -1 +- 2i, residues 0.5 -+ 0.25i
        [r, p] = residue([1 2], [1 2 5]);
        [ps, i] = sort(imag(p));
        assert(norm(p(i) - [-1-2i; -1+2i]) < 1e-12 && norm(r(i) - [0.5+0.25i; 0.5-0.25i]) < 1e-12);
    """)


def test_residue_inverse(m):
    ok(m, """
        [b, a] = residue([-1; 1], [-2; -1], []); assert(norm(a - [1 3 2]) < 1e-12 && norm(b - 1) < 1e-12);
        [b, a] = residue([3; 3; 2], [1; 1; 1], 1); assert(norm(a - [1 -3 3 -1]) < 1e-12 && norm(b - [1 0 0 1]) < 1e-12);
        [b, a] = residue([0.5-0.25i; 0.5+0.25i], [-1+2i; -1-2i], []);
        assert(isreal(a) && norm(a - [1 2 5]) < 1e-12 && norm(b - [1 2]) < 1e-12);
        [r, p, k] = residue([2 1 0 3], [1 4 5 2]); [b, a] = residue(r, p, k);
        assert(norm(a - [1 4 5 2]) < 1e-8 && norm(b - [2 1 0 3]) < 1e-7);
    """)


def test_residuez(m):
    # (1 + 2 x)/((1 - 0.5 x)(1 - 0.25 x)), x = z^-1: residues 10 and -9
    ok(m, """
        [r, p, k] = residuez([1 2], [1 -0.75 0.125]);
        [ps, i] = sort(p); assert(norm(ps - [0.25; 0.5]) < 1e-12 && norm(r(i) - [-9; 10]) < 1e-10 && isempty(k));
        [b, a] = residuez(r, p, k); assert(norm(b - [1 2]) < 1e-10 && norm(a - [1 -0.75 0.125]) < 1e-12);
        [r, p, k] = residuez([1 1 1 1], [1 -1 0.25]); [b, a] = residuez(r, p, k);
        assert(norm(b - [1 1 1 1]) < 1e-7 && norm(a - [1 -1 0.25]) < 1e-10 && numel(k) == 2);
    """)


def test_polyvalm_polyeig_mpoles(m):
    ok(m, """
        assert(isequal(polyvalm([1 0 0], [1 2; 3 4]), [7 10; 15 22]));
        assert(isequal(polyvalm([1 2 3], [1 2; 3 4]), [1 2; 3 4]^2 + 2 * [1 2; 3 4] + 3 * eye(2)));
        e = polyeig([-2 0; 0 -3], eye(2)); assert(norm(sort(e) - [2; 3]) < 1e-12);
        % lambda^2 - 3 lambda + 2 = 0 for a 1-by-1 problem
        e = polyeig(2, -3, 1); assert(norm(sort(e) - [1; 2]) < 1e-12);
        M = eye(2); C = [0.1 0; 0 0.2]; K = [2 -1; -1 2];
        [X, e, s] = polyeig(K, C, M);
        for j = 1:4, assert(norm((K + e(j) * C + e(j)^2 * M) * X(:, j)) < 1e-10); end
        assert(numel(s) == 4 && all(s > 0));
        [mults, idx] = mpoles([2 3 1 1 2]); assert(isequal(mults', [1 1 2 1 2]) && isequal(idx', [2 1 5 3 4]));
    """)


def test_filter_initial_and_final_conditions(m):
    # y(n) = x(n) + 0.5 y(n-1): impulse response 1, 0.5, 0.25, ...
    ok(m, """
        assert(isequal(filter(1, [1 -0.5], [1 0 0]), [1 0.5 0.25]));
        [y, zf] = filter(1, [1 -0.5], [1 0], 0); assert(isequal(y, [1 0.5]) && zf == 0.25);
        % filtering in two pieces with the state passed on equals filtering in one piece
        x = [1 2 3 4 5 6]; b = [1 2 1]; a = [1 -0.3 0.1];
        yall = filter(b, a, x); [y1, z] = filter(b, a, x(1:3)); y2 = filter(b, a, x(4:6), z);
        assert(norm([y1 y2] - yall) < 1e-14 && isequal(size(z), [2 1]));
        [y, zf] = filter([1 1], 1, [1 2 3]); assert(isequal(y, [1 3 5]) && zf == 3);
        Y = filter([1 1], 1, [1 2; 3 4]); assert(isequal(Y, [1 2; 4 6]));
        Y = filter([1 1], 1, [1 2; 3 4], [], 2); assert(isequal(Y, [1 3; 3 7]));
        [Y, zf] = filter([1 1], 1, [1 2; 3 4], [10 20]); assert(isequal(Y, [11 22; 4 6]) && isequal(zf, [3 4]));
        assert(isequal(filter(2, 2, [1 2 3]), [1 2 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("filter(1, [0 1], [1 2 3])")
    with pytest.raises(MatlabError):
        m.run("filter(1, [1 -0.5], [1 2 3], [1 2 3])")


def test_filtic_and_filter2(m):
    ok(m, """
        % one past output of 2 and one past input of 4 for y(n) = x(n) + x(n-1) + 0.5 y(n-1)
        assert(filtic([1 1], [1 -0.5], 2, 4) == 5);
        z = filtic([1 2 1], [1 -0.3 0.1], [3 2], [1 4]);
        x = [5 6 7]; y = filter([1 2 1], [1 -0.3 0.1], x, z);
        assert(abs(y(1) - (5 + 2 * 1 + 4 + 0.3 * 3 - 0.1 * 2)) < 1e-14);
        % correlation: each output is the sum of the products with the kernel placed on the data
        A = magic(3); h = [1 2; 3 4];
        V = filter2(h, A, 'valid'); assert(isequal(V, [sum(sum(h .* A(1:2, 1:2))) sum(sum(h .* A(1:2, 2:3)));
                                                      sum(sum(h .* A(2:3, 1:2))) sum(sum(h .* A(2:3, 2:3)))]));
        assert(isequal(size(filter2(h, A)), [3 3]) && isequal(size(filter2(h, A, 'full')), [4 4]));
    """)


def test_conv_and_deconv(m):
    ok(m, """
        assert(isequal(conv([1 2 3], [1 1]), [1 3 5 3]) && isequal(conv([1 2 3], [1 1], 'same'), [3 5 3]));
        assert(isequal(conv([1 2 3], [1 1], 'valid'), [3 5]) && iscolumn(conv([1; 2; 3], [1 1])));
        [q, r] = deconv([1 3 5 3], [1 1]); assert(isequal(q, [1 2 3]) && all(r == 0));
        [q, r] = deconv([1 5 6 1], [1 2]); assert(isequal(q, [1 3 0]) && isequal(r, [0 0 0 1]));
        [x, r] = deconv([1 3 5 3], [1 1], 'Method', 'least-squares');
        assert(norm(x - [1 2 3]) < 1e-12 && norm(r) < 1e-12);
        y = conv([1 2 3], [1 1], 'same'); x = deconv(y, [1 1], 'same', 'Method', 'least-squares');
        assert(norm(conv(x, [1 1], 'same') - y) < 1e-10);
        x0 = deconv([1 3 5 3], [1 1], 'Method', 'least-squares', 'RegularizationFactor', 10);
        assert(norm(x0) < norm([1 2 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("conv([1 2], [1 1], 'bogus')")
    with pytest.raises(MatlabError):
        m.run("deconv([1 2 3], [0 1])")


def test_fft_real_results_and_dimensions(m):
    ok(m, """
        assert(isreal(fft([1 0 0 0])) && isequal(fft([1 0 0 0]), [1 1 1 1]));
        assert(isreal(fft(ones(1, 4))) && isequal(fft(ones(1, 4)), [4 0 0 0]) && ~isreal(fft([1 2 3])));
        assert(isreal(ifft(fft([1 2 3 4 5]))) && isreal(fft2(eye(2))) && isequal(fft2(eye(2)), [2 0; 0 2]));
        X = reshape(1:24, 2, 3, 4); Y = fft(X, [], 3);
        assert(isequal(size(Y), [2 3 4]) && norm(Y(:, :, 1) - sum(X, 3)) < 1e-12);
        assert(isequal(size(fft([1 2 3]', 5)), [5 1]) && isa(fft(single([1 2 3])), 'single'));
        assert(norm(fft([1 2 3 4]) - [10, -2+2i, -2, -2-2i]) < 1e-14);
    """)
