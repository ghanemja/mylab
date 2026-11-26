"""Special functions, exact trigonometry in degrees and number theory.

Reference values are closed forms, standard table values or defining identities.
"""

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


def test_error_gamma_and_beta_families(m):
    ok(m, """
        assert(erfcx(0) == 1 && abs(erfcx(1) - exp(1) * erfc(1)) < 1e-15 && abs(erfcx(1) - 0.427583576155807) < 1e-14);
        % the first terms of the asymptotic series for large x: 1/(x sqrt(pi)) (1 - 1/(2x^2) + 3/(4x^4))
        assert(abs(erfcx(30) - 1 / (30 * sqrt(pi)) * (1 - 1 / 1800 + 3 / 3240000)) < 1e-10);
        assert(abs(betaln(2, 3) - log(1 / 12)) < 1e-14 && ...
            abs(betaln(200, 300) - (gammaln(200) + gammaln(300) - gammaln(500))) < 1e-9);
        % the median of the exponential distribution is log(2)
        assert(abs(gammaincinv(0.5, 1) - log(2)) < 1e-14 && abs(gammaincinv(0.5, 1, 'upper') - log(2)) < 1e-14);
        assert(abs(gammainc(gammaincinv(0.3, 2.5), 2.5) - 0.3) < 1e-13);
        % I_x(2, 3) = 6 x^2 - 8 x^3 + 3 x^4, which is 0.6875 at x = 0.5
        assert(abs(betaincinv(0.6875, 2, 3) - 0.5) < 1e-13 && abs(betaincinv(1 - 0.6875, 2, 3, 'upper') - 0.5) < 1e-13);
    """)
    with pytest.raises(MatlabError):
        m.run("gammaincinv(1.5, 1)")
    with pytest.raises(MatlabError):
        m.run("betaincinv(0.5, 2, 3, 'sideways')")


def test_gammainc_tails(m):
    # for a = 1: P = 1 - exp(-x), Q = exp(-x); the scaling factor is gamma(a+1) exp(x) / x^a
    ok(m, """
        x = 2; assert(abs(gammainc(x, 1) - (1 - exp(-2))) < 1e-15 && abs(gammainc(x, 1, 'upper') - exp(-2)) < 1e-15);
        assert(abs(gammainc(x, 1, 'scaledlower') - (1 - exp(-2)) * exp(2) / 2) < 1e-13);
        assert(abs(gammainc(x, 1, 'scaledupper') - 0.5) < 1e-14);
    """)
    with pytest.raises(MatlabError):
        m.run("gammainc(1, 2, 'middle')")


def test_bessel_hankel_and_elliptic(m):
    ok(m, """
        h = besselh(0, 1); assert(abs(h - (0.765197686557967 + 0.088256964215677i)) < 1e-14);
        assert(abs(besselh(0, 1, 1) - h) < 1e-15 && abs(besselh(0, 2, 1) - conj(h)) < 1e-15);
        assert(abs(besselh(1, 1, 2.5) - (besselj(1, 2.5) + 1i * bessely(1, 2.5))) < 1e-14);
        assert(abs(besselh(0, 1, 1, 1) - h * exp(-1i)) < 1e-14);
        [sn, cn, dn] = ellipj(0.5, 0.3);
        assert(abs(sn^2 + cn^2 - 1) < 1e-14 && abs(dn^2 + 0.3 * sn^2 - 1) < 1e-14);
        [sn, cn, dn] = ellipj(0.7, 0); assert(abs(sn - sin(0.7)) < 1e-15 && abs(cn - cos(0.7)) < 1e-15 && dn == 1);
        [sn, cn, dn] = ellipj(0.7, 1);
        assert(abs(sn - tanh(0.7)) < 1e-14 && abs(cn - sech(0.7)) < 1e-14 && abs(dn - cn) < 1e-14);
        [sn, cn] = ellipj([0 0.5], 0.3); assert(isequal(size(sn), [1 2]) && sn(1) == 0 && cn(1) == 1);
        [K, E] = ellipke(0.5, 1e-6); assert(abs(K - 1.854074677301372) < 1e-12 && abs(E - 1.350643881047675) < 1e-12);
    """)
    with pytest.raises(MatlabError):
        m.run("ellipj(0.5, 1.5)")
    with pytest.raises(MatlabError):
        m.run("besselh(0, 3, 1)")


def test_legendre_normalizations(m):
    # P_2^0 = (3x^2-1)/2, P_2^1 = -3x sqrt(1-x^2), P_2^2 = 3(1-x^2)
    ok(m, """
        x = 0.5; s = sqrt(1 - x^2);
        P = legendre(2, x); assert(norm(P - [(3 * x^2 - 1) / 2; -3 * x * s; 3 * s^2]) < 1e-14);
        S = legendre(2, x, 'sch'); assert(norm(S - [P(1); -sqrt(2 / 6) * P(2); sqrt(2 / 24) * P(3)]) < 1e-14);
        N = legendre(2, x, 'norm');
        assert(norm(N - [sqrt(2.5) * P(1); -sqrt(2.5 / 6) * P(2); sqrt(2.5 / 24) * P(3)]) < 1e-14);
        assert(isequal(size(legendre(3, [0 0.1 0.2])), [4 3]) && isequal(size(legendre(3, zeros(2, 5))), [4 2 5]));
        assert(isequal(legendre(0, 0.3), 1) && isequal(legendre(1, [-1 0 1]), [-1 0 1; 0 -1 0]));
        % the fully normalized functions have unit norm over [-1, 1] (checked with Gauss-Legendre-free trapezoids)
        t = linspace(-1, 1, 20001); Q = legendre(40, t, 'norm'); assert(abs(trapz(t, Q(11, :).^2) - 1) < 1e-6);
        assert(all(isfinite(legendre(150, 0.3, 'norm'))));
    """)
    with pytest.raises(MatlabError):
        m.run("legendre(2, 1.5)")
    with pytest.raises(MatlabError):
        m.run("legendre(2, 0.5, 'other')")


def test_degree_trigonometry_is_exact(m):
    ok(m, """
        assert(sind(30) == 0.5 && sind(150) == 0.5 && sind(-30) == -0.5 && sind(210) == -0.5 && sind(390) == 0.5);
        assert(cosd(60) == 0.5 && cosd(120) == -0.5 && cosd(300) == 0.5 && cosd(-60) == 0.5);
        assert(sind(0) == 0 && sind(180) == 0 && sind(360) == 0 && sind(90) == 1 && sind(270) == -1);
        assert(cosd(90) == 0 && cosd(270) == 0 && cosd(0) == 1 && cosd(180) == -1);
        assert(tand(45) == 1 && tand(135) == -1 && tand(225) == 1 && tand(-45) == -1 && tand(180) == 0);
        assert(tand(90) == Inf && tand(-90) == -Inf && cosd(45) == sind(45));
        assert(abs(sind(1e6) - sin(pi * 280 / 180)) < 1e-15);        % 1e6 = 2777*360 + 280
        assert(secd(60) == 2 && cscd(30) == 2 && cotd(45) == 1 && cotd(90) == 0 && isinf(cotd(0)));
        assert(abs(asecd(2) - 60) < 1e-13 && abs(acscd(2) - 30) < 1e-13 && abs(acotd(1) - 45) < 1e-13);
        assert(sinpi(1) == 0 && sinpi(0.5) == 1 && cospi(0.5) == 0 && cospi(1) == -1 && sinpi(1 / 6) == 0.5);
        assert(isa(sind(single(30)), 'single') && isequal(sind([0 30 90]), [0 0.5 1]));
        assert(abs(cosd(1 + 2i) - cos((1 + 2i) * pi / 180)) < 1e-15);
    """)


def test_nthroot_and_real_functions(m):
    ok(m, """
        assert(nthroot(27, 3) == 3 && nthroot(-8, 3) == -2 && nthroot(16, 4) == 2 && nthroot(1000, 3) == 10);
        assert(isequal(nthroot([8 27 64 125], 3), [2 3 4 5]) && nthroot(0, 5) == 0 && nthroot(Inf, 3) == Inf);
        assert(abs(nthroot(2, 2) - sqrt(2)) < 1e-15 && nthroot(2, -1) == 0.5);
        assert(reallog(1) == 0 && abs(reallog(exp(2)) - 2) < 1e-15 && realpow(2, 3) == 8);
    """)
    with pytest.raises(MatlabError):
        m.run("nthroot(-8, 2)")
    with pytest.raises(MatlabError):
        m.run("reallog(-1)")
    with pytest.raises(MatlabError):
        m.run("realpow(-8, 1/3)")


def test_number_theory_classes(m):
    ok(m, """
        p = primes(uint8(20)); assert(isa(p, 'uint8') && isequal(p, uint8([2 3 5 7 11 13 17 19])));
        assert(isa(primes(20), 'double') && isempty(primes(1)) && isa(primes(single(10)), 'single'));
        f = factor(uint16(60)); assert(isa(f, 'uint16') && isequal(f, uint16([2 2 3 5])));
        assert(isequal(factor(97), 97) && isequal(factor(1), 1) && isequal(factor(2^10 * 3), [2 * ones(1, 10) 3]));
        assert(isequal(factor(uint64(600851475143)), uint64([71 839 1471 6857])));
        assert(nchoosek(40, 20) == 137846528820 && isequal(nchoosek(uint8(5), 2), uint8(10)));
    """)


def test_rats(m):
    ok(m, """
        assert(strcmp(strtrim(rats(0.75)), '3/4') && strcmp(strtrim(rats(pi)), '355/113'));
        assert(strcmp(strtrim(rats(2)), '2') && strcmp(strtrim(rats(-0.5)), '-1/2') && numel(rats(0.75)) == 14);
        s = rats([0.5 0.25; 1/3 2]); assert(isequal(size(s), [2 28]) && contains(s(2, :), '1/3'));
        assert(strcmp(strtrim(rats(Inf)), 'Inf') && strcmp(strtrim(rats(pi, 5)), '22/7'));
        [n, d] = rat(0.75); assert(n == 3 && d == 4);
    """)
