"""Geometry, scattered interpolation, N-D transforms and extra statistics."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def test_pdist2_and_knnsearch(m):
    m.run("X = [0 0; 3 4; 1 1]; Y = [0 0; 1 0]; D = pdist2(X, Y); C = pdist2(X, Y, 'cityblock');"
          "[Ds, I] = pdist2(X, Y, 'euclidean', 'Smallest', 2);"
          "[idx, d] = knnsearch(X, Y, 'K', 2); i1 = knnsearch(X, [2.9 4]);")
    np.testing.assert_allclose(m.val("D"), [[0, 1], [5, np.sqrt(20)], [np.sqrt(2), 1]])
    np.testing.assert_allclose(m.val("C"), [[0, 1], [7, 6], [2, 1]])
    np.testing.assert_array_equal(m.val("I"), [[1, 1], [3, 3]])
    np.testing.assert_allclose(m.val("Ds"), [[0, 1], [np.sqrt(2), 1]])
    np.testing.assert_array_equal(m.val("idx"), [[1, 3], [1, 3]])
    np.testing.assert_allclose(m.val("d"), [[0, np.sqrt(2)], [1, 1]])
    assert m.num("i1") == 2
    with pytest.raises(MatlabError, match="same number of columns"):
        m.run("pdist2([1 2], [1 2 3])")
    with pytest.raises(MatlabError, match="unsupported distance"):
        m.run("pdist2([1 2], [1 2], 'nope')")


def test_inpolygon_polyarea_rectint(m):
    m.run("[in, on] = inpolygon([0.5 1 2 0 0.5], [0.5 0.5 0.5 0 1], [0 1 1 0], [0 0 1 1]);"
          "a = polyarea([0 1 1 0], [0 0 1 1]); b = polyarea([0 2 2; 0 0 2]', [0 0 1; 0 3 0]');"
          "r = rectint([0 0 2 2], [1 1 2 2; 5 5 1 1]);")
    np.testing.assert_array_equal(m.val("in"), [[True, True, False, True, True]])
    np.testing.assert_array_equal(m.val("on"), [[False, True, False, True, True]])
    assert m.num("a") == 1
    np.testing.assert_allclose(m.val("b"), [[1, 3]])
    np.testing.assert_array_equal(m.val("r"), [[1, 0]])
    m.run("q = inpolygon(1.5, 1.5, [0 3 3 0 NaN 1 2 2 1], [0 0 3 3 NaN 1 1 2 2]);")  # a hole
    assert not m.num("q")


def test_convhull_and_delaunay(m):
    m.run("x = [0 1 1 0 0.5 0.2]; y = [0 0 1 1 0.5 0.9]; k = convhull(x, y);"
          "[k2, a] = convhull([0 0; 2 0; 2 1; 0 1; 1 0.5]);"
          "t = delaunay([0 1 0 1], [0 0 1 1]);"
          "[K, v] = convhull(rand(20, 3)); P = [0 0 0; 1 0 0; 0 1 0; 0 0 1]; [K3, v3] = convhulln(P);")
    np.testing.assert_array_equal(m.val("k").T, [[1, 2, 3, 4, 1]])
    assert m.num("a") == 2
    t = m.val("t")
    assert t.shape == (2, 3) and sorted(np.unique(t)) == [1, 2, 3, 4]
    assert m.val("K").shape[1] == 3 and 0 < m.num("v") < 1
    np.testing.assert_allclose(m.num("v3"), 1 / 6)
    with pytest.raises(MatlabError, match="degenerate|at least"):
        m.run("convhull([0 1 2 3], [0 1 2 3])")


def test_griddata_and_scattered_interpolant(m):
    m.run("[xq, yq] = meshgrid(0:0.5:1); x = [0 1 0 1 0.5]; y = [0 0 1 1 0.5]; v = [1 2 3 4 10];"
          "g = griddata(x, y, v, xq, yq); gn = griddata(x, y, v, 0.9, 0.1, 'nearest');"
          "out = griddata(x, y, v, 5, 5);"
          "F = scatteredInterpolant(x', y', (x + 2*y)'); f1 = F(0.25, 0.5); f2 = F([0.5 0.5; 2 2]);"
          "F.Method = 'nearest'; f3 = F(0.9, 0.9); n = size(F.Points, 1);"
          "G = scatteredInterpolant(x', y', v', 'linear', 'none'); g2 = G(3, 3);"
          "g4 = griddata(x, y, x + y, 0.3, 0.6, 'v4');")
    np.testing.assert_allclose(m.val("g"), [[1, 1.5, 2], [2, 10, 3], [3, 3.5, 4]])
    assert m.num("gn") == 2 and np.isnan(m.num("out"))
    np.testing.assert_allclose(m.num("f1"), 1.25)
    np.testing.assert_allclose(m.val("f2"), [[1.5], [6.0]])  # linear data extrapolates exactly
    assert m.num("f3") == 3 and m.num("n") == 5 and np.isnan(m.num("g2"))
    np.testing.assert_allclose(m.num("g4"), 0.8606130579, atol=1e-9)  # value from Octave
    assert "scatteredInterpolant with properties" in m.run("F")
    with pytest.raises(MatlabError, match="sample points but"):
        m.run("scatteredInterpolant([1;2;3], [1;2;3], [1;2])")


def test_sphere_cylinder_ellipsoid(m):
    m.run("[X, Y, Z] = sphere(4); [a, b, c] = cylinder([1 2], 4); [ex, ey, ez] = ellipsoid(1, 2, 3, 2, 2, 5, 6);"
          "[s1, s2, s3] = sphere;")
    np.testing.assert_allclose(m.val("Z")[:, 0], [-1, -np.sqrt(0.5), 0, np.sqrt(0.5), 1])
    np.testing.assert_allclose(m.val("X")[2], [-1, 0, 1, 0, -1], atol=1e-15)
    np.testing.assert_allclose(m.val("X") ** 2 + m.val("Y") ** 2 + m.val("Z") ** 2, np.ones((5, 5)))
    np.testing.assert_allclose(m.val("a"), [[1, 0, -1, 0, 1], [2, 0, -2, 0, 2]], atol=1e-15)
    np.testing.assert_array_equal(m.val("c"), [[0] * 5, [1] * 5])
    assert m.val("ex").shape == (7, 7) and m.val("s1").shape == (21, 21)
    np.testing.assert_allclose(m.val("ez").max(), 8)
    m.run("figure; sphere(8); h = gca; close all")


def test_fftn_convn_rcond(m):
    m.run("A = reshape(1:24, 2, 3, 4); F = fftn(A); B = ifftn(F); f2 = F(:, :, 2);"
          "c = convn(ones(3, 3, 2), ones(2, 2, 2), 'same'); cf = convn([1 2 3], [1 1]); "
          "cv = convn(magic(4), ones(2), 'valid');"
          "r = rcond([1 2; 3 4]); r0 = rcond([1 2; 2 4]); r1 = rcond(eye(3)); P = fftn([1 2; 3 4], [4 4]);")
    np.testing.assert_allclose(m.val("B"), np.arange(1, 25).reshape((2, 3, 4), order="F"), atol=1e-12)
    assert not np.iscomplexobj(m.val("B"))
    np.testing.assert_allclose(m.val("f2"), [[-72 + 72j, 0, 0], [0, 0, 0]], atol=1e-10)
    np.testing.assert_array_equal(m.val("c")[:, :, 0], [[8, 8, 4], [8, 8, 4], [4, 4, 2]])
    np.testing.assert_array_equal(m.val("c")[:, :, 1], [[4, 4, 2], [4, 4, 2], [2, 2, 1]])
    np.testing.assert_array_equal(m.val("cf"), [[1, 3, 5, 3]])
    assert m.val("cv").shape == (3, 3)
    np.testing.assert_allclose(m.num("r"), 1 / 21)
    assert m.num("r0") < 1e-15 and m.num("r1") == 1 and m.val("P").shape == (4, 4)


def test_interpn_interp3(m):
    m.run("[X, Y, Z] = meshgrid(1:3, 1:4, 1:2); V = X + 10*Y + 100*Z;"
          "a = interp3(X, Y, Z, V, 2.5, 1.5, 1.5); b = interp3(V, 2, 3, 1); o = interp3(V, 9, 1, 1);"
          "[A, B] = ndgrid(1:4, 1:3); W = A + 10*B; c = interpn(A, B, W, 2.5, 1.5); d = interpn(W, 3, 2);")
    np.testing.assert_allclose(m.num("a"), 2.5 + 15 + 150)
    assert m.num("b") == 132 and np.isnan(m.num("o"))
    np.testing.assert_allclose(m.num("c"), 17.5)
    assert m.num("d") == 23


def test_datasample_and_samplers(m):
    m.run("rng(1); [y, i] = datasample([10 20 30 40], 3, 'Replace', false);"
          "M = datasample(magic(4), 2); w = datasample(1:3, 50, 'Weights', [0 0 1]);"
          "g = gamrnd(2, 3, 1000, 1); b = betarnd(2, 2, [1 5]); c = chi2rnd(3, 2, 2);"
          "R = mvnrnd([1 2], [1 0.5; 0.5 2], 4000); p = mvnpdf([0 0], [0 0], eye(2));"
          "p2 = mvnpdf([1 2; 0 0], [1 2], [2 0; 0 0.5]);")
    y, i = m.val("y"), m.val("i")
    assert y.shape == (1, 3) and len(set(y.flatten())) == 3
    np.testing.assert_array_equal(y, np.array([[10, 20, 30, 40]])[:, i.flatten().astype(int) - 1])
    assert m.val("M").shape == (2, 4) and np.all(m.val("w") == 3)
    assert abs(m.val("g").mean() - 6) < 0.5 and m.val("b").shape == (1, 5) and m.val("c").shape == (2, 2)
    np.testing.assert_allclose(m.val("R").mean(axis=0), [1, 2], atol=0.1)
    np.testing.assert_allclose(np.cov(m.val("R").T), [[1, 0.5], [0.5, 2]], atol=0.15)
    np.testing.assert_allclose(m.num("p"), 1 / (2 * np.pi))
    np.testing.assert_allclose(m.val("p2")[0, 0], 1 / (2 * np.pi))
    with pytest.raises(MatlabError, match="without replacement"):
        m.run("datasample(1:3, 5, 'Replace', false)")


def test_pca(m):
    m.run("X = [2 0; 0 0; -2 0; 0 1; 0 -1]; [coeff, score, latent, t2, explained, mu] = pca(X);"
          "c1 = pca(X, 'NumComponents', 1);")
    np.testing.assert_allclose(np.abs(m.val("coeff")), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(m.val("latent"), [[2], [0.5]])
    np.testing.assert_allclose(m.val("explained"), [[80], [20]])
    np.testing.assert_allclose(m.val("score"), m.val("X") @ m.val("coeff"), atol=1e-12)
    np.testing.assert_allclose(m.val("mu"), [[0, 0]])
    assert m.val("c1").shape == (2, 1) and m.val("t2").shape == (5, 1)


def test_grp2idx_ksdensity_means(m):
    m.run("[g, names] = grp2idx({'b'; 'a'; 'b'; 'c'}); [g2, n2, lv] = grp2idx([5 3 5 NaN]);"
          "[f, xi, bw] = ksdensity([1 2 3 4 5]); f0 = ksdensity([0 1], 0.5, 'Bandwidth', 1);"
          "cdf = ksdensity([1 2 3], 2, 'Function', 'cdf');"
          "gm = geomean([1 4 16]); hm = harmmean([1 2 4]); nm = nanmedian([1 NaN 3 5]);")
    np.testing.assert_array_equal(m.val("g"), [[1], [2], [1], [3]])
    assert m.str("names{2}") == "a"
    np.testing.assert_array_equal(m.val("g2"), [[2], [1], [2], [np.nan]])
    np.testing.assert_array_equal(m.val("lv"), [[3], [5]])
    assert m.val("f").shape == (1, 100)
    area = np.sum(np.diff(m.val("xi").flatten()) * (m.val("f").flatten()[1:] + m.val("f").flatten()[:-1]) / 2)
    np.testing.assert_allclose(area, 1, atol=0.01)
    np.testing.assert_allclose(m.num("f0"), np.exp(-0.125) / np.sqrt(2 * np.pi))
    np.testing.assert_allclose(m.num("cdf"), 0.5)
    np.testing.assert_allclose([m.num(v).item() for v in ("gm", "hm", "nm")], [4, 12 / 7, 3])


def test_inline_and_builtin(m):
    m.run("f = inline('x.^2 + 1'); a = f(3); g = inline('a*t + b', 't', 'a', 'b'); b = g(2, 3, 1);"
          "h = inline('sin(x) + y'); c = h(0, 5); d = builtin('numel', [1 2 3]);")
    assert m.num("a") == 10 and m.num("b") == 7 and m.num("c") == 5 and m.num("d") == 3


def test_random_spectrogram_and_utilities(m):
    m.run("rng(3); r = random('Normal', 10, 0.1, 2, 3); p = random('poisson', 4, [1 5]); e = random('exp', 2);"
          "fs = 1000; t = (0:999)/fs; x = sin(2*pi*100*t);"
          "[S, F, T, P] = spectrogram(x, 128, 64, 256, fs); [~, k] = max(abs(S(:, 1)));"
          "[S2, W] = spectrogram(x); u = upsample([1 2 3], 2); d = downsample(1:7, 3);"
          "b = unicode2native('h\u00e9', 'UTF-8'); s = native2unicode(uint8([104 105]));"
          "A = [4 1 2; 1 3 0; 2 0 5]; [L, D, Pm] = ldl(A); err = norm(Pm' * A * Pm - L * D * L');")
    assert m.val("r").shape == (2, 3) and abs(m.val("r").mean() - 10) < 0.5 and m.val("p").shape == (1, 5)
    assert m.val("S").shape == (129, 14) and m.val("T").shape == (1, 14) and m.val("P").shape == (129, 14)
    assert abs(m.val("F")[int(m.num("k").item()) - 1, 0] - 100) < 4 and m.num("T(1)") == 0.064
    assert m.val("S2").shape[0] == 129 and np.isclose(m.val("W")[-1, 0], np.pi)
    np.testing.assert_array_equal(m.val("u"), [[1, 0, 2, 0, 3, 0]])
    np.testing.assert_array_equal(m.val("d"), [[1, 4, 7]])
    np.testing.assert_array_equal(m.val("b"), [[104, 195, 169]])
    assert m.str("s") == "hi" and m.num("err") < 1e-12
    m.run("figure; spectrogram(x, 128, 64, 256, fs, 'yaxis'); close all")
    with pytest.raises(MatlabError, match="unknown distribution"):
        m.run("random('nope', 1)")
