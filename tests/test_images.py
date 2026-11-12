"""Image files and image processing, checked against independent NumPy/SciPy/Pillow computations."""

import os

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage, signal

from mylab.errors import MatlabError


@pytest.fixture
def im(m):
    """Session with a few test images already in the workspace."""
    rng = np.random.default_rng(7)
    m.A = rng.integers(0, 256, (17, 23)).astype(np.uint8)
    m.D = rng.random((12, 15))
    m.RGB = rng.integers(0, 256, (9, 11, 3)).astype(np.uint8)
    m.BW = rng.random((14, 16)) > 0.6
    for name in ("A", "D", "RGB", "BW"):
        m.interp.base.set(name, getattr(m, name))
    m.rng = rng
    return m


def put(m, name, value):
    m.interp.set_var(name, value)


def scalar(m, expr):
    return float(np.asarray(m.num(expr)).ravel()[0])


# -- class conversion ---------------------------------------------------------------------------

def test_im2double_and_back(im):
    d = im.num("im2double(A)")
    assert d.dtype == np.float64 and np.allclose(d, im.A / 255)
    assert np.array_equal(im.num("im2uint8(im2double(A))"), im.A)
    assert im.num("im2single(A)").dtype == np.float32
    assert np.array_equal(im.num("im2uint16(A)"), im.A.astype(np.uint16) * 257)
    assert np.array_equal(im.num("im2uint8(im2uint16(A))"), im.A)
    assert np.array_equal(im.num("im2uint8([0 0.5 1 2 -1])"), [[0, 128, 255, 255, 0]])
    assert np.array_equal(im.num("im2uint8([true false])"), [[255, 0]])
    assert np.array_equal(im.num("im2double(int16([-32768 0 32767]))"), [[0, 32768 / 65535, 1]])
    assert np.array_equal(im.num("im2double(BW)"), im.BW.astype(float))


def test_mat2gray(m):
    assert np.allclose(m.num("mat2gray([1 2 3; 4 5 6])"), (np.arange(6).reshape(2, 3)) / 5)
    assert np.allclose(m.num("mat2gray([0 5 10 20], [5 10])"), [[0, 0, 1, 1]])
    assert np.allclose(m.num("mat2gray([3 3 3])"), [[1, 1, 1]])


def test_rgb2gray_weights_and_class(im):
    ref = np.floor(im.RGB.astype(float) @ [0.298936021293775, 0.587043074451121, 0.114020904255103] + 0.5)
    g = im.num("rgb2gray(RGB)")
    assert g.dtype == np.uint8 and np.array_equal(g, ref)
    gd = im.num("rgb2gray(im2double(RGB))")
    assert gd.dtype == np.float64 and np.allclose(gd * 255, im.RGB.astype(float) @ [0.2989, 0.5870, 0.1140], atol=0.05)
    assert np.allclose(im.num("rgb2gray([1 0 0; 0 1 0])"), [[0.2989] * 3, [0.5870] * 3], atol=1e-4)
    with pytest.raises(MatlabError):
        im.run("rgb2gray(ones(2, 2, 4))")


def test_hsv_round_trip(im):
    import matplotlib.colors as mc
    assert np.allclose(im.num("rgb2hsv(RGB)"), mc.rgb_to_hsv(im.RGB / 255))
    put(im, "H", im.rng.random((5, 6, 3)))
    assert np.allclose(im.num("hsv2rgb(H)"), mc.hsv_to_rgb(im.num("H")))
    assert np.allclose(im.num("hsv2rgb(rgb2hsv(RGB))"), im.RGB / 255)
    assert np.allclose(im.num("rgb2hsv([1 0 0; 0 1 0; 0 0 1; 1 1 1])"),
                       [[0, 1, 1], [1 / 3, 1, 1], [2 / 3, 1, 1], [0, 0, 1]])
    im.run("[h, s, v] = rgb2hsv(RGB);")
    assert im.num("h").shape == (9, 11)


def test_indexed_images(im):
    im.run("[X, map] = rgb2ind(RGB, 500);")
    assert im.num("X").dtype == np.uint16 or im.num("X").dtype == np.uint8
    assert np.allclose(im.num("ind2rgb(X, map)") * 255, im.RGB)
    im.run("[X8, map8] = rgb2ind(RGB, 8);")
    assert im.num("map8").shape[0] <= 8 and im.num("X8").dtype == np.uint8
    assert int(im.num("X8").max()) < im.num("map8").shape[0]
    # nearest colour in a given map
    assert np.array_equal(im.num("rgb2ind(cat(3, [1 0], [0 0], [0 1]), [0 0 1; 1 0 0])"), [[1, 0]])
    assert np.array_equal(im.num("ind2rgb([1 2], [1 0 0; 0 0 1])"), [[[1, 0, 0], [0, 0, 1]]])
    assert np.array_equal(im.num("ind2rgb(uint8([0 1]), [1 0 0; 0 0 1])"), [[[1, 0, 0], [0, 0, 1]]])
    im.run("[G, gm] = gray2ind(uint8([0 128 255]), 4);")
    assert np.array_equal(im.num("G"), [[0, 2, 3]]) and np.allclose(im.num("gm")[:, 0], [0, 1 / 3, 2 / 3, 1])


def test_label2rgb(m):
    out = m.num("label2rgb([0 1; 2 2], [1 0 0; 0 1 0], 'k')")
    assert out.dtype == np.uint8
    assert np.array_equal(out[0, 0], [0, 0, 0]) and np.array_equal(out[0, 1], [255, 0, 0])
    assert np.array_equal(out[1, 0], [0, 255, 0])
    assert m.num("label2rgb([0 1 2 3])").shape == (1, 4, 3)


def test_imcomplement(im):
    assert np.array_equal(im.num("imcomplement(A)"), 255 - im.A)
    assert np.allclose(im.num("imcomplement(D)"), 1 - im.D)
    assert np.array_equal(im.num("imcomplement(BW)"), ~im.BW)


# -- thresholds and histograms ----------------------------------------------------------------

def _otsu(img):
    c = np.bincount(img.ravel(), minlength=256).astype(float)
    p = c / c.sum()
    best, arg = -1.0, []
    for t in range(255):
        w0 = p[:t + 1].sum()
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = (p[:t + 1] * np.arange(t + 1)).sum() / w0
        m1 = (p[t + 1:] * np.arange(t + 1, 256)).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best * (1 + 1e-12):
            best, arg = v, [t]
        elif abs(v - best) <= 1e-12 * best:
            arg.append(t)
    return np.mean(arg) / 255


def test_graythresh_matches_brute_force_otsu(im):
    vals = np.concatenate([im.rng.normal(60, 15, 400), im.rng.normal(180, 20, 300)])
    B = np.clip(vals, 0, 255).astype(np.uint8).reshape(28, 25)
    put(im, "B", B)
    assert abs(scalar(im, "graythresh(B)") - _otsu(B)) < 1e-12
    assert np.array_equal(im.num("imbinarize(B)"), B / 255 > _otsu(B))
    im.run("[lvl, em] = graythresh(B);")
    assert 0.5 < scalar(im, "em") <= 1
    assert abs(scalar(im, "otsuthresh(imhist(B))") - _otsu(B)) < 1e-12
    # every level between the two values separates them equally well; the middle one is returned
    assert scalar(im, "graythresh(uint8([10 10 200 200]))") == pytest.approx(104.5 / 255)


def test_im2bw_and_imbinarize(im):
    assert np.array_equal(im.num("im2bw(A, 0.4)"), im.A > 0.4 * 255)
    assert np.array_equal(im.num("im2bw(A)"), im.A > 127.5)
    assert np.array_equal(im.num("imbinarize(D, 0.3)"), im.D > 0.3)
    assert np.array_equal(im.num("im2bw(RGB, 0.5)"), im.num("rgb2gray(RGB)") > 127.5)
    assert im.num("imbinarize(A, 'adaptive')").dtype == bool
    # dark writing on unevenly lit paper: a single threshold fails, the local one does not
    y, x = np.mgrid[0:64, 0:64]
    paper = 0.3 + 0.6 * x / 63
    ink = (y % 16 == 8) & (x % 4 < 2)
    put(im, "S", np.where(ink, paper * 0.5, paper))
    local = im.num("imbinarize(S, 'adaptive')")
    assert np.array_equal(local, ~ink)
    assert not np.array_equal(im.num("imbinarize(S)"), ~ink)
    assert im.num("imbinarize(S, 'adaptive', 'Sensitivity', 0)").sum() < local.sum()


def test_imhist(im):
    im.run("[c, x] = imhist(A);")
    assert np.array_equal(im.num("c").ravel(), np.bincount(im.A.ravel(), minlength=256))
    assert np.array_equal(im.num("x").ravel(), np.arange(256)) and im.num("c").shape == (256, 1)
    im.run("[c, x] = imhist(D, 5);")
    assert np.array_equal(im.num("c").ravel(), np.bincount(np.floor(im.D * 4 + 0.5).astype(int).ravel(), minlength=5))
    assert np.allclose(im.num("x").ravel(), [0, 0.25, 0.5, 0.75, 1])
    assert np.array_equal(im.num("imhist(BW)").ravel(), [(~im.BW).sum(), im.BW.sum()])


def test_imhist_without_outputs_draws_bars(im):
    assert im.run("imhist(A)") == ""
    assert im.num("xlim")[0, 1] >= 255
    im.run("close all")


def test_imadjust_and_stretchlim(im):
    assert np.array_equal(im.num("imadjust(uint8([0 64 128 255]), [0.25 0.75], [0 1])"), [[0, 0, 129, 255]])
    assert np.allclose(im.num("imadjust([0 0.25 0.5 1], [0 1], [1 0])"), [[1, 0.75, 0.5, 0]])
    assert np.allclose(im.num("imadjust([0 0.25 1], [], [], 2)"), [[0, 0.0625, 1]])
    G = (im.rng.random((64, 64)) ** 3 * 200 + 20).astype(np.uint8)
    put(im, "G", G)
    lim = im.num("stretchlim(G)").ravel()
    cdf = np.cumsum(np.bincount(G.ravel(), minlength=256)) / G.size
    assert lim[0] == pytest.approx(np.argmax(cdf > 0.01) / 255)
    assert lim[1] == pytest.approx(np.argmax(cdf >= 0.99) / 255)
    J = im.num("imadjust(G)")
    assert J.dtype == np.uint8 and J.min() == 0 and J.max() == 255
    assert np.array_equal(J, im.num("imadjust(G, stretchlim(G), [])"))
    assert im.num("imadjust(RGB, [0.1 0.2 0.3; 0.9 0.8 0.7], [])").shape == (9, 11, 3)


def test_histeq_flattens_histogram(im):
    G = (im.rng.random((80, 80)) ** 3 * 255).astype(np.uint8)
    put(im, "G", G)
    J = im.num("histeq(G)")
    assert J.dtype == np.uint8 and len(np.unique(J)) <= 64
    assert abs(J.mean() - 127.5) < 6 < abs(G.mean() - 127.5)
    # the mapping is monotonic
    order = np.argsort(G.ravel(), kind="stable")
    assert np.all(np.diff(J.ravel()[order].astype(int)) >= 0)
    im.run("[J2, T] = histeq(G, 256);")
    assert im.num("T").shape == (1, 256) and np.all(np.diff(im.num("T").ravel()) >= 0)


def test_entropy(m):
    assert scalar(m, "entropy(uint8([0 0 255 255]))") == pytest.approx(1.0)
    assert scalar(m, "entropy(uint8(zeros(4)))") == 0.0


# -- arithmetic ----------------------------------------------------------------------------------

def test_saturating_arithmetic(m):
    assert np.array_equal(m.num("imadd(uint8([200 100]), uint8([100 100]))"), [[255, 200]])
    assert np.array_equal(m.num("imsubtract(uint8([50 100]), uint8([100 30]))"), [[0, 70]])
    assert np.array_equal(m.num("imabsdiff(uint8([50 100]), uint8([100 30]))"), [[50, 70]])
    assert np.array_equal(m.num("immultiply(uint8([20 100]), 3)"), [[60, 255]])
    assert np.array_equal(m.num("imdivide(uint8([20 101]), 2)"), [[10, 51]])
    assert m.num("imadd(uint8(1), uint8(1))").dtype == np.uint8
    assert np.allclose(m.num("imabsdiff([0.2 0.9], [0.5 0.1])"), [[0.3, 0.8]])


def test_quality_metrics(im):
    N = np.clip(im.A.astype(int) + im.rng.integers(-10, 10, im.A.shape), 0, 255).astype(np.uint8)
    put(im, "N", N)
    mse = ((im.A.astype(float) - N) ** 2).mean()
    assert scalar(im, "immse(N, A)") == pytest.approx(mse)
    assert scalar(im, "psnr(N, A)") == pytest.approx(10 * np.log10(255 ** 2 / mse))
    assert scalar(im, "psnr(im2double(N), im2double(A))") == pytest.approx(10 * np.log10(255 ** 2 / mse))
    assert scalar(im, "psnr(N, A, 100)") == pytest.approx(10 * np.log10(100 ** 2 / mse))
    assert scalar(im, "ssim(A, A)") == pytest.approx(1.0)
    s = scalar(im, "ssim(N, A)")
    assert 0.9 < s < 1
    im.run("[s2, smap] = ssim(N, A);")
    assert im.num("smap").shape == im.A.shape and im.num("smap").mean() == pytest.approx(s)
    assert scalar(im, "corr2(A, A)") == pytest.approx(1.0)
    assert scalar(im, "mean2(A)") == pytest.approx(im.A.mean())
    assert scalar(im, "std2(A)") == pytest.approx(im.A.astype(float).std(ddof=1))


def test_ssim_against_direct_formula(im):
    # constant images: means differ, variances are zero
    put(im, "P", np.full((20, 20), 0.5))
    put(im, "Q", np.full((20, 20), 0.6))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    expected = (2 * 0.5 * 0.6 + c1) / (0.25 + 0.36 + c1) * (c2 / c2)
    assert scalar(im, "ssim(P, Q)") == pytest.approx(expected)


def test_integral_image(im):
    J = im.num("integralImage(A)")
    assert J.shape == (18, 24) and np.all(J[0] == 0) and np.all(J[:, 0] == 0)
    assert J[5, 7] == im.A[:5, :7].sum() and J[-1, -1] == im.A.sum()


# -- filtering -----------------------------------------------------------------------------------

def _brute_filter(a, h, mode, conv=False, full=False):
    a = a.astype(float)
    if conv:
        h = h[::-1, ::-1]
    kr, kc = h.shape
    cr, cc = (kr - 1) // 2, (kc - 1) // 2
    kw = {"zero": {"mode": "constant"}, "replicate": {"mode": "edge"}, "symmetric": {"mode": "symmetric"},
          "circular": {"mode": "wrap"}}[mode]
    P = np.pad(a, ((kr - 1, kr - 1), (kc - 1, kc - 1)), **kw)
    R, C = a.shape
    if full:
        return np.array([[(P[i:i + kr, j:j + kc] * h).sum() for j in range(C + kc - 1)] for i in range(R + kr - 1)])
    r0, c0 = kr - 1 - cr, kc - 1 - cc
    return np.array([[(P[i + r0:i + r0 + kr, j + c0:j + c0 + kc] * h).sum() for j in range(C)] for i in range(R)])


@pytest.mark.parametrize("shape", [(3, 3), (2, 2), (4, 3), (1, 5), (2, 5)])
def test_imfilter_against_brute_force(im, shape):
    h = im.rng.random(shape)
    put(im, "h", h)
    for mode, opt in [("zero", ""), ("replicate", ", 'replicate'"), ("symmetric", ", 'symmetric'"),
                      ("circular", ", 'circular'")]:
        for conv in (False, True):
            c = ", 'conv'" if conv else ""
            assert np.allclose(im.num(f"imfilter(D, h{opt}{c})"), _brute_filter(im.D, h, mode, conv))
            assert np.allclose(im.num(f"imfilter(D, h{opt}, 'full'{c})"), _brute_filter(im.D, h, mode, conv, True))
    assert np.allclose(im.num("imfilter(D, h, 'conv')"), signal.convolve2d(im.D, h, mode="full")[
        (shape[0] - 1) - (shape[0] - 1) // 2:, (shape[1] - 1) - (shape[1] - 1) // 2:][:12, :15])
    assert np.allclose(im.num("imfilter(D, h, 'conv')"), im.num("conv2(D, h, 'same')"))
    assert np.allclose(im.num("imfilter(D, h, 'full', 'conv')"), signal.convolve2d(im.D, h, mode="full"))
    assert np.allclose(im.num("filter2(h, D, 'valid')"), signal.correlate2d(im.D, h, mode="valid"))
    assert np.allclose(im.num("filter2(h, D, 'full')"), signal.correlate2d(im.D, h, mode="full"))
    assert np.allclose(im.num("filter2(h, D)"), _brute_filter(im.D, h, "zero"))


def test_imfilter_classes_and_padding_value(im):
    put(im, "h", np.ones((3, 3)) / 9)
    g = im.num("imfilter(A, h, 'replicate')")
    ref = np.clip(np.floor(_brute_filter(im.A, np.ones((3, 3)) / 9, "replicate") + 0.5), 0, 255)
    assert g.dtype == np.uint8 and np.array_equal(g, ref)
    # negative responses clip to 0 for unsigned images, and survive for double
    assert im.num("imfilter(uint8([10 20 30]), [1 0 -1])").tolist() == [[0, 0, 20]]
    assert im.num("imfilter([10 20 30], [1 0 -1])").tolist() == [[-20, -20, 20]]
    assert np.allclose(im.num("imfilter(ones(3), ones(3), 5)"), [[29, 21, 29], [21, 9, 21], [29, 21, 29]])
    out = im.num("imfilter(RGB, h)")
    assert out.shape == im.RGB.shape and np.array_equal(out[..., 1], im.num("imfilter(RGB(:,:,2), h)"))
    with pytest.raises(MatlabError):
        im.run("imfilter(A, h, 'bogus')")


def test_fspecial(m):
    assert np.allclose(m.num("fspecial('average')"), np.full((3, 3), 1 / 9))
    assert np.allclose(m.num("fspecial('average', [2 4])"), np.full((2, 4), 1 / 8))
    x = np.arange(-2, 3)
    g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 2)
    assert np.allclose(m.num("fspecial('gaussian', 5, 1)"), g / g.sum())
    assert m.num("fspecial('gaussian')").shape == (3, 3)
    assert np.array_equal(m.num("fspecial('sobel')"), [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    assert np.array_equal(m.num("fspecial('prewitt')"), [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    assert np.allclose(m.num("fspecial('laplacian', 0)"), [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    assert np.allclose(m.num("fspecial('laplacian')"), np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]]) / 6)
    u = m.num("fspecial('unsharp')")
    assert u.sum() == pytest.approx(1) and u[1, 1] > 1
    log = m.num("fspecial('log', 7, 1)")
    assert log.shape == (7, 7) and abs(log.sum()) < 1e-12 and log[3, 3] < 0 < log[3, 0] + 1
    d = m.num("fspecial('disk', 3)")
    assert d.shape == (7, 7) and d.sum() == pytest.approx(1) and np.allclose(d, d.T) and np.allclose(d, d[::-1])
    assert d[3, 3] * np.pi * 9 == pytest.approx(1, abs=1e-4)      # a fully covered pixel has area 1
    assert d[0, 0] == 0 and 0 < d[0, 3] < d[3, 3]                    # corners outside, edge pixels half covered
    assert d[0, 3] * np.pi * 9 == pytest.approx(0.5 - 1 / 72, abs=1e-3)   # half a pixel minus the cap
    assert np.allclose(m.num("fspecial('motion', 5, 0)"), np.full((1, 5), 0.2))
    assert m.num("fspecial('motion', 9, 90)").shape == (9, 1)
    assert m.num("fspecial('motion', 7, 45)").sum() == pytest.approx(1)
    with pytest.raises(MatlabError):
        m.run("fspecial('nonsense')")


def test_imgaussfilt_and_median(im):
    assert np.allclose(im.num("imgaussfilt(D, 2)"), ndimage.gaussian_filter(im.D, 2, mode="nearest", truncate=2.0))
    assert np.allclose(im.num("imgaussfilt(D)"), ndimage.gaussian_filter(im.D, 0.5, mode="nearest", truncate=2.0))
    assert im.num("imgaussfilt(A, 1)").dtype == np.uint8
    assert im.num("imgaussfilt(RGB, 1)").shape == im.RGB.shape
    assert np.allclose(im.num("imgaussfilt(D, 1, 'FilterSize', 3)"),
                       ndimage.gaussian_filter(im.D, 1, mode="nearest", radius=1))
    assert np.array_equal(im.num("medfilt2(A)"), ndimage.median_filter(im.A, 3, mode="constant"))
    assert np.array_equal(im.num("medfilt2(A, [5 3], 'symmetric')"),
                          ndimage.median_filter(im.A, (5, 3), mode="reflect"))
    assert np.array_equal(im.num("ordfilt2(A, 9, ones(3))"), ndimage.maximum_filter(im.A, 3, mode="constant"))
    assert np.array_equal(im.num("ordfilt2(A, 5, ones(3))"), im.num("medfilt2(A)"))
    # salt and pepper noise is removed
    put(im, "F", np.full((9, 9), 100, np.uint8))
    im.run("F(5, 5) = 255; F(2, 7) = 0;")
    assert np.all(im.num("medfilt2(F)")[1:-1, 1:-1] == 100)


def test_wiener2_and_imnoise(im):
    im.run("rng(3); clean = repmat(linspace(0.2, 0.8, 64), 64, 1); noisy = imnoise(clean, 'gaussian', 0, 0.01);")
    assert im.num("noisy").std() > 0.15
    assert scalar(im, "immse(wiener2(noisy, [5 5]), clean)") < scalar(im, "immse(noisy, clean)") / 3
    assert scalar(im, "std2(noisy - clean)") == pytest.approx(0.1, rel=0.1)
    im.run("sp = imnoise(uint8(128 * ones(100)), 'salt & pepper', 0.1);")
    sp = im.num("sp")
    assert sp.dtype == np.uint8 and set(np.unique(sp)) == {0, 128, 255}
    assert 0.06 < (sp != 128).mean() < 0.14
    assert im.num("imnoise(A, 'speckle')").shape == im.A.shape


def test_padarray(m):
    assert np.array_equal(m.num("padarray([1 2; 3 4], [1 2])"), np.pad([[1, 2], [3, 4]], ((1, 1), (2, 2))))
    assert np.array_equal(m.num("padarray([1 2; 3 4], [1 0], 9, 'pre')"), [[9, 9], [1, 2], [3, 4]])
    assert np.array_equal(m.num("padarray([1 2 3], [0 2], 'replicate', 'post')"), [[1, 2, 3, 3, 3]])
    assert np.array_equal(m.num("padarray([1 2 3], [0 2], 'symmetric')"), [[2, 1, 1, 2, 3, 3, 2]])
    assert np.array_equal(m.num("padarray([1 2 3], [0 2], 'circular')"), [[2, 3, 1, 2, 3, 1, 2]])
    assert np.array_equal(m.num("padarray([1 2 3], 1)"), [[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    assert m.num("padarray(uint8(ones(2, 2, 3)), [1 1])").shape == (4, 4, 3)
    assert m.num("padarray(ones(2), [1 1 1])").shape == (4, 4, 3)
    assert m.num("padarray(uint8([1 2]), [0 1])").dtype == np.uint8


def test_im2col(m):
    A = np.arange(1, 13).reshape(3, 4, order="F")
    m.interp.base.set("A", A.astype(float))
    s = m.num("im2col(A, [2 2])")
    assert s.shape == (4, 6)
    assert np.array_equal(s[:, 0], [1, 2, 4, 5]) and np.array_equal(s[:, 1], [2, 3, 5, 6])
    assert np.array_equal(s[:, 5], [8, 9, 11, 12])
    d = m.num("im2col(A, [2 2], 'distinct')")
    assert d.shape == (4, 4) and np.array_equal(d[:, 0], [1, 2, 4, 5]) and np.array_equal(d[:, 1], [3, 0, 6, 0])


# -- edges -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["sobel", "prewitt", "roberts", "canny", "log"])
def test_edge_finds_a_step(m, method):
    E = np.zeros((20, 20))
    E[:, 10:] = 1
    m.interp.base.set("E", E)
    bw = m.num(f"edge(E, '{method}')")
    assert bw.dtype == bool and bw.shape == (20, 20)
    cols = np.flatnonzero(bw.any(axis=0))
    assert len(cols) == 1 and cols[0] in (9, 10)       # one pixel wide, at the step
    assert bw[:, cols[0]].sum() >= 18
    assert not m.num(f"edge(zeros(12), '{method}')").any()


def test_edge_options(m):
    m.run("I = zeros(20); I(6:15, 6:15) = 1;")
    both = m.num("edge(I)")
    assert both.sum() > 20 and not both[10, 10] and not both[0, 0]
    horiz = m.num("edge(I, 'sobel', [], 'horizontal')")
    vert = m.num("edge(I, 'sobel', [], 'vertical')")
    assert horiz[4:6, 10].any() and not horiz[10, 4:7].any()
    assert vert[10, 4:6].any() and not vert[4:6, 10].any()
    m.run("[bw, t] = edge(I, 'sobel');")
    assert 0 < scalar(m, "t") < 1
    assert m.num("edge(I, 'sobel', 10)").sum() == 0
    assert m.num("edge(I, 'sobel', 0.01, 'nothinning')").sum() > both.sum()
    m.run("[bw, t] = edge(I, 'canny', [0.1 0.3]);")
    assert np.allclose(m.num("t"), [[0.1, 0.3]])
    assert m.num("edge(uint8(I * 255))").sum() == both.sum()
    with pytest.raises(MatlabError):
        m.run("edge(I, 'fancy')")


def test_canny_circle_outline(m):
    m.run("[x, y] = meshgrid(1:60); I = double((x - 30).^2 + (y - 30).^2 < 15^2); bw = edge(I, 'canny');")
    bw = m.num("bw")
    yy, xx = np.nonzero(bw)
    r = np.hypot(xx + 1 - 30, yy + 1 - 30)
    assert bw.sum() > 60 and np.all(np.abs(r - 15) < 2.5)


# -- morphology -----------------------------------------------------------------------------------

def test_strel_shapes(m):
    assert np.array_equal(m.num("strel('disk', 1)"), [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    d2 = m.num("strel('disk', 2)")
    assert d2.dtype == bool and np.array_equal(d2.astype(int), [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                                                             [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    y, x = np.mgrid[-5:6, -5:6]
    assert np.array_equal(m.num("strel('disk', 5)"), x * x + y * y <= 25)
    assert np.array_equal(m.num("strel('diamond', 2)"), np.abs(x[3:8, 3:8]) + np.abs(y[3:8, 3:8]) <= 2)
    assert np.array_equal(m.num("strel('square', 3)"), np.ones((3, 3), bool))
    assert np.array_equal(m.num("strel('rectangle', [2 4])"), np.ones((2, 4), bool))
    assert np.array_equal(m.num("strel('line', 5, 0)"), np.ones((1, 5), bool))
    assert np.array_equal(m.num("strel('line', 3, 90)"), np.ones((3, 1), bool))
    assert np.array_equal(m.num("strel('line', 5, 45)"), np.eye(3, dtype=bool)[::-1])
    assert np.array_equal(m.num("strel('line', 5, -45)"), np.eye(3, dtype=bool))
    assert np.array_equal(m.num("strel([0 1 0; 1 1 1])"), [[0, 1, 0], [1, 1, 1]])
    assert np.array_equal(m.num("getnhood(strel('square', 2))"), np.ones((2, 2), bool))
    assert m.num("strel('octagon', 3)").shape == (7, 7)
    with pytest.raises(MatlabError):
        m.run("strel('blob', 3)")


def _brute_morph(a, se, dilate):
    kr, kc = se.shape
    cr, cc = (kr - 1) // 2, (kc - 1) // 2
    R, C = a.shape
    out = np.zeros_like(a)
    for i in range(R):
        for j in range(C):
            vals = []
            for p in range(kr):
                for q in range(kc):
                    if not se[p, q]:
                        continue
                    ii, jj = (i - (p - cr), j - (q - cc)) if dilate else (i + (p - cr), j + (q - cc))
                    if 0 <= ii < R and 0 <= jj < C:
                        vals.append(a[ii, jj])
            out[i, j] = (max(vals) if dilate else min(vals)) if vals else (0 if dilate else 255)
    return out


@pytest.mark.parametrize("se", [np.ones((3, 3), bool), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool),
                                np.ones((2, 2), bool), np.array([[1, 0, 0], [0, 1, 1]], bool),
                                np.ones((1, 4), bool), np.ones((4, 2), bool)])
def test_grayscale_morphology_against_definition(im, se):
    put(im, "se", se)
    assert np.array_equal(im.num("imdilate(A, se)"), _brute_morph(im.A, se, True))
    assert np.array_equal(im.num("imerode(A, se)"), _brute_morph(im.A, se, False))


def test_binary_morphology(im):
    se = np.ones((3, 3), bool)
    put(im, "se", se)
    er = ndimage.binary_erosion(im.BW, se, border_value=1)
    di = ndimage.binary_dilation(im.BW, se)
    assert im.num("imdilate(BW, se)").dtype == bool
    assert np.array_equal(im.num("imdilate(BW, se)"), di)
    assert np.array_equal(im.num("imerode(BW, se)"), er)
    assert np.array_equal(im.num("imopen(BW, se)"), ndimage.binary_dilation(er, se))
    assert np.array_equal(im.num("imclose(BW, se)"), ndimage.binary_erosion(di, se, border_value=1))
    assert np.array_equal(im.num("imdilate(BW, strel('disk', 1))"), ndimage.binary_dilation(im.BW))
    assert np.all(im.num("imopen(A, ones(3))") <= im.A) and np.all(im.num("imclose(A, ones(3))") >= im.A)
    assert np.array_equal(im.num("imtophat(A, ones(3))"), im.A - im.num("imopen(A, ones(3))"))
    assert np.array_equal(im.num("imbothat(A, ones(3))"), im.num("imclose(A, ones(3))") - im.A)
    assert np.array_equal(im.num("imdilate(D, se)"), ndimage.grey_dilation(im.D, size=3, mode="constant", cval=-np.inf))
    assert im.num("imdilate(RGB, se)").shape == im.RGB.shape


def test_bwlabel(im):
    im.run("[L, n] = bwlabel(BW);")
    L, n = im.num("L"), int(scalar(im, "n"))
    ref, nref = ndimage.label(im.BW, np.ones((3, 3)))
    assert L.dtype == np.float64 and n == nref
    assert len(set(zip(L[im.BW].astype(int), ref[im.BW]))) == n          # same grouping
    firsts = [np.flatnonzero((L == k).ravel(order="F"))[0] for k in range(1, n + 1)]
    assert firsts == sorted(firsts)                                       # numbered down the columns
    assert int(im.num("bwlabel(BW, 4)").max()) == ndimage.label(im.BW)[1]
    assert np.array_equal(im.num("bwlabel([1 0 1; 0 1 0; 1 0 1], 4)"), [[1, 0, 4], [0, 3, 0], [2, 0, 5]])
    assert np.array_equal(im.num("bwlabel([1 0 1; 0 1 0; 1 0 1])"), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    with pytest.raises(MatlabError):
        im.run("bwlabel(BW, 6)")


def test_bwconncomp_and_labelmatrix(im):
    im.run("CC = bwconncomp(BW); L = bwlabel(BW);")
    assert scalar(im, "CC.NumObjects") == im.num("L").max()
    assert np.array_equal(im.num("CC.ImageSize"), [[14, 16]])
    assert np.array_equal(im.num("double(labelmatrix(CC))"), im.num("L"))
    assert np.array_equal(im.num("CC.PixelIdxList{2}").ravel(), im.num("find(L == 2)").ravel())
    assert scalar(im, "numel(regionprops(CC))") == im.num("L").max()


def test_bwareaopen_bwperim_imfill(im):
    L, n = ndimage.label(im.BW, np.ones((3, 3)))
    sizes = np.bincount(L.ravel())
    keep = np.isin(L, [k for k in range(1, n + 1) if sizes[k] >= 4])
    assert np.array_equal(im.num("bwareaopen(BW, 4)"), keep)
    assert np.array_equal(im.num("bwperim(true(4))"), [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    im.run("R = false(7); R(2:6, 2:6) = true; R(4, 4) = false;")
    assert np.array_equal(im.num("imfill(R, 'holes')"), np.pad(np.ones((5, 5), bool), 1))
    assert im.num("bwperim(R)").sum() == 16 + 4
    assert np.array_equal(im.num("imfill(BW, 'holes')"), ndimage.binary_fill_holes(im.BW))
    im.run("Q = false(5); Q(:, 3) = true;")
    assert np.array_equal(im.num("imfill(Q, [1 1])"), np.array([[1, 1, 1, 0, 0]] * 5, bool))


def test_bwdist(im):
    assert np.allclose(im.num("bwdist(BW)"), ndimage.distance_transform_edt(~im.BW))
    assert im.num("bwdist(BW)").dtype == np.float32
    assert np.array_equal(im.num("bwdist([1 0 0 0])"), [[0, 1, 2, 3]])
    assert np.array_equal(im.num("bwdist([1 0; 0 0], 'cityblock')"), [[0, 1], [1, 2]])
    assert np.array_equal(im.num("bwdist([1 0; 0 0], 'chessboard')"), [[0, 1], [1, 1]])
    im.run("[d, idx] = bwdist([0 0 1; 0 0 0]);")
    assert np.array_equal(im.num("idx"), [[5, 5, 5], [5, 5, 5]])


def test_regionprops_basic(im):
    L, n = im.num("bwlabel(BW)"), int(im.num("bwlabel(BW)").max())
    im.run("s = regionprops(BW, 'Area', 'Centroid', 'BoundingBox', 'PixelIdxList');")
    assert im.num("size(s)").tolist() == [[n, 1]]
    assert np.array_equal(im.num("[s.Area]").ravel(), np.bincount(L[im.BW].astype(int))[1:])
    com = np.array(ndimage.center_of_mass(im.BW, L, range(1, n + 1)))
    assert np.allclose(im.num("cat(1, s.Centroid)"), com[:, ::-1] + 1)
    bb = im.num("cat(1, s.BoundingBox)")
    for k, sl in enumerate(ndimage.find_objects(L.astype(int))):
        assert np.allclose(bb[k], [sl[1].start + 0.5, sl[0].start + 0.5, sl[1].stop - sl[1].start,
                                   sl[0].stop - sl[0].start])
    assert np.array_equal(im.num("s(3).PixelIdxList").ravel(), np.flatnonzero((L == 3).ravel(order="F")) + 1)
    im.run("b = regionprops(BW);")
    assert im.str("strjoin(fieldnames(b)', ',')") == "Area,Centroid,BoundingBox"
    assert scalar(im, "numel(regionprops(bwlabel(BW), 'Area'))") == n


def test_regionprops_shape_measurements(m):
    m.run("R = false(20, 30); R(5:14, 8:27) = true; s = regionprops(R, 'all');")
    assert scalar(m, "s.Area") == 200
    assert np.allclose(m.num("s.Centroid"), [[17.5, 9.5]])
    assert np.allclose(m.num("s.BoundingBox"), [[7.5, 4.5, 20, 10]])
    assert scalar(m, "s.Extent") == 1 and scalar(m, "s.EulerNumber") == 1
    assert scalar(m, "s.EquivDiameter") == pytest.approx(np.sqrt(800 / np.pi))
    assert scalar(m, "s.Orientation") == pytest.approx(0)
    # an axis-aligned w-by-h rectangle has second moments w^2/12 and h^2/12
    assert scalar(m, "s.MajorAxisLength") == pytest.approx(4 * np.sqrt(400 / 12))
    assert scalar(m, "s.MinorAxisLength") == pytest.approx(4 * np.sqrt(100 / 12))
    assert scalar(m, "s.Eccentricity") == pytest.approx(np.sqrt(1 - 100 / 400))
    assert scalar(m, "s.Perimeter") == pytest.approx(2 * (19 + 9), rel=0.05)
    assert m.num("s.Image").shape == (10, 20) and m.num("s.Image").all()
    assert np.array_equal(m.num("size(s.PixelList)"), [[200, 2]])
    assert np.array_equal(m.num("imcrop(double(R), s.BoundingBox)"), np.ones((10, 20)))
    m.run("T = false(20); T(3:12, 9) = true; t = regionprops(T, 'Orientation', 'Perimeter');")
    assert abs(scalar(m, "t.Orientation")) == pytest.approx(90)
    m.run("[x, y] = meshgrid(1:80); C = (x - 40).^2 + (y - 40).^2 <= 30^2; c = regionprops(C, 'Perimeter', "
          "'Circularity', 'Eccentricity', 'FilledArea');")
    assert scalar(m, "c.Perimeter") == pytest.approx(2 * np.pi * 30, rel=0.03)
    assert scalar(m, "c.Circularity") == pytest.approx(1, abs=0.06) and scalar(m, "c.Eccentricity") < 0.05
    m.run("C(40, 40) = false; h = regionprops(C, 'EulerNumber', 'FilledArea', 'Area');")
    assert scalar(m, "h.EulerNumber") == 0 and scalar(m, "h.FilledArea") == scalar(m, "h.Area") + 1


def test_regionprops_intensity_and_errors(m):
    m.run("L = [1 1 0 2; 1 0 0 2]; I = uint8([10 20 0 50; 30 0 0 70]);")
    m.run("s = regionprops(L, I, 'MeanIntensity', 'MaxIntensity', 'MinIntensity', 'PixelValues', 'WeightedCentroid');")
    assert np.allclose(m.num("[s.MeanIntensity]"), [[20, 60]])
    assert np.array_equal(m.num("[s.MaxIntensity]"), [[30, 70]]) and m.num("s(1).MaxIntensity").dtype == np.uint8
    assert np.array_equal(m.num("s(1).PixelValues").ravel(), [10, 30, 20])
    assert np.allclose(m.num("s(2).WeightedCentroid"), [[4, (50 + 140) / 120]])
    m.run("e = regionprops([0 3], 'Area');")
    assert np.array_equal(m.num("[e.Area]"), [[0, 0, 1]])
    assert scalar(m, "numel(regionprops(false(3)))") == 0
    with pytest.raises(MatlabError, match="not supported"):
        m.run("regionprops(L, 'ConvexHull')")
    with pytest.raises(MatlabError, match="MeanIntensity"):
        m.run("regionprops(L, 'MeanIntensity')")


# -- geometry --------------------------------------------------------------------------------------

def test_imresize_known_values(m):
    assert np.allclose(m.num("imresize([0 10], [1 4], 'bilinear')"), [[0, 2.5, 7.5, 10]])
    assert np.array_equal(m.num("imresize([1 2; 3 4], 2, 'nearest')"), np.kron([[1, 2], [3, 4]], np.ones((2, 2))))
    assert np.array_equal(m.num("imresize(1:4, 0.5, 'nearest')"), [[2, 4]])
    assert np.allclose(m.num("imresize([1 2 3 4], 0.5, 'bilinear')"), [[1.625, 3.375]])   # widened kernel
    assert np.allclose(m.num("imresize([1 2 3 4], 0.5, 'bilinear', 'Antialiasing', false)"), [[1.5, 3.5]])
    assert np.allclose(m.num("imresize(ones(7, 9), 0.37)"), 1)
    assert np.allclose(m.num("imresize(magic(4), 1)"), m.num("magic(4)"))
    # a linear ramp stays linear away from the borders
    r = m.num("imresize(repmat(1:10, 3, 1), [3 20], 'bicubic')")
    assert np.allclose(np.diff(r[1, 4:16]), 0.5)


@pytest.mark.parametrize("method,pil", [("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC),
                                        ("lanczos3", Image.LANCZOS)])
def test_imresize_matches_pillow_resampling(im, method, pil):
    F = im.rng.random((40, 50)).astype(np.float32)
    put(im, "F", F.astype(float))
    for size in [(20, 25), (80, 100), (13, 31), (64, 20)]:
        got = im.num(f"imresize(F, [{size[0]} {size[1]}], '{method}')")
        ref = np.asarray(Image.fromarray(F).resize((size[1], size[0]), pil))
        assert got.shape == size
        assert np.abs(got - ref)[6:-6, 6:-6].max() < 2e-4      # borders are handled differently


def test_imresize_sizes_and_classes(im):
    assert im.num("imresize(A, 0.5)").shape == (9, 12) and im.num("imresize(A, 0.5)").dtype == np.uint8
    assert im.num("imresize(A, [NaN 46])").shape == (34, 46)
    assert im.num("imresize(A, [34 NaN])").shape == (34, 46)
    assert im.num("imresize(RGB, 2)").shape == (18, 22, 3)
    assert im.num("imresize(A, [5 7], 'nearest')").shape == (5, 7)
    assert im.num("imresize(A, 'Scale', [2 3])").shape == (34, 69)
    assert im.num("imresize(A, 'OutputSize', [4 4], 'Method', 'bilinear')").shape == (4, 4)
    assert im.num("imresize(BW, 2)").dtype == bool
    assert set(np.unique(im.num("imresize(A, 3, 'nearest')"))) <= set(np.unique(im.A))
    big = im.num("imresize(A, 3)")
    assert big.dtype == np.uint8 and abs(big.mean() - im.A.mean()) < 3
    # shrinking with antialiasing averages; without it, it samples
    im.run("C = repmat([0 1], 8, 8);")
    assert np.allclose(im.num("imresize(C, 0.5, 'bilinear')")[:, 1:-1], 0.5)
    assert im.num("imresize(C, 0.5, 'bilinear', 'Antialiasing', false)").std() == pytest.approx(0)
    with pytest.raises(MatlabError):
        im.run("imresize(A, 2, 'sinc')")
    with pytest.raises(MatlabError):
        im.run("imresize(A, [NaN NaN])")


def test_imrotate(im):
    assert np.array_equal(im.num("imrotate(A, 90)"), np.rot90(im.A))
    assert np.array_equal(im.num("imrotate(A, -90)"), np.rot90(im.A, -1))
    assert np.array_equal(im.num("imrotate(A, 180)"), im.A[::-1, ::-1])
    assert np.array_equal(im.num("imrotate(RGB, 270)"), np.rot90(im.RGB, 3))
    assert im.num("imrotate(A, 30, 'bilinear', 'crop')").shape == im.A.shape
    loose = im.num("imrotate(A, 30, 'bilinear')")
    assert loose.dtype == np.uint8 and loose.shape == (27, 29) and loose[0, 0] == 0
    # rotating a centred square by 45 degrees puts a corner on the horizontal axis
    im.run("S = zeros(41); S(11:31, 11:31) = 1; R = imrotate(S, 45, 'nearest', 'crop');")
    R = im.num("R")
    assert R[20, 7] == 1 and im.num("S")[20, 7] == 0 and R[11, 11] == 0
    assert abs(R.sum() - 441) < 25
    # counter-clockwise: a point right of centre moves up
    im.run("P = zeros(21); P(11, 18) = 1; Q = imrotate(P, 90, 'bilinear', 'crop');")
    assert im.num("Q")[3, 10] == 1


def test_imcrop(im):
    assert np.array_equal(im.num("imcrop(A, [2 3 4 5])"), im.A[2:8, 1:6])
    assert np.array_equal(im.num("imcrop(A, [0.5 0.5 10 10])"), im.A[:10, :10])
    assert np.array_equal(im.num("imcrop(RGB, [3 2 100 100])"), im.RGB[1:, 2:])
    im.run("[J, r] = imcrop(A, [2 3 4 5]);")
    assert np.array_equal(im.num("r"), [[2, 3, 4, 5]])
    with pytest.raises(MatlabError, match="mouse"):
        im.run("imcrop(A)")


def test_imtranslate(m):
    assert np.array_equal(m.num("imtranslate([1 2 3; 4 5 6], [1 0])"), [[0, 1, 2], [0, 4, 5]])
    assert np.array_equal(m.num("imtranslate([1 2 3; 4 5 6], [0 -1], 'FillValues', 9)"), [[4, 5, 6], [9, 9, 9]])


# -- files -----------------------------------------------------------------------------------------

def test_png_round_trips(im, tmp_path):
    put(im, "d", str(tmp_path))
    im.run("imwrite(A, fullfile(d, 'a.png')); B = imread(fullfile(d, 'a.png'));")
    assert im.num("B").dtype == np.uint8 and np.array_equal(im.num("B"), im.A)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "a.png")), im.A)
    im.run("imwrite(RGB, fullfile(d, 'c.png')); C = imread(fullfile(d, 'c.png'), 'png');")
    assert np.array_equal(im.num("C"), im.RGB) and im.num("C").shape == (9, 11, 3)
    im.run("imwrite(BW, fullfile(d, 'b.png')); Bb = imread(fullfile(d, 'b.png'));")
    assert im.num("Bb").dtype == bool and np.array_equal(im.num("Bb"), im.BW)
    im.run("imwrite(D, fullfile(d, 'd.png')); Dd = imread(fullfile(d, 'd.png'));")
    assert np.array_equal(im.num("Dd"), np.floor(im.D * 255 + 0.5))
    im.run("imwrite(uint16(A) * 257, fullfile(d, 'u.png')); U = imread(fullfile(d, 'u.png'));")
    assert im.num("U").dtype == np.uint16 and np.array_equal(im.num("U"), im.A.astype(np.uint16) * 257)


def test_other_formats(im, tmp_path):
    put(im, "d", str(tmp_path))
    im.run("d = char(d);")
    for ext in ("bmp", "tif", "ppm"):
        im.run(f"imwrite(RGB, fullfile(d, 'c.{ext}')); C = imread(fullfile(d, 'c.{ext}'));")
        assert np.array_equal(im.num("C"), im.RGB), ext
    im.run("imwrite(uint16(A) * 100, fullfile(d, 'u.tif')); U = imread(fullfile(d, 'u.tif'));")
    assert im.num("U").dtype == np.uint16 and np.array_equal(im.num("U"), im.A.astype(np.uint16) * 100)
    im.run("imwrite(A, fullfile(d, 'noext'), 'png'); N = imread(fullfile(d, 'noext'));")
    assert np.array_equal(im.num("N"), im.A)
    with pytest.raises(MatlabError, match="format"):
        im.run("imwrite(A, fullfile(d, 'noext2'))")


def test_jpeg_quality(im, tmp_path):
    y, x = np.mgrid[0:64, 0:64]
    smooth = np.stack([x * 4, y * 4, (x + y) * 2], axis=-1).clip(0, 255).astype(np.uint8)
    put(im, "S", smooth)
    put(im, "d", str(tmp_path))
    im.run("imwrite(S, fullfile(d, 'hi.jpg'), 'Quality', 95); imwrite(S, fullfile(d, 'lo.jpg'), 'Quality', 5);")
    assert os.path.getsize(tmp_path / "hi.jpg") > os.path.getsize(tmp_path / "lo.jpg")
    im.run("H = imread(fullfile(d, 'hi.jpg')); L = imread(fullfile(d, 'lo.jpg'));")
    assert im.num("H").shape == (64, 64, 3) and im.num("H").dtype == np.uint8
    err_hi = np.abs(im.num("H").astype(float) - smooth).mean()
    err_lo = np.abs(im.num("L").astype(float) - smooth).mean()
    assert err_hi < 3 < err_lo
    im.run("imwrite(rgb2gray(S), fullfile(d, 'g.jpg')); G = imread(fullfile(d, 'g.jpg'));")
    assert im.num("G").ndim == 2
    with pytest.raises(MatlabError):
        im.run("imwrite(S, fullfile(d, 'x.jpg'), 'Quality', 500)")


def test_indexed_and_alpha_files(im, tmp_path):
    put(im, "d", str(tmp_path))
    im.run("[X, map] = rgb2ind(RGB, 16); imwrite(X, map, fullfile(d, 'i.png'));")
    im.run("[X2, map2] = imread(fullfile(d, 'i.png'));")
    assert im.num("X2").dtype == np.uint8 and np.array_equal(im.num("X2"), im.num("X"))
    assert np.allclose(im.num("map2"), np.round(im.num("map") * 255) / 255)
    assert np.asarray(Image.open(tmp_path / "i.png").convert("RGB")).shape == (9, 11, 3)
    im.run("al = uint8(randi(255, 9, 11)); imwrite(RGB, fullfile(d, 'al.png'), 'Alpha', al);")
    im.run("[P, m3, al2] = imread(fullfile(d, 'al.png'));")
    assert np.array_equal(im.num("al2"), im.num("al")) and np.array_equal(im.num("P"), im.RGB)
    assert im.num("m3").size == 0
    im.run("[g, gm, ga] = imread(fullfile(d, 'i.png'));")
    assert im.num("ga").size == 0
    # one-based double indices are written as the same colours
    im.run("imwrite(double(X) + 1, map, fullfile(d, 'j.png')); [X3, map3] = imread(fullfile(d, 'j.png'));")
    assert np.array_equal(im.num("X3"), im.num("X"))


def test_animated_gif_built_by_appending_frames(m, tmp_path):
    f = tmp_path / "anim.gif"
    m.interp.set_var("f", str(f))
    m.run("""
for k = 1:5
    im = uint8(zeros(20, 30, 3)); im(:, 1:k*5, 1) = 255; im(:, :, 3) = 40*k;
    [X, map] = rgb2ind(im, 256);
    if k == 1
        imwrite(X, map, f, 'gif', 'LoopCount', Inf, 'DelayTime', 0.2);
    else
        imwrite(X, map, f, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
    end
end
info = imfinfo(f);
""")
    g = Image.open(f)
    assert g.n_frames == 5 and g.info.get("loop") == 0 and g.info.get("duration") == 200
    for k in range(5):
        g.seek(k)
        expected = np.zeros((20, 30, 3), np.uint8)
        expected[:, :(k + 1) * 5, 0] = 255
        expected[:, :, 2] = 40 * (k + 1)
        assert np.array_equal(np.asarray(g.convert("RGB")), expected)
    assert scalar(m, "numel(info)") == 5 and scalar(m, "info(1).DelayTime") == 20
    assert m.str("info(2).Format") == "gif" and m.str("info(1).ColorType") == "indexed"
    m.run("[X3, map3] = imread(f, 3); every = imread(f, 'Frames', 'all');")
    assert m.num("every").shape == (20, 30, 1, 5)
    expected = np.zeros((20, 30, 3))
    expected[:, :15, 0] = 1
    expected[:, :, 2] = 120 / 255
    assert np.allclose(m.num("ind2rgb(X3, map3)"), expected)
    with pytest.raises(MatlabError):
        m.run("imread(f, 9)")


def test_gif_rgb_frames_grayscale_and_append_to_foreign_file(m, tmp_path):
    f = tmp_path / "g.gif"
    m.interp.set_var("f", str(f))
    m.run("imwrite(uint8(magic(8) * 3), f); imwrite(uint8(magic(8)), f, 'WriteMode', 'append');")
    g = Image.open(f)
    assert g.n_frames == 2
    assert np.array_equal(np.asarray(g.convert("L")), (np.asarray(m.num("magic(8)")) * 3).astype(np.uint8))
    # a new session has no memory of the file and must read the frames back before appending
    from mylab import Interpreter
    other = Interpreter(out=lambda s: None)
    other.base.set("f", m.interp.base.get("f"))
    other.execute("imwrite(uint8(zeros(8)), f, 'WriteMode', 'append', 'DelayTime', 1);")
    g = Image.open(f)
    assert g.n_frames == 3
    g.seek(2)
    assert not np.asarray(g.convert("L")).any()
    with pytest.raises(MatlabError, match="does not exist"):
        m.run("imwrite(uint8(zeros(4)), [f '.missing.gif'], 'WriteMode', 'append');")
    m.run("imwrite(uint8(cat(4, zeros(4, 4), 255 * ones(4, 4))), [f '.4d.gif'], 'DelayTime', 0.05, 'LoopCount', 3);")
    g = Image.open(str(f) + ".4d.gif")
    assert g.n_frames == 2 and g.info.get("loop") == 3


def test_gif_from_figure_frames(m, tmp_path):
    f = tmp_path / "plot.gif"
    m.interp.set_var("f", str(f))
    m.run("""
for k = 1:3
    plot(1:10, (1:10).^k);
    fr = getframe(gcf);
    img = frame2im(fr);
    [X, map] = rgb2ind(img, 256);
    if k == 1
        imwrite(X, map, f, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(X, map, f, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
close all
""")
    g = Image.open(f)
    assert g.n_frames == 3 and g.size[0] > 100
    img = m.num("img")
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert scalar(m, "isstruct(im2frame(img))") == 1


def test_imfinfo(im, tmp_path):
    put(im, "d", str(tmp_path))
    im.run("imwrite(RGB, fullfile(d, 'c.png')); imwrite(A, fullfile(d, 'a.jpg')); "
           "p = imfinfo(fullfile(d, 'c.png')); j = imfinfo(fullfile(d, 'a.jpg'));")
    assert scalar(im, "p.Width") == 11 and scalar(im, "p.Height") == 9 and scalar(im, "p.BitDepth") == 24
    assert im.str("p.ColorType") == "truecolor" and im.str("p.Format") == "png"
    assert im.str("j.ColorType") == "grayscale" and im.str("j.Format") == "jpg" and scalar(im, "j.BitDepth") == 8
    assert scalar(im, "p.FileSize") == os.path.getsize(tmp_path / "c.png")
    assert im.str("p.Filename") == str(tmp_path / "c.png")


def test_imread_errors_and_search_path(im, tmp_path):
    with pytest.raises(MatlabError, match="does not exist"):
        im.run("imread('cameraman.tif')")
    (tmp_path / "notimage.png").write_text("hello")
    put(im, "d", str(tmp_path))
    with pytest.raises(MatlabError, match="not an image"):
        im.run("imread(fullfile(char(d), 'notimage.png'))")
    im.run("imwrite(A, fullfile(d, 'onpath.png')); addpath(d); B = imread('onpath.png'); rmpath(d);")
    assert np.array_equal(im.num("B"), im.A)
    im.run("C = imread(fullfile(d, 'onpath.png'), 'PixelRegion', {[2 5], [3 2 9]});")
    assert np.array_equal(im.num("C"), im.A[1:5, 2:9:2])


def test_typical_coursework_pipeline(m, tmp_path):
    """Threshold, clean up, label and measure blobs in a synthetic picture read from disk."""
    y, x = np.mgrid[0:100, 0:120]
    pic = np.full((100, 120), 40.0)
    for cy, cx, r in [(30, 30, 12), (65, 80, 18), (20, 95, 6)]:
        pic[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = 210
    pic += np.random.default_rng(0).normal(0, 6, pic.shape)
    Image.fromarray(np.clip(pic, 0, 255).astype(np.uint8)).save(tmp_path / "blobs.png")
    m.interp.set_var("f", str(tmp_path / "blobs.png"))
    m.run("""
I = imread(char(f));
I = imgaussfilt(I, 1);
bw = imbinarize(I, graythresh(I));
bw = imopen(bw, strel('disk', 2));
bw = bwareaopen(bw, 20);
[L, n] = bwlabel(bw);
s = regionprops(L, 'Area', 'Centroid', 'EquivDiameter');
[~, order] = sort([s.Area], 'descend');
c = cat(1, s(order).Centroid);
rgb = label2rgb(L);
small = imresize(rgb, 0.5);
imwrite(small, [char(f) '.out.jpg']);
""")
    assert scalar(m, "n") == 3
    assert np.allclose(m.num("c"), [[81, 66], [31, 31], [96, 21]], atol=1)
    areas = np.sort(m.num("[s.Area]").ravel())[::-1]
    assert np.allclose(areas, np.pi * np.array([18, 12, 6]) ** 2, rtol=0.12)
    assert m.num("small").shape == (50, 60, 3)
    assert os.path.getsize(str(tmp_path / "blobs.png") + ".out.jpg") > 0


def test_rotate_colour_image_with_interpolation(im):
    out = im.num("imrotate(RGB, 33, 'bilinear')")
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8 and out.shape[0] > 9
    assert np.array_equal(out[..., 0], im.num("imrotate(RGB(:, :, 1), 33, 'bilinear')"))


def test_bwboundaries(m):
    m.run("bw = false(9); bw(3:7, 3:7) = true; bw(1, 9) = true; [B, L, n] = bwboundaries(bw);")
    assert scalar(m, "n") == 2 and m.num("size(B)").tolist() == [[2, 1]]
    pts = m.num("B{1}")
    assert pts.shape == (17, 2) and np.array_equal(pts[0], pts[-1]) and pts[0].tolist() == [3, 3]
    assert pts[1].tolist() == [3, 4]                                  # clockwise: first step goes right
    assert set(map(tuple, pts.astype(int))) == {(r, c) for r in range(3, 8) for c in range(3, 8)
                                                 if r in (3, 7) or c in (3, 7)}
    assert m.num("B{2}").tolist() == [[1, 9], [1, 9]]
    assert np.array_equal(m.num("L"), m.num("bwlabel(bw)"))
    x, y = pts[:, 1], pts[:, 0]
    assert abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2 == 16      # area enclosed by the pixel centres


def test_poly2mask(m):
    mask = m.num("poly2mask([2.5 6.5 6.5 2.5], [1.5 1.5 4.5 4.5], 6, 8)")
    expected = np.zeros((6, 8), bool)
    expected[1:4, 2:6] = True
    assert mask.dtype == bool and np.array_equal(mask, expected)
    tri = m.num("poly2mask([1 20 1], [1 1 20], 20, 20)")
    assert abs(tri.sum() - 200) < 25 and tri[1, 1] and not tri[18, 18]


def test_bwmorph(m):
    m.run("bar = false(7, 12); bar(3:5, 2:11) = true; t = bwmorph(bar, 'thin', Inf);")
    t = m.num("t")
    assert t.sum(axis=0).max() == 1 and t[3].sum() >= 6 and not t[[2, 4]].any()      # a one-pixel line
    assert m.num("bwmorph(t, 'endpoints')").sum() == 2
    assert np.array_equal(m.num("bwmorph(bar, 'skel', Inf)"), t)
    once = m.num("bwmorph(bar, 'thin')")
    assert t.sum() <= once.sum() < 30
    m.run("blob = false(15, 30); blob(3:13, 3:28) = true;")
    assert m.num("bwmorph(blob, 'thin', Inf)").sum() < m.num("bwmorph(blob, 'thin')").sum() < 286
    m.run("sq = false(7); sq(2:6, 2:6) = true;")
    assert m.num("bwmorph(sq, 'remove')").sum() == 16
    assert np.array_equal(m.num("bwmorph(sq, 'erode')"), np.pad(np.ones((3, 3), bool), 2))
    assert np.array_equal(m.num("bwmorph(sq, 'dilate')"), np.ones((7, 7), bool))
    m.run("n = false(5); n(1, 1) = true; n(3:4, 3:4) = true; h = true(5); h(3, 3) = false;")
    assert m.num("bwmorph(n, 'clean')").sum() == 4 and m.num("bwmorph(h, 'fill')").all()
    assert m.num("bwmorph([1 0 1], 'bridge')").all()
    assert m.num("bwmorph([1 1 0; 1 0 0; 0 0 0], 'majority')").sum() == 0
    with pytest.raises(MatlabError, match="not supported"):
        m.run("bwmorph(sq, 'shrink')")


def test_imshow_integer_gray_logical_and_indexed(m, tmp_path):
    out = tmp_path / "shown.png"
    m.run(f"""
rgb = uint8(cat(3, 255 * ones(4, 6), zeros(4, 6), zeros(4, 6)));
subplot(2, 2, 1); h = imshow(rgb);
subplot(2, 2, 2); imshow(uint8([0 128; 255 64]));
subplot(2, 2, 3); imshow([1 2; 3 4], []);
subplot(2, 2, 4); imshow(logical([1 0; 0 1]));
saveas(gcf, '{out}');
""")
    assert out.exists()
    st = m.interp.state["plot"]
    axes = st.figures[st.current].axes
    rgb_data = axes[0].images[0].get_array()
    assert rgb_data.dtype == np.uint8 and rgb_data[0, 0, 0] == 255
    assert axes[1].images[0].get_clim() == (0.0, 255.0)
    assert axes[2].images[0].get_clim() == (1.0, 4.0)
    assert axes[3].images[0].get_clim() == (0.0, 1.0)
    m.run("close all")


def test_uint64_top_of_range(m):
    assert int(m.val("x").item()) == 18446744073709551615 if m.run("x = uint64(18446744073709551615);") == "" else False
    assert int(m.num("intmax('int64')").item()) == 9223372036854775807
