"""Feature detection, point objects, descriptors and matching (Computer Vision functions)."""

import numpy as np
import pytest

from mylab.errors import MatlabError

cv2 = pytest.importorskip("cv2")

SCENE = r"""
rng(1); I = zeros(200, 240);
for k = 1:25
  x = randi([20 220]); y = randi([20 180]); w = randi([6 18]); h = randi([6 18]);
  I(y:y+h, x:x+w) = rand*0.8 + 0.2;
end
I = imgaussfilt(I, 1);
J = imtranslate(I, [7 4]);
"""


@pytest.fixture
def scene(m):
    m.run(SCENE)
    return m


@pytest.mark.parametrize("det", ["detectHarrisFeatures", "detectMinEigenFeatures", "detectFASTFeatures",
                                 "detectORBFeatures", "detectSIFTFeatures", "detectBRISKFeatures",
                                 "detectKAZEFeatures", "detectSURFFeatures"])
def test_detectors_find_points_and_sort_by_strength(scene, det):
    m = scene
    m.run(f"p = {det}(I);")
    n = int(m.num("p.Count").item())
    assert n > 5
    loc = m.num("p.Location")
    assert loc.shape == (n, 2) and loc.dtype == np.float32
    assert loc[:, 0].min() >= 1 and loc[:, 0].max() <= 240 and loc[:, 1].min() >= 1 and loc[:, 1].max() <= 200
    metric = m.num("p.Metric").ravel()
    assert np.all(np.diff(metric) <= 1e-6)
    assert m.num("size(p)").ravel().tolist() == [n, 1]


@pytest.mark.parametrize("det", ["detectHarrisFeatures", "detectSIFTFeatures", "detectORBFeatures",
                                 "detectKAZEFeatures", "detectSURFFeatures", "detectBRISKFeatures"])
def test_extract_and_match_recover_a_translation(scene, det):
    m = scene
    m.run(f"""
        [f1, v1] = extractFeatures(I, {det}(I));
        [f2, v2] = extractFeatures(J, {det}(J));
        pairs = matchFeatures(f1, f2);
        d = v2.Location(pairs(:,2),:) - v1.Location(pairs(:,1),:);
    """)
    assert m.num("size(pairs, 1)").item() >= 3
    assert m.str("class(pairs)") == "uint32"
    med = m.num("median(d, 1)").ravel()
    assert np.allclose(med, [7, 4], atol=0.6)


def test_corner_location_is_one_based(m):
    m.run("I = zeros(60); I(21:40, 21:40) = 1; I = imgaussfilt(I, 1); p = detectMinEigenFeatures(I);")
    loc = m.num("p.selectStrongest(4).Location").astype(float)
    # the four corners are placed symmetrically about the square's centre, pixel (30.5, 30.5)
    assert np.allclose(loc.mean(axis=0), [30.5, 30.5], atol=1e-3)
    assert np.all(np.abs(loc - 30.5).max(axis=1) < 10.5)


def test_roi_option_keeps_full_image_coordinates(scene):
    m = scene
    m.run("p = detectHarrisFeatures(I, 'ROI', [101 51 100 100]);")
    loc = m.num("p.Location")
    assert loc.shape[0] > 0
    assert loc[:, 0].min() >= 101 and loc[:, 0].max() <= 200
    assert loc[:, 1].min() >= 51 and loc[:, 1].max() <= 150


def test_point_object_indexing_and_methods(scene):
    m = scene
    m.run("s = detectSIFTFeatures(I); n = s.Count; a = s(2:4); b = s(end); c = [a; b];"
          "t = selectStrongest(s, 5); u = selectUniform(s, 6, size(I));")
    assert m.num("a.Count").item() == 3
    assert m.str("class(c)") == "SIFTPoints"
    assert m.num("c.Count").item() == 4
    assert np.array_equal(m.num("c.Location(4,:)"), m.num("s.Location(end,:)"))
    assert m.num("t.Count").item() == 5
    assert m.num("u.Count").item() == 6
    assert m.num("numel(s)").item() == m.num("n").item()
    assert m.num("isempty(s([]))").item() == 1
    m.run("s2 = s; s2.Metric = zeros(n, 1);")
    assert np.all(m.num("s2.Metric") == 0)
    out = m.run("a")
    assert "SIFTPoints" in out and "Location" in out and "Orientation" in out


def test_point_constructors(m):
    m.run("p = SURFPoints([10 20; 30 40], 'Scale', [2; 3], 'Metric', [5; 6]);"
          "c = cornerPoints([1 2; 3 4; 5 6]);")
    assert m.num("p.Scale").ravel().tolist() == [2, 3]
    assert m.num("p.SignOfLaplacian").dtype == np.int8
    assert m.num("c.Count").item() == 3
    assert m.num("c.Metric").ravel().tolist() == [0, 0, 0]
    with pytest.raises(MatlabError):
        m.run("p = SURFPoints([1 2; 3 4], 'Scale', [1 2 3]);")


def test_mser_regions(m):
    m.run("[x, y] = meshgrid(1:100); I = uint8(60 + 140*exp(-((x-31).^2 + (y-31).^2)/128) + "
          "100*exp(-((x-66).^2 + (y-71).^2)/72)); [r, cc] = detectMSERFeatures(I);")
    n = int(m.num("r.Count").item())
    assert n >= 2
    assert m.num("cc.NumObjects").item() == n
    locs = m.num("r.Location")
    assert np.min(np.hypot(locs[:, 0] - 31, locs[:, 1] - 31)) < 1
    assert np.min(np.hypot(locs[:, 0] - 66, locs[:, 1] - 71)) < 1
    assert m.str("class(r.PixelList)") == "cell"


def test_binary_features_and_hamming_matching(m):
    m.run("a = binaryFeatures(uint8([0 0; 255 255; 15 0]));"
          "b = binaryFeatures(uint8([255 255; 0 1; 15 0]));"
          "[pairs, mm] = matchFeatures(a, b, 'MatchThreshold', 20, 'MaxRatio', 1);")
    assert m.num("a.NumBits").item() == 16
    pairs = m.num("pairs")
    assert pairs.tolist() == [[1, 2], [2, 1], [3, 3]]
    assert np.allclose(m.num("mm").ravel(), [100 / 16, 0, 0])


def test_match_features_numeric_unique_and_ratio(m):
    m.run("rng(0); f1 = rand(20, 8); f2 = f1([5 3 1], :) + 1e-3;"
          "p = matchFeatures(f1, f2, 'Unique', true);"
          "q = matchFeatures(f2, f1);")
    assert sorted(map(tuple, m.num("p").tolist())) == [(1, 3), (3, 2), (5, 1)]
    assert m.num("q").tolist() == [[1, 5], [2, 3], [3, 1]]
    m.run("g1 = [1 0 0; 0.999 0.01 0; 0 1 0]; g2 = [1 0 0; 0 0 1];"
          "a = matchFeatures(g1, g2, 'MaxRatio', 1); b = matchFeatures(g1, g2, 'MaxRatio', 1, 'Unique', true);")
    assert m.num("a").tolist() == [[1, 1], [2, 1]]
    assert m.num("b").tolist() == [[1, 1]]


def test_block_descriptor(m):
    m.run("I = magic(20)/400; [f, v] = extractFeatures(I, [10 10; 1 1; 15 12], 'Method', 'Block', 'BlockSize', 3);")
    assert m.num("size(f)").ravel().tolist() == [2, 9]
    assert m.num("v").tolist() == [[10, 10], [15, 12]]
    assert np.allclose(m.num("f(1,:)"), m.num("reshape(I(9:11, 9:11), 1, [])"))


def test_show_matched_features_and_plot_draw(scene):
    m = scene
    m.run("figure; s = detectSURFFeatures(I); plot(s.selectStrongest(5));"
          "showMatchedFeatures(I, J, [10 10; 20 30], [17 14; 27 34], 'montage');"
          "ax = showMatchedFeatures(I, J, [10 10], [17 14]); close all")


def test_missing_opencv_gives_install_hint(m, monkeypatch):
    import sys

    from mylab.builtins import _cv
    monkeypatch.setattr(_cv, "_cv2", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(MatlabError, match="pip install"):
        m.run("detectFASTFeatures(zeros(20));")
