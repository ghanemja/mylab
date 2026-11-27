"""pointCloud objects, PLY/PCD files and point cloud processing."""

import numpy as np
import pytest

from mylab.errors import MatlabError

PLANE = r"""
rng(5);
[X, Y] = meshgrid(linspace(-1, 1, 30)); Z = 0.2*X - 0.1*Y + 0.5;
P = [X(:) Y(:) Z(:)]; P = [P; rand(40, 3)*2 - 1];
pc = pointCloud(P, 'Color', uint8(repmat([255 0 0], size(P, 1), 1)));
"""

# a curved, non-symmetric surface so that ICP has a unique answer
SURFACE = r"""
[u, v] = meshgrid(linspace(-1, 1, 40), linspace(-0.6, 0.8, 30));
w = 0.3*u.^2 - 0.2*v.^3 + 0.1*u.*v;
S = pointCloud([u(:) v(:) w(:)]);
"""


def test_point_cloud_properties(m):
    m.run(PLANE)
    assert m.num("pc.Count").item() == 940
    assert np.allclose(m.num("pc.XLimits"), [-1, 1])
    assert m.str("class(pc.Color)") == "uint8"
    out = m.run("pc")
    assert "pointCloud with properties" in out and "Location" in out
    m.run("q = pointCloud(rand(4, 5, 3)); n = q.Count; L = q.Location;")
    assert m.num("n").item() == 20
    assert m.num("size(L)").ravel().tolist() == [4, 5, 3]


def test_select_remove_invalid_and_neighbors(m):
    m.run("P = [0 0 0; 1 0 0; NaN 0 0; 0 2 0; 5 5 5]; pc = pointCloud(P);"
          "[c, idx] = removeInvalidPoints(pc); s = select(pc, [1 2]);"
          "[k, d] = findNearestNeighbors(pc, [0.9 0 0], 2);"
          "[r, rd] = findNeighborsInRadius(pc, [0 0 0], 1.5);"
          "roi = findPointsInROI(pc, [-1 1.5 -1 1 -1 1]);")
    assert m.num("c.Count").item() == 4
    assert m.num("idx").ravel().tolist() == [1, 2, 4, 5]
    assert m.num("s.Count").item() == 2
    assert m.num("k").ravel().tolist() == [2, 1]
    assert np.allclose(m.num("d").ravel(), [0.1, 0.9])
    assert sorted(m.num("r").ravel().tolist()) == [1, 2]
    assert m.num("roi").ravel().tolist() == [1, 2]


@pytest.mark.parametrize("ext,enc", [("ply", "ascii"), ("ply", "binary"), ("pcd", "ascii"), ("pcd", "binary")])
def test_file_round_trip(m, tmp_path, ext, enc):
    f = tmp_path / f"cloud.{ext}"
    m.interp.set_var("f", np.array([list(str(f))]))
    m.run(PLANE + f"pc.Normal = pcnormals(pc); pc.Intensity = (1:pc.Count)'; pcwrite(pc, f, 'Encoding', '{enc}');"
          "q = pcread(f);")
    assert np.allclose(m.num("q.Location"), m.num("pc.Location"), atol=1e-6)
    assert np.array_equal(m.num("q.Color"), m.num("pc.Color"))
    assert np.allclose(m.num("q.Normal"), m.num("pc.Normal"), atol=1e-6)
    assert np.allclose(m.num("q.Intensity"), m.num("pc.Intensity"))


def test_read_hand_written_ply_with_faces(m, tmp_path):
    f = tmp_path / "tri.ply"
    f.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                 "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    m.interp.set_var("f", np.array([list(str(f))]))
    m.run("q = pcread(f);")
    assert m.num("q.Location").tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    g = tmp_path / "cloud.xyz"
    g.write_text("0 0 0\n")
    m.interp.set_var("g", np.array([list(str(g))]))
    with pytest.raises(MatlabError, match="PLY and PCD"):
        m.run("pcread(g)")


def test_fit_plane(m):
    m.run(PLANE + "[mdl, inl, outl, err] = pcfitplane(pc, 0.01); p = mdl.Parameters;")
    p = m.num("p").ravel()
    p = p / -p[2]
    assert np.allclose(p, [0.2, -0.1, -1, 0.5], atol=1e-6)
    inl = m.num("inl").ravel()
    assert np.all(np.isin(np.arange(1, 901), inl))
    assert m.num("err").item() < 0.005
    m.run("[m2, i2] = pcfitplane(pc, 0.01, [0 0 1], 20); n = m2.Normal;")
    assert abs(abs(m.num("n").ravel() @ [0, 0, 1]) - 1 / np.sqrt(1.05)) < 1e-6


def test_downsample_denoise_normals(m):
    m.run(PLANE + """
        g = pcdownsample(pc, 'gridAverage', 0.2);
        r = pcdownsample(pc, 'random', 0.25);
        u = pcdownsample(pc, 'nonuniformGridSample', 10);
        [d, inl, outl] = pcdenoise(pc);
        n = pcnormals(pc);
    """)
    assert 50 < m.num("g.Count").item() < 400
    assert m.num("r.Count").item() == 235
    assert 0 < m.num("u.Count").item() < 200
    assert m.num("d.Count").item() < 940 and m.num("numel(outl)").item() > 0
    n = m.num("n")[:900]
    truth = np.array([0.2, -0.1, -1]) / np.linalg.norm([0.2, -0.1, -1])
    assert np.mean(np.abs(np.abs(n @ truth) - 1) < 1e-4) > 0.95


def test_icp_recovers_a_rigid_motion(m):
    m.run(SURFACE + """
        tf = rigidtform3d([4 -6 8], [0.05 0.1 -0.05]);
        M = pctransform(S, tf);
        [est, reg, rmse] = pcregistericp(M, S, 'MaxIterations', 60, 'Tolerance', [1e-6 1e-4]);
        [est2, ~, rmse2] = pcregistericp(M, S, 'Metric', 'pointToPlane', 'MaxIterations', 30);
        A = est.A * tf.A; B = est2.A * tf.A;
    """)
    assert m.str("class(est)") == "rigidtform3d"
    assert np.allclose(m.num("A"), np.eye(4), atol=5e-3)
    assert np.allclose(m.num("B"), np.eye(4), atol=5e-3)
    assert m.num("rmse").item() < 0.01


def test_transform_merge_and_segment(m):
    m.run("a = pointCloud([0 0 0; 1 0 0]); b = pointCloud([0 0 0.001; 5 5 5]);"
          "t = pctransform(a, rigidtform3d(eye(3), [1 2 3])); L = t.Location;"
          "mg = pcmerge(a, b, 0.1); [lab, n] = pcsegdist(pointCloud([0 0 0; 0.1 0 0; 3 3 3]), 0.5);"
          "d = pctransform(a, [1 1 1; 2 2 2]);")
    assert m.num("L").tolist() == [[1, 2, 3], [2, 2, 3]]
    assert m.num("mg.Count").item() == 3
    assert m.num("n").item() == 2
    assert m.num("lab").ravel().tolist() == [1, 1, 2]
    assert m.num("d.Location").tolist() == [[1, 1, 1], [3, 2, 2]]


def test_pcshow_draws(m):
    m.run(PLANE + "figure; pcshow(pc); pcshow(P); pcshowpair(pc, pc); close all")
