"""Robust 2-D transform fitting, two-view geometry, triangulation and 3-D transforms."""

import numpy as np
import pytest

from mylab.errors import MatlabError

TWO_VIEW = r"""
rng(7);
K = [700 0 321; 0 690 241; 0 0 1];
intr = cameraIntrinsics([700 690], [321 241], [480 640]);
X = [rand(80,1)*4-2, rand(80,1)*3-1.5, rand(80,1)*4+6];
R2 = rotz(8)*roty(-10)*rotx(3); t2 = [-1; 0.2; 0.1];
x1 = (K*X')'; x1 = x1(:,1:2)./x1(:,3);
Xc2 = (R2*X' + t2)'; x2 = (K*Xc2')'; x2 = x2(:,1:2)./x2(:,3);
"""


@pytest.mark.parametrize("kind,ctor,cls", [
    ("rigid", "rigidtform2d(30, [12 -5])", "rigidtform2d"),
    ("similarity", "simtform2d(1.3, 25, [40 -10])", "simtform2d"),
    ("affine", "affinetform2d([1.1 0.2 5; -0.1 0.9 7; 0 0 1])", "affinetform2d"),
    ("projective", "projtform2d([1 0.1 5; 0.05 0.9 -3; 1e-4 2e-4 1])", "projtform2d"),
])
def test_estgeotform2d_recovers_transform_despite_outliers(m, kind, ctor, cls):
    m.run(f"""
        rng(3); p = rand(60,2)*300; tf = {ctor};
        q = transformPointsForward(tf, p) + randn(60,2)*0.05;
        q(1:15,:) = rand(15,2)*300;
        [est, inl, status] = estgeotform2d(p, q, '{kind}');
    """)
    assert m.str("class(est)") == cls
    assert m.num("status").item() == 0
    inl = m.num("inl").ravel()
    assert inl.dtype == bool and inl.shape == (60,)
    assert not inl[:15].any() or inl[:15].sum() <= 1
    assert inl[15:].sum() >= 43
    m.run("[gx, gy] = meshgrid(0:50:300); g = [gx(:) gy(:)];"
          "e = max(vecnorm(transformPointsForward(est, g) - transformPointsForward(tf, g), 2, 2));")
    assert m.num("e").item() < 0.1


def test_estgeotform2d_status_and_errors(m):
    m.run("p = [0 0; 1 1]; [t, inl, st] = estgeotform2d(p, p, 'affine');")
    assert m.num("st").item() == 1
    with pytest.raises(MatlabError, match="matched point pairs"):
        m.run("t = estgeotform2d(p, p, 'affine');")
    with pytest.raises(MatlabError, match="transformation type"):
        m.run("t = estgeotform2d(p, p, 'shear');")


def test_estimated_transform_works_with_imwarp(m):
    m.run("""
        I = zeros(80); I(30:50, 20:40) = 1;
        tf = estgeotform2d([0 0; 10 0; 0 10], [5 3; 15 3; 5 13], 'similarity');
        J = imwarp(I, tf, 'OutputView', imref2d(size(I)));
    """)
    J = m.num("J")
    ys, xs = np.nonzero(J > 0.5)
    assert abs(xs.mean() + 1 - 35) < 0.6 and abs(ys.mean() + 1 - 43) < 0.6


def test_older_estimate_functions(m):
    m.run("""
        rng(1); p = rand(30,2)*100; q = p*[0.9 0.1; -0.1 0.9] + [3 4];
        [t1, inl] = estimateGeometricTransform2D(p, q, 'affine');
        [t2, a, b, st] = estimateGeometricTransform(p, q, 'similarity');
    """)
    assert m.str("class(t1)") == "affine2d"
    assert np.allclose(m.num("t1.T"), [[0.9, 0.1, 0], [-0.1, 0.9, 0], [3, 4, 1]], atol=1e-6)
    assert m.num("size(a)").ravel().tolist() == [30, 2]


def test_fundamental_matrix_satisfies_epipolar_constraint(m):
    m.run(TWO_VIEW)
    for method in ("Norm8Point", "RANSAC", "MSAC", "LMedS", "LTS"):
        m.run(f"[F, inl, st] = estimateFundamentalMatrix(x1, x2, 'Method', '{method}');"
              "h1 = [x1 ones(80,1)]; h2 = [x2 ones(80,1)]; r = abs(sum((h2*F).*h1, 2));")
        assert m.num("st").item() == 0
        assert m.num("max(r)").item() < 1e-6, method
        assert abs(np.linalg.norm(m.num("F")) - 1) < 1e-9
        assert abs(np.linalg.det(m.num("F"))) < 1e-12


def test_fundamental_rejects_outliers(m):
    m.run(TWO_VIEW + "x2o = x2; x2o(1:10,:) = x2o(1:10,:) + 25;"
          "[F, inl] = estimateFundamentalMatrix(x1, x2o, 'Method', 'MSAC', 'NumTrials', 2000);")
    inl = m.num("inl").ravel()
    assert not inl[:10].any() and inl[10:].all()


def test_essential_matrix_and_relative_pose(m):
    pytest.importorskip("cv2")
    m.run(TWO_VIEW + """
        [E, inlE, st] = estimateEssentialMatrix(x1, x2, intr);
        [pose, frac] = estrelpose(E, intr, x1, x2);
        [o, l, f2] = relativeCameraPose(E, intr, x1, x2);
    """)
    pytest.importorskip("cv2")
    assert m.num("st").item() == 0 and m.num("nnz(inlE)").item() >= 75
    assert m.str("class(pose)") == "rigidtform3d"
    R2 = m.num("R2")
    t2 = m.num("t2").ravel()
    assert np.allclose(m.num("pose.R"), R2.T, atol=1e-6)
    expected_t = -R2.T @ t2 / np.linalg.norm(t2)
    assert np.allclose(m.num("pose.Translation").ravel(), expected_t, atol=1e-6)
    assert m.num("frac").item() == 1
    assert np.allclose(m.num("o"), R2, atol=1e-6)
    assert np.allclose(m.num("l").ravel(), expected_t, atol=1e-6)


def test_relative_pose_from_fundamental_matrix(m):
    m.run(TWO_VIEW + "F = estimateFundamentalMatrix(x1, x2, 'Method', 'Norm8Point');"
          "pose = estrelpose(F, intr, x1, x2);")
    assert np.allclose(m.num("pose.R"), m.num("R2").T, atol=1e-5)


def test_triangulate_recovers_points(m):
    m.run(TWO_VIEW + """
        P1 = cameraProjection(intr, rigidtform3d); P2 = cameraProjection(intr, rigidtform3d(R2, t2));
        [W, err, valid] = triangulate(x1, x2, P1, P2);
        C1 = cameraMatrix(intr, eye(3), [0 0 0]); C2 = cameraMatrix(intr, R2', t2');
        W2 = triangulate(x1, x2, C1, C2);
    """)
    X = m.num("X")
    assert np.allclose(m.num("W"), X, atol=1e-8)
    assert np.allclose(m.num("W2"), X, atol=1e-8)
    assert m.num("max(err)").item() < 1e-8
    assert m.num("all(valid)").item()


def test_epipolar_line(m):
    m.run(TWO_VIEW + "F = estimateFundamentalMatrix(x1, x2, 'Method', 'Norm8Point'); L = epipolarLine(F, x1);"
          "d = abs(sum(L .* [x2 ones(80,1)], 2)) ./ hypot(L(:,1), L(:,2));")
    assert m.num("max(d)").item() < 1e-6


def test_rigidtform3d_and_rotation_vectors(m):
    m.run("""
        T = rigidtform3d([10 20 30], [1 2 3]); P = [1 0 0; 0 1 0];
        Q = transformPointsForward(T, P); P2 = transformPointsInverse(T, Q);
        Ti = invert(T); v = rotmat2vec3d(T.R); R = rotvec2mat3d(v);
        r3 = rigid3d(T.R', [1 2 3]); Q3 = transformPointsForward(r3, P);
    """)
    assert np.allclose(m.num("P2"), [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    assert np.allclose(m.num("R"), m.num("T.R"), atol=1e-12)
    assert np.allclose(m.num("Q3"), m.num("Q"), atol=1e-12)
    assert np.allclose(m.num("Ti.A") @ m.num("T.A"), np.eye(4), atol=1e-12)
    Rz = m.num("T.R")
    assert np.allclose(Rz @ Rz.T, np.eye(3), atol=1e-12)
    with pytest.raises(MatlabError, match="rotation"):
        m.run("rigidtform3d([1 0 0; 0 2 0; 0 0 1], [0 0 0])")



def test_insert_shape_rectangle_edges_are_one_based(m):
    pytest.importorskip("cv2")
    m.run("I = zeros(50, 60, 'uint8'); J = insertShape(I, 'rectangle', [11 21 10 5], 'Color', 'red', "
          "'SmoothEdges', false);")
    J = m.num("J")
    assert J.shape == (50, 60, 3) and J.dtype == np.uint8
    red = J[:, :, 0] == 255
    ys, xs = np.nonzero(red)
    assert xs.min() + 1 == 11 and xs.max() + 1 == 20 and ys.min() + 1 == 21 and ys.max() + 1 == 25
    assert not J[:, :, 1].any()


def test_insert_shape_filled_and_opacity(m):
    pytest.importorskip("cv2")
    m.run("I = zeros(40, 'double'); J = insertShape(I, 'filled-circle', [20 20 6], 'Color', [0 0 1], "
          "'Opacity', 0.5, 'SmoothEdges', false);")
    J = m.num("J")
    assert J.dtype == np.float64
    assert abs(J[19, 19, 2] - 0.5) < 0.01 and J[0, 0, 2] == 0
    m.run("K = insertShape(uint8(zeros(30)), 'line', [1 15 30 15], 'Color', 'white', 'SmoothEdges', false);")
    assert np.all(m.num("K")[14, :, 0] == 255)
    m.run("P = insertShape(uint8(zeros(30)), 'polygon', {[5 5 25 5 15 25]}, 'Color', 'green');")
    assert m.num("P")[:, :, 1].any()


def test_insert_marker_text_and_annotation(m):
    pytest.importorskip("cv2")
    m.run("I = zeros(60, 80, 'uint8'); M = insertMarker(I, [10 10; 30 40], 'x', 'Size', 3);"
          "T = insertText(I, [5 5], 'mylab', 'FontSize', 14);"
          "A = insertObjectAnnotation(I, 'rectangle', [20 30 30 20; 5 5 10 10], {'car', 'dog'});"
          "C = insertObjectAnnotation(I, 'circle', [40 30 10], 0.95);")
    M = m.num("M")
    assert M[9, 9, 1] == 255 and M[9, 9, 0] == 0
    assert M[39, 29, 1] == 255
    assert m.num("T").any() and m.num("A").any() and m.num("C").any()
    with pytest.raises(MatlabError, match="colour"):
        m.run("insertShape(I, 'circle', [10 10 3], 'Color', 'chartreuse')")
