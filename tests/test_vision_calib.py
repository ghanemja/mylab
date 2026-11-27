"""Camera models, checkerboard calibration (rendered at known poses), undistortion, projection and stereo."""

import math

import numpy as np
import pytest

from mylab.errors import MatlabError

cv2 = pytest.importorskip("cv2")

H, W = 360, 480
K0 = np.array([[500.0, 0, 240], [0, 505.0, 180], [0, 0, 1]])  # 0-based principal point
ROWS, COLS, SQ = 7, 10, 20.0  # squares, square size


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render_board(K, R, t, ss=3):
    """A checkerboard on the plane Z = 0 seen by a pinhole camera; inner corner (0, 0) at the world origin."""
    ys, xs = np.mgrid[0:H * ss, 0:W * ss]
    u = (xs + 0.5) / ss - 0.5
    v = (ys + 0.5) / ss - 0.5
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])
    w = np.linalg.inv(Hm) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    X, Y = w[0] / w[2], w[1] / w[2]
    i = np.floor(X / SQ).astype(int) + 1
    j = np.floor(Y / SQ).astype(int) + 1
    inside = (i >= 0) & (i < COLS) & (j >= 0) & (j < ROWS)
    val = np.where(inside, ((i + j) % 2 == 0) * 1.0, 1.0)
    img = val.reshape(H * ss, W * ss).reshape(H, ss, W, ss).mean(axis=(1, 3))
    return (img * 200 + 30).astype(np.uint8)


POSES = [(_rot("x", 10), [-90, -60, 520]), (_rot("y", 20), [-80, -65, 560]),
         (_rot("x", -15) @ _rot("y", -10), [-100, -50, 500]), (_rot("y", -25) @ _rot("x", 5), [-75, -60, 600]),
         (_rot("x", 20) @ _rot("y", 15), [-90, -70, 540]), (_rot("y", 10) @ _rot("x", -20), [-95, -55, 580])]


@pytest.fixture(scope="module")
def board_images():
    return np.stack([render_board(K0, R, np.array(t, float)) for R, t in POSES], axis=2)


def test_generate_checkerboard_points_order(m):
    wp = m.num("generateCheckerboardPoints([4 5], 10)")
    assert wp.shape == (12, 2)
    assert wp[:4].tolist() == [[0, 0], [0, 10], [0, 20], [10, 0]]
    assert wp[-1].tolist() == [30, 20]


def test_detect_checkerboard_points_matches_projection(m, board_images):
    m.interp.set_var("I", board_images[:, :, 0].copy())
    m.run("[ip, bs] = detectCheckerboardPoints(I); wp = generateCheckerboardPoints(bs, 20);")
    assert m.num("bs").ravel().tolist() == [ROWS, COLS]
    ip = m.num("ip")
    wp = m.num("wp")
    R, t = POSES[0][0], np.array(POSES[0][1], float)
    C = np.column_stack([wp, np.zeros(len(wp))]) @ R.T + t
    proj = (C @ K0.T)[:, :2] / C[:, 2:3] + 1.0  # MATLAB pixel coordinates
    err = np.abs(ip - proj)
    assert err.max() < 0.3 and err.mean() < 0.1


def test_calibration_recovers_intrinsics(m, board_images):
    m.interp.set_var("I", board_images.reshape(H, W, 1, -1).copy())
    m.run("""
        [ip, bs, used] = detectCheckerboardPoints(I);
        wp = generateCheckerboardPoints(bs, 20);
        [params, used2, errs] = estimateCameraParameters(ip, wp, 'ImageSize', [360 480]);
        K = params.K;
    """)
    assert m.num("all(used)").item()
    K = m.num("K")
    assert abs(K[0, 0] / 500 - 1) < 0.015 and abs(K[1, 1] / 505 - 1) < 0.015
    assert abs(K[0, 2] - 241) < 3 and abs(K[1, 2] - 181) < 3
    assert m.num("params.MeanReprojectionError").item() < 0.2
    assert m.num("params.NumPatterns").item() == len(POSES)
    assert m.num("size(params.ReprojectionErrors)").ravel().tolist() == [54, 2, len(POSES)]
    assert np.allclose(m.num("params.IntrinsicMatrix"), K.T)
    # extrinsics of the first view
    R = m.num("params.RotationMatrices(:,:,1)")
    t = m.num("params.TranslationVectors(1,:)").ravel()
    assert np.allclose(R.T, POSES[0][0], atol=0.02)
    assert np.allclose(t, POSES[0][1], rtol=0.03, atol=3)
    m.run("p = worldToImage(params, params.RotationMatrices(:,:,1), params.TranslationVectors(1,:), wp, "
          "'ApplyDistortion', true); e = max(vecnorm(p - ip(:,:,1), 2, 2));")
    assert m.num("e").item() < 0.5
    m.run("tf = params.PatternExtrinsics{2}; q = world2img(wp, tf, params.Intrinsics, 'ApplyDistortion', true);"
          "e2 = max(vecnorm(q - ip(:,:,2), 2, 2)); w = img2world2d(ip(:,:,2), tf, params.Intrinsics);"
          "e3 = max(vecnorm(w - wp, 2, 2));")
    assert m.num("e2").item() < 0.5
    assert m.num("e3").item() < 0.5
    m.run("ext = estimateExtrinsics(ip(:,:,3), wp, params.Intrinsics);")
    assert np.allclose(m.num("ext.R"), POSES[2][0], atol=0.02)


def test_camera_intrinsics_object(m):
    m.run("intr = cameraIntrinsics([800 810], [320 240], [480 640], 'RadialDistortion', [-0.2 0.05]);")
    assert np.allclose(m.num("intr.K"), [[800, 0, 320], [0, 810, 240], [0, 0, 1]])
    out = m.run("intr")
    assert "cameraIntrinsics" in out and "RadialDistortion" in out
    with pytest.raises(MatlabError):
        m.run("intr.FocalLength = [1 2];")


def _distort(p, K, k1, k2):
    x = (p[:, 0] - K[0, 2]) / K[0, 0]
    y = (p[:, 1] - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    return np.column_stack([x * f * K[0, 0] + K[0, 2], y * f * K[1, 1] + K[1, 2]])


def test_undistort_points_inverts_the_model(m):
    K = np.array([[600.0, 0, 321], [0, 600.0, 241], [0, 0, 1]])
    ideal = np.array([[50.0, 60], [600, 400], [321, 241], [100, 450]])
    m.interp.set_var("pd", _distort(ideal, K, -0.25, 0.08))
    m.run("intr = cameraIntrinsics([600 600], [321 241], [480 640], 'RadialDistortion', [-0.25 0.08]);"
          "pu = undistortPoints(pd, intr);")
    assert np.allclose(m.num("pu"), ideal, atol=1e-6)


def test_undistort_image_moves_a_dot_to_its_ideal_place(m):
    K = np.array([[300.0, 0, 161], [0, 300.0, 121], [0, 0, 1]])
    ideal = np.array([[40.0, 30.0]])
    pd = _distort(ideal, K, -0.3, 0.0)[0]
    img = np.zeros((240, 320))
    yy, xx = np.mgrid[1:241, 1:321]
    img += np.exp(-((xx - pd[0]) ** 2 + (yy - pd[1]) ** 2) / (2 * 1.5 ** 2))
    m.interp.set_var("I", img)
    m.run("intr = cameraIntrinsics([300 300], [161 121], [240 320], 'RadialDistortion', [-0.3 0]);"
          "J = undistortImage(I, intr); [Jf, origin] = undistortImage(I, intr, 'OutputView', 'full');")
    J = m.num("J")
    w = J * (J > 0.3 * J.max())
    cy = (w * yy).sum() / w.sum()
    cx = (w * xx).sum() / w.sum()
    assert abs(cx - ideal[0, 0]) < 0.3 and abs(cy - ideal[0, 1]) < 0.3
    Jf = m.num("Jf")
    assert Jf.shape[0] >= 240 and Jf.shape[1] >= 320
    origin = m.num("origin").ravel()
    assert origin[0] <= 1 and origin[1] <= 1


def test_stereo_calibration_rectification_and_reconstruction(m, board_images):
    base = 60.0
    right = np.stack([render_board(K0, R, np.array(t, float) + [-base, 0, 0]) for R, t in POSES], axis=2)
    m.interp.set_var("I1", board_images.reshape(H, W, 1, -1).copy())
    m.interp.set_var("I2", right.reshape(H, W, 1, -1).copy())
    m.run("""
        [ip, bs, used] = detectCheckerboardPoints(I1, I2);
        wp = generateCheckerboardPoints(bs, 20);
        sp = estimateCameraParameters(ip, wp, 'ImageSize', [360 480]);
        T = sp.TranslationOfCamera2; R = sp.RotationOfCamera2;
    """)
    assert m.str("class(sp)") == "stereoParameters"
    # The baseline's length is well determined; its direction less so with six synthetic views (OpenCV 4 and 5
    # differ by a few degrees there), so check the length tightly and the direction loosely.
    T = m.num("T").ravel()
    assert abs(np.linalg.norm(T) - base) < 0.01 * base
    assert T[0] < 0 and np.degrees(np.arccos(-T[0] / np.linalg.norm(T))) < 5.0
    assert np.allclose(m.num("R"), np.eye(3), atol=0.01)
    m.run("""
        [J1, J2, Q] = rectifyStereoImages(I1(:,:,1,1), I2(:,:,1,1), sp, 'OutputView', 'full');
        [c1, b1] = detectCheckerboardPoints(J1); [c2, b2] = detectCheckerboardPoints(J2);
        dy = max(abs(c1(:,2) - c2(:,2)));
        d = c1(:,1) - c2(:,1);
        h = [c1 d ones(size(d))] * Q; X = h(:,1:3) ./ h(:,4);
        Wt = [wp zeros(size(wp,1),1)];
    """)
    assert m.num("dy").item() < 0.5
    X = m.num("X")
    R0, t0 = POSES[0][0], np.array(POSES[0][1])
    Wt = m.num("Wt")
    truth = Wt @ R0.T + t0
    assert np.allclose(np.linalg.norm(X - X.mean(0), axis=1), np.linalg.norm(truth - truth.mean(0), axis=1),
                       atol=2.0)
    assert abs(X[:, 2].mean() - truth[:, 2].mean()) < 0.03 * truth[:, 2].mean()


def test_disparity_of_a_shifted_texture(m):
    m.run("""
        rng(1); T = im2uint8(mat2gray(imgaussfilt(rand(200, 300), 1.2)));
        L = T(:, 31:270); R = T(:, 31+9:270+9);
        D1 = disparitySGM(L, R, 'DisparityRange', [0 48]);
        D2 = disparityBM(L, R, 'DisparityRange', [0 32]);
        c1 = D1(30:170, 60:200); c2 = D2(30:170, 60:200);
    """)
    for name in ("c1", "c2"):
        c = m.num(name)
        assert c.dtype == np.float32
        assert abs(np.nanmedian(c) - 9) < 0.25
    assert np.isnan(m.num("D1")).any()


def test_reconstruct_scene_with_known_geometry(m):
    m.run("""
        rng(1); T = im2uint8(mat2gray(imgaussfilt(rand(240, 360), 1.2)));
        L = T(:, 41:320); R = T(:, 53:332);
        cp = cameraParameters('K', [500 0 140.5; 0 500 120.5; 0 0 1], 'ImageSize', [240 280]);
        sp = stereoParameters(cp, cp, eye(3), [-60 0 0]);
        [J1, J2, Q] = rectifyStereoImages(L, R, sp, 'OutputView', 'full');
        D = disparitySGM(J1, J2, 'DisparityRange', [0 64]);
        xyz = reconstructScene(D, Q); z = xyz(80:160, 100:180, 3);
        W = triangulate([140.5 120.5], [128.5 120.5], sp);
    """)
    assert abs(np.nanmedian(m.num("z")) - 2500) < 30
    assert np.allclose(m.num("W"), [0, 0, 2500], atol=1e-6)


def test_checkerboard_not_found(m):
    m.run("[ip, bs, used] = detectCheckerboardPoints(zeros(100, 120, 'uint8'));")
    assert m.num("isempty(ip)").item() and not m.num("used").any()
