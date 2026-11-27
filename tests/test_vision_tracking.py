"""Point tracking, dense optical flow and Kalman filters."""

import numpy as np
import pytest

from mylab.errors import MatlabError

TEXTURE = "rng(2); I = mat2gray(imgaussfilt(rand(120, 160), 2));"


def test_point_tracker_follows_a_translation(m):
    pytest.importorskip("cv2")
    m.run(TEXTURE + """
        J = imtranslate(I, [2.5 -1.5]);
        pts = [40 40; 80 60; 120 30; 60 90; 100 80];
        tr = vision.PointTracker('MaxBidirectionalError', 1);
        initialize(tr, pts, I);
        [p2, v, sc] = tr(J);
        [p3, v3] = step(tr, J);
    """)
    v = m.num("v").ravel()
    assert v.all()
    assert np.allclose(m.num("p2") - m.num("pts"), [2.5, -1.5], atol=0.1)
    sc = m.num("sc").ravel()
    assert np.all((sc >= 0) & (sc <= 1))
    assert np.allclose(m.num("p3"), m.num("p2"), atol=0.1)  # no motion between the last two frames
    m.run("setPoints(tr, [10 10]); [p4, v4] = tr(J); release(tr);")
    assert m.num("size(p4)").ravel().tolist() == [1, 2]
    with pytest.raises(MatlabError, match="initialize"):
        m.run("tr(J)")


def test_point_tracker_drops_occluded_points(m):
    pytest.importorskip("cv2")
    m.run(TEXTURE + "J = imtranslate(I, [1 0]); J(30:90, 10:70) = 0.5;"
          "tr = vision.PointTracker('MaxBidirectionalError', 1);"
          "initialize(tr, [40 60; 120 60], I); [p, v] = tr(J);")
    v = m.num("v").ravel()
    assert not v[0] and v[1]
    assert np.allclose(m.num("p(2,:)"), [121, 60], atol=0.1)


@pytest.mark.parametrize("ctor,shift,tol", [("opticalFlowLK", (0.5, 0.3), 0.05),
                                            ("opticalFlowHS", (0.5, 0.3), 0.08),
                                            ("opticalFlowFarneback", (2.5, -1.5), 0.05)])
def test_dense_optical_flow(m, ctor, shift, tol):
    if ctor == "opticalFlowFarneback":
        pytest.importorskip("cv2")
    m.run(TEXTURE + f"""
        of = {ctor};
        f0 = estimateFlow(of, I);
        f = estimateFlow(of, imtranslate(I, [{shift[0]} {shift[1]}]));
        vx = f.Vx(30:90, 40:120); vy = f.Vy(30:90, 40:120);
    """)
    assert np.all(m.num("f0.Vx") == 0)
    assert abs(np.median(m.num("vx")) - shift[0]) < tol
    assert abs(np.median(m.num("vy")) - shift[1]) < tol
    assert np.allclose(m.num("f.Magnitude"), np.hypot(m.num("f.Vx"), m.num("f.Vy")))
    m.run("reset(of); g = estimateFlow(of, I);")
    assert np.all(m.num("g.Vx") == 0)


def test_optical_flow_object_and_plot(m):
    m.run("f = opticalFlow(ones(10, 12), -ones(10, 12)); figure; plot(f, 'DecimationFactor', [2 2]); close all")
    assert np.allclose(m.num("f.Orientation"), -np.pi / 4)


def test_configure_kalman_filter_tracks_constant_velocity(m):
    m.run("""
        rng(4);
        kf = configureKalmanFilter('ConstantVelocity', [10 20], [1 1]*1e5, [25 10], 25);
        for k = 1:30
            predict(kf);
            correct(kf, [10 + 2*k, 20 - k] + randn(1, 2)*0.5);
        end
        x = kf.State; [zp, xp] = predict(kf);
    """)
    x = m.num("x").ravel()
    assert m.num("size(kf.StateTransitionModel)").ravel().tolist() == [4, 4]
    assert abs(x[0] - 70) < 2 and abs(x[1] - 2) < 0.3 and abs(x[2] + 10) < 2 and abs(x[3] + 1) < 0.3
    assert np.allclose(m.num("zp").ravel(), [x[0] + x[1], x[2] + x[3]])


def test_kalman_filter_matches_textbook_update(m):
    m.run("""
        kf = vision.KalmanFilter([1 1; 0 1], [1 0], 'State', [0; 1], 'StateCovariance', eye(2), ...
                                 'ProcessNoise', 0.01, 'MeasurementNoise', 0.5);
        [zp, xp, Pp] = predict(kf);
        [zc, xc, Pc] = correct(kf, 1.2);
        d = distance(kf, [1; 2; 3]);
        k2 = clone(kf); predict(k2);
    """)
    A = np.array([[1.0, 1], [0, 1]])
    Hm = np.array([[1.0, 0]])
    P = A @ np.eye(2) @ A.T + 0.01 * np.eye(2)
    x = A @ np.array([0.0, 1])
    assert np.allclose(m.num("xp").ravel(), x)
    assert np.allclose(m.num("Pp"), P)
    S = Hm @ P @ Hm.T + 0.5
    Kg = P @ Hm.T / S
    xc = x + (Kg * (1.2 - x[0])).ravel()
    assert np.allclose(m.num("xc").ravel(), xc)
    assert m.num("size(d)").ravel().tolist() == [1, 3]
    assert np.allclose(m.num("kf.State").ravel(), xc)  # the clone is independent
    with pytest.raises(MatlabError, match="MotionModel"):
        m.run("configureKalmanFilter('Spiral', [0 0], [1 1], [1 1], 1)")
