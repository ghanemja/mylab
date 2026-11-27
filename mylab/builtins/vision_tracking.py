"""Tracking and motion: vision.PointTracker (pyramidal KLT through OpenCV), dense optical flow objects
(Lucas-Kanade and Horn-Schunck in NumPy, Farneback through OpenCV) and Kalman filters.

Optical flow is reported in pixels per frame, Vx to the right and Vy downwards. Point locations are
1-based [x y] like everywhere else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ._cv import (
    VisionObject, col, cv2mod, fmat, fscalar, fvec, gray_float, gray_u8, method, options, scalar,
)
from ._registry import builtin
from ._util import is_text, text
from .vision_detect import system_object

# -- point tracker ----------------------------------------------------------------------------------------


class PointTracker(VisionObject):
    CLASS = "vision.PointTracker"
    PROPS = ("NumPyramidLevels", "MaxBidirectionalError", "BlockSize", "MaxIterations")
    SETTABLE = frozenset(PROPS)
    is_handle = True

    def __init__(self):
        super().__init__(NumPyramidLevels=scalar(3), MaxBidirectionalError=scalar(np.inf),
                         BlockSize=np.array([[31.0, 31.0]]), MaxIterations=scalar(30))
        self.pts = None      # N-by-2, 0-based
        self.valid = None
        self.prev = None

    def check_set(self, name: str, value: Any) -> Any:
        if self.pts is not None and name in self.PROPS:
            raise MatlabError(f"vision.PointTracker: {name} cannot be changed after initialize; call release first.",
                              "mylab:vision:lockedProperty")
        v = np.asarray(value, dtype=float).reshape(1, -1)
        if name == "BlockSize" and (v.size != 2 or np.any(v < 5) or np.any(v % 2 == 0)):
            raise MatlabError("vision.PointTracker: BlockSize must be [height width] with odd values of at least 5.")
        return v

    def _lk(self, a: np.ndarray, b: np.ndarray, p: np.ndarray):
        cv2 = cv2mod(self.CLASS)
        bs = self.p["BlockSize"].ravel()
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(self.p["MaxIterations"].flat[0]), 0.01)
        levels = int(self.p["NumPyramidLevels"].flat[0]) - 1
        nxt, st, err = cv2.calcOpticalFlowPyrLK(a, b, p.astype(np.float32).reshape(-1, 1, 2), None,
                                                winSize=(int(bs[1]), int(bs[0])), maxLevel=max(levels, 0),
                                                criteria=crit)
        return nxt.reshape(-1, 2).astype(float), st.ravel().astype(bool), err.ravel().astype(float)


@method(PointTracker, "initialize")
def _pt_initialize(self, interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("initialize: expected initialize(tracker, points, I).", "MATLAB:minrhs")
    p = fmat(args[0], "initialize", "points", 2)
    self.prev = gray_u8(args[1], "initialize")
    self.pts = p - 1.0
    self.valid = np.ones(len(p), dtype=bool)
    return None


@method(PointTracker, "step")
def _pt_step(self, interp, args, nargout):
    if self.pts is None:
        raise MatlabError("vision.PointTracker: call initialize(tracker, points, I) before tracking.",
                          "mylab:vision:notInitialized")
    if len(args) != 1:
        raise MatlabError("vision.PointTracker: call tracker(I) with the next frame.")
    cur = gray_u8(args[0], "step")
    if cur.shape != self.prev.shape:
        raise MatlabError("vision.PointTracker: every frame must have the size of the first one.")
    n = len(self.pts)
    out = self.pts.copy()
    valid = self.valid.copy()
    scores = np.zeros(n)
    idx = np.nonzero(valid)[0]
    if idx.size:
        nxt, st, err = self._lk(self.prev, cur, self.pts[idx])
        H, W = cur.shape
        inside = (nxt[:, 0] >= 0) & (nxt[:, 1] >= 0) & (nxt[:, 0] <= W - 1) & (nxt[:, 1] <= H - 1)
        ok = st & inside
        mbe = float(self.p["MaxBidirectionalError"].flat[0])
        if np.isfinite(mbe):
            back, st2, _ = self._lk(cur, self.prev, nxt)
            fb = np.sqrt(((back - self.pts[idx]) ** 2).sum(1))
            ok &= st2 & (fb <= mbe)
        out[idx] = nxt
        valid[idx] = ok
        # a confidence in [0, 1] from OpenCV's mean absolute patch difference (0..255 per pixel)
        scores[idx] = np.where(ok, np.clip(1.0 - err / 255.0, 0.0, 1.0), 0.0)
    self.pts = out
    self.valid = valid
    self.prev = cur
    return out + 1.0, valid.reshape(-1, 1), col(scores)


@method(PointTracker, "setPoints")
def _pt_set_points(self, interp, args, nargout):
    if self.pts is None:
        raise MatlabError("setPoints: initialize the tracker first.", "mylab:vision:notInitialized")
    p = fmat(args[0], "setPoints", "points", 2)
    self.pts = p - 1.0
    self.valid = np.ones(len(p), dtype=bool) if len(args) < 2 else \
        np.asarray(args[1]).astype(bool).reshape(-1)
    return None


@method(PointTracker, "release")
def _pt_release(self, interp, args, nargout):
    self.pts = self.prev = self.valid = None
    return None


system_object(PointTracker)


@builtin("vision.PointTracker")
def _point_tracker(interp, args, nargout):
    _, opts = options(list(args), PointTracker.PROPS, "vision.PointTracker", 0)
    t = PointTracker()
    t.apply_options(opts)
    return t


# -- optical flow ------------------------------------------------------------------------------------------

class OpticalFlow(VisionObject):
    CLASS = "opticalFlow"
    PROPS = ("Vx", "Vy", "Orientation", "Magnitude")

    def __init__(self, vx: np.ndarray, vy: np.ndarray):
        super().__init__()
        self.vx = np.asarray(vx, dtype=float)
        self.vy = np.asarray(vy, dtype=float)

    def prop(self, name: str) -> Any:
        if name == "Vx":
            return self.vx.copy()
        if name == "Vy":
            return self.vy.copy()
        if name == "Orientation":
            return np.arctan2(self.vy, self.vx)
        return np.hypot(self.vx, self.vy)


@method(OpticalFlow, "plot")
def _flow_plot(self, interp, args, nargout):
    ax = []
    if args and hasattr(args[0], "isa_class"):
        ax, args = [args[0]], args[1:]
    _, opts = options(list(args), ("DecimationFactor", "ScaleFactor", "Parent"), "plot", 0)
    if "Parent" in opts:
        ax = [opts["Parent"]]
    dec = fvec(opts["DecimationFactor"], "plot", "DecimationFactor") if "DecimationFactor" in opts \
        else np.array([1.0, 1.0])
    dec = np.repeat(dec, 2) if dec.size == 1 else dec
    sc = fscalar(opts["ScaleFactor"], "plot", "ScaleFactor") if "ScaleFactor" in opts else 1.0
    H, W = self.vx.shape
    ys = np.arange(0, H, int(dec[0]))
    xs = np.arange(0, W, int(dec[1]))
    X, Y = np.meshgrid(xs + 1.0, ys + 1.0)
    U = self.vx[np.ix_(ys, xs)] * sc
    V = self.vy[np.ix_(ys, xs)] * sc
    interp.call_named("quiver", [*ax, X, Y, U, V, scalar(0)], 0, interp.frame)
    return None


@builtin("opticalFlow")
def _optical_flow(interp, args, nargout):
    if not args:
        return OpticalFlow(np.zeros((0, 0)), np.zeros((0, 0)))
    if len(args) == 1:
        c = np.asarray(args[0])
        if not np.iscomplexobj(c):
            raise MatlabError("opticalFlow: give opticalFlow(Vx, Vy) or a complex matrix Vx + 1i*Vy.")
        return OpticalFlow(c.real, c.imag)
    vx, vy = np.asarray(args[0], dtype=float), np.asarray(args[1], dtype=float)
    if vx.shape != vy.shape:
        raise MatlabError("opticalFlow: Vx and Vy must have the same size.")
    return OpticalFlow(vx, vy)


class _FlowEstimator(VisionObject):
    is_handle = True

    def __init__(self, **props):
        super().__init__(**props)
        self.prev = None

    def compute(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _estimate_flow(self: _FlowEstimator, interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("estimateFlow: expected estimateFlow(flowObject, I).", "MATLAB:minrhs")
    img = args[0]
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise MatlabError("estimateFlow: the frame must be a 2-D grayscale image.", "mylab:vision:invalidImage")
    cur = gray_float(img, "estimateFlow")
    if self.prev is None or self.prev.shape != cur.shape:
        self.prev = cur
        z = np.zeros(cur.shape)
        return OpticalFlow(z, z.copy())
    # the gradient-based estimators work on 0..255 intensities (thresholds and smoothness are in those units)
    vx, vy = self.compute(self.prev * 255.0, cur * 255.0) if not isinstance(self, FlowFarneback) else \
        self.compute(self.prev, cur)
    self.prev = cur
    return OpticalFlow(vx, vy)


def _reset(self: _FlowEstimator, interp, args, nargout):
    self.prev = None
    return None


def _box(a: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage
    return ndimage.uniform_filter(a, size, mode="nearest")


def _derivs(a: np.ndarray, b: np.ndarray):
    from scipy import ndimage
    k = np.array([-1.0, 8.0, 0.0, -8.0, 1.0]) / 12.0  # five-point central difference
    m = (a + b) / 2
    ix = ndimage.correlate1d(m, -k, axis=1, mode="nearest")
    iy = ndimage.correlate1d(m, -k, axis=0, mode="nearest")
    return ix, iy, b - a


class FlowLK(_FlowEstimator):
    CLASS = "opticalFlowLK"
    PROPS = ("NoiseThreshold",)
    SETTABLE = frozenset(PROPS)

    def compute(self, a, b):
        from scipy import ndimage
        a = ndimage.gaussian_filter(a, 1.0)
        b = ndimage.gaussian_filter(b, 1.0)
        ix, iy, it = _derivs(a, b)
        w = 5
        sxx, syy, sxy = _box(ix * ix, w), _box(iy * iy, w), _box(ix * iy, w)
        sxt, syt = _box(ix * it, w), _box(iy * it, w)
        det = sxx * syy - sxy * sxy
        tr = sxx + syy
        lam_min = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        th = float(self.p["NoiseThreshold"].flat[0])
        ok = (lam_min > th) & (np.abs(det) > 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            vx = np.where(ok, (-syy * sxt + sxy * syt) / det, 0.0)
            vy = np.where(ok, (sxy * sxt - sxx * syt) / det, 0.0)
        return vx, vy


class FlowHS(_FlowEstimator):
    CLASS = "opticalFlowHS"
    PROPS = ("Smoothness", "MaxIteration", "VelocityDifference")
    SETTABLE = frozenset(PROPS)

    def compute(self, a, b):
        from scipy import ndimage
        ix, iy, it = _derivs(a, b)
        alpha2 = float(self.p["Smoothness"].flat[0])
        n = int(self.p["MaxIteration"].flat[0])
        tol = float(self.p["VelocityDifference"].flat[0])
        kern = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=float) / 12.0
        u = np.zeros_like(a)
        v = np.zeros_like(a)
        den = alpha2 + ix * ix + iy * iy
        for _ in range(n):
            ub = ndimage.convolve(u, kern, mode="nearest")
            vb = ndimage.convolve(v, kern, mode="nearest")
            t = (ix * ub + iy * vb + it) / den
            un, vn = ub - ix * t, vb - iy * t
            change = max(np.abs(un - u).max(), np.abs(vn - v).max())
            u, v = un, vn
            if change < tol:
                break
        return u, v


class FlowFarneback(_FlowEstimator):
    CLASS = "opticalFlowFarneback"
    PROPS = ("NumPyramidLevels", "PyramidScale", "NumIterations", "NeighborhoodSize", "FilterSize")
    SETTABLE = frozenset(PROPS)

    def compute(self, a, b):
        cv2 = cv2mod(self.CLASS)
        p = {k: float(v.flat[0]) for k, v in self.p.items()}
        n = int(p["NeighborhoodSize"])
        a8 = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
        b8 = np.clip(np.round(b * 255), 0, 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(a8, b8, None, p["PyramidScale"], int(p["NumPyramidLevels"]),
                                            int(p["FilterSize"]), int(p["NumIterations"]), n,
                                            1.1 if n <= 5 else 1.5, 0)
        return flow[:, :, 0].astype(float), flow[:, :, 1].astype(float)


for _cls in (FlowLK, FlowHS, FlowFarneback):
    method(_cls, "estimateFlow")(_estimate_flow)
    method(_cls, "reset")(_reset)


def _flow_ctor(cls, defaults: dict):
    def fn(interp, args, nargout):
        _, opts = options(list(args), cls.PROPS, cls.CLASS, 0)
        obj = cls(**{k: scalar(v) for k, v in defaults.items()})
        for k, v in opts.items():
            obj.p[k] = scalar(fscalar(v, cls.CLASS, k))
        return obj
    builtin(cls.CLASS)(fn)


_flow_ctor(FlowLK, {"NoiseThreshold": 0.0039})
_flow_ctor(FlowHS, {"Smoothness": 1.0, "MaxIteration": 10, "VelocityDifference": 0.0})
_flow_ctor(FlowFarneback, {"NumPyramidLevels": 3, "PyramidScale": 0.5, "NumIterations": 3, "NeighborhoodSize": 5,
                           "FilterSize": 15})


# -- Kalman filter ----------------------------------------------------------------------------------------

class KalmanFilter(VisionObject):
    CLASS = "vision.KalmanFilter"
    PROPS = ("StateTransitionModel", "MeasurementModel", "ControlModel", "State", "StateCovariance",
             "ProcessNoise", "MeasurementNoise")
    SETTABLE = frozenset(PROPS)
    is_handle = True

    def m(self, name: str) -> np.ndarray:
        return np.asarray(self.p[name], dtype=float)

    def check_set(self, name: str, value: Any) -> Any:
        v = np.asarray(value, dtype=float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if name == "State":
            return v.reshape(-1, 1)
        if name in ("StateCovariance", "ProcessNoise", "MeasurementNoise") and v.size == 1:
            n = self.m("StateTransitionModel").shape[0] if name != "MeasurementNoise" \
                else self.m("MeasurementModel").shape[0]
            return np.eye(n) * float(v.flat[0])
        return v


@method(KalmanFilter, "predict")
def _kf_predict(self, interp, args, nargout):
    A = self.m("StateTransitionModel")
    x = self.m("State").reshape(-1, 1)
    if args:
        B = self.m("ControlModel")
        if B.size == 0:
            raise MatlabError("predict: the filter has no ControlModel, so it takes no control input.")
        x = A @ x + B @ fvec(args[0], "predict", "control input").reshape(-1, 1)
    else:
        x = A @ x
    P = A @ self.m("StateCovariance") @ A.T + self.m("ProcessNoise")
    self.p["State"], self.p["StateCovariance"] = x, P
    H = self.m("MeasurementModel")
    return (H @ x).T, x.T, P


@method(KalmanFilter, "correct")
def _kf_correct(self, interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("correct: expected correct(kalmanFilter, z).", "MATLAB:minrhs")
    H = self.m("MeasurementModel")
    z = fvec(args[0], "correct", "measurement").reshape(-1, 1)
    if z.size != H.shape[0]:
        raise MatlabError(f"correct: the measurement must have {H.shape[0]} elements.")
    x = self.m("State").reshape(-1, 1)
    P = self.m("StateCovariance")
    S = H @ P @ H.T + self.m("MeasurementNoise")
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(P.shape[0]) - K @ H) @ P
    self.p["State"], self.p["StateCovariance"] = x, P
    return (H @ x).T, x.T, P


@method(KalmanFilter, "distance")
def _kf_distance(self, interp, args, nargout):
    H = self.m("MeasurementModel")
    Z = fmat(args[0], "distance", "measurements", H.shape[0])
    x = self.m("State").reshape(-1, 1)
    S = H @ self.m("StateCovariance") @ H.T + self.m("MeasurementNoise")
    Si = np.linalg.inv(S)
    d = Z - (H @ x).T
    m2 = np.einsum("ij,jk,ik->i", d, Si, d)
    return (m2 + np.log(np.linalg.det(S))).reshape(1, -1)


@method(KalmanFilter, "clone")
def _kf_clone(self, interp, args, nargout):
    c = self.clone()
    c.p = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in self.p.items()}
    return c


@builtin("vision.KalmanFilter")
def _kalman_filter(interp, args, nargout):
    pos, opts = options(list(args), KalmanFilter.PROPS, "vision.KalmanFilter", 3)
    A = fmat(pos[0], "vision.KalmanFilter", "StateTransitionModel") if pos else np.array([[1.0, 1.0], [0.0, 1.0]])
    H = fmat(pos[1], "vision.KalmanFilter", "MeasurementModel") if len(pos) > 1 else np.array([[1.0, 0.0]])
    B = fmat(pos[2], "vision.KalmanFilter", "ControlModel") if len(pos) > 2 else np.zeros((0, 0))
    n, m = A.shape[0], H.shape[0]
    if A.shape != (n, n) or H.shape[1] != n:
        raise MatlabError("vision.KalmanFilter: the state transition model must be N-by-N and the measurement "
                          "model M-by-N.", "mylab:vision:invalidModel")
    kf = KalmanFilter(StateTransitionModel=A, MeasurementModel=H, ControlModel=B, State=np.zeros((n, 1)),
                      StateCovariance=np.eye(n), ProcessNoise=np.eye(n), MeasurementNoise=np.eye(m))
    kf.apply_options(opts)
    return kf


@builtin("configureKalmanFilter")
def _configure_kalman_filter(interp, args, nargout):
    fname = "configureKalmanFilter"
    if len(args) != 5:
        raise MatlabError("configureKalmanFilter: expected configureKalmanFilter(MotionModel, InitialLocation, "
                          "InitialEstimateError, MotionNoise, MeasurementNoise).", "MATLAB:minrhs")
    if is_text(args[0]):
        model = text(args[0]).lower()
        order = {"constantvelocity": 2, "constantacceleration": 3}.get(model)
        if order is None:
            raise MatlabError("configureKalmanFilter: MotionModel must be 'ConstantVelocity' or "
                              "'ConstantAcceleration'.", f"mylab:{fname}:invalidModel")
    else:
        raise MatlabError("configureKalmanFilter: MotionModel must be 'ConstantVelocity' or 'ConstantAcceleration'.")
    loc = fvec(args[1], fname, "InitialLocation")
    d = loc.size
    est = fvec(args[2], fname, "InitialEstimateError")
    mot = fvec(args[3], fname, "MotionNoise")
    meas = fscalar(args[4], fname, "MeasurementNoise")
    if est.size != order or mot.size != order:
        raise MatlabError(f"configureKalmanFilter: InitialEstimateError and MotionNoise must have {order} elements "
                          "for this motion model.", f"mylab:{fname}:invalidSize")
    # per-dimension block of the state [position velocity (acceleration)], dimensions interleaved
    blk = np.array([[1.0, 1.0], [0.0, 1.0]]) if order == 2 else np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1.0]])
    A = np.kron(np.eye(d), blk)
    H = np.kron(np.eye(d), np.eye(1, order))
    x0 = np.zeros((d * order, 1))
    x0[::order, 0] = loc
    P0 = np.diag(np.tile(est, d))
    Q = np.diag(np.tile(mot, d))
    R = np.eye(d) * meas
    return KalmanFilter(StateTransitionModel=A, MeasurementModel=H, ControlModel=np.zeros((0, 0)), State=x0,
                        StateCovariance=P0, ProcessNoise=Q, MeasurementNoise=R)

