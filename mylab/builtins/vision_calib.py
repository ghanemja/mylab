"""Camera models and calibration: cameraIntrinsics, cameraParameters, stereoParameters, checkerboard
detection and generation, calibration, undistortion, projection, stereo rectification, disparity and
3-D reconstruction.

Conventions: the intrinsic matrix K = [fx s cx; 0 fy cy; 0 0 1] uses MATLAB's 1-based pixel
coordinates (OpenCV's principal point is one pixel smaller in each direction). Lens distortion is
the usual radial (k1, k2, k3) plus tangential (p1, p2) model. Extrinsics of the newer functions are
rigidtform3d objects that map world points into camera coordinates; the older functions use the
row-vector convention [x y z] = [X Y Z] * R + t.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, char, is_char
from ._cv import (
    VisionObject, cv2mod, flag, fmat, fscalar, fvec, gray_u8, image_arg, method, opt_text, options, row,
    scalar, text_list,
)
from ._registry import builtin
from ._util import is_text, text
from .vision_geometry import GeoTform3, rodrigues, tform3_arg

# -- distortion model --------------------------------------------------------------------------------------


def distort_normalized(xn: np.ndarray, radial: np.ndarray, tang: np.ndarray) -> np.ndarray:
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    k = np.zeros(3)
    k[:radial.size] = radial[:3]
    rad = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    p1, p2 = (tang[0], tang[1]) if tang.size >= 2 else (0.0, 0.0)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([xd, yd])


def _undistort_iter(xd: np.ndarray, radial: np.ndarray, tang: np.ndarray, iters: int = 50) -> np.ndarray:
    """Invert the distortion model by fixed-point iteration (as OpenCV does), then polish with Newton steps."""
    x = xd.copy()
    if not np.any(radial) and not np.any(tang):
        return x
    for _ in range(iters):
        d = distort_normalized(x, radial, tang) - x
        x = xd - d
    return x


class CameraIntrinsics(VisionObject):
    CLASS = "cameraIntrinsics"
    PROPS = ("FocalLength", "PrincipalPoint", "ImageSize", "RadialDistortion", "TangentialDistortion", "Skew", "K")

    def __init__(self, focal, pp, size, radial=(0.0, 0.0), tang=(0.0, 0.0), skew=0.0):
        super().__init__()
        self.focal = np.asarray(focal, dtype=float).reshape(2)
        self.pp = np.asarray(pp, dtype=float).reshape(2)
        self.size = np.asarray(size, dtype=float).reshape(2)
        self.radial = np.asarray(radial, dtype=float).reshape(-1)
        self.tang = np.asarray(tang, dtype=float).reshape(-1)
        self.skew = float(skew)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal[0], self.skew, self.pp[0]], [0, self.focal[1], self.pp[1]], [0, 0, 1.0]])

    def K0(self) -> np.ndarray:
        """K for OpenCV's 0-based pixel coordinates."""
        K = self.K
        K[0, 2] -= 1
        K[1, 2] -= 1
        return K

    def dist_cv(self) -> np.ndarray:
        k = np.zeros(3)
        k[:self.radial.size] = self.radial[:3]
        p = self.tang if self.tang.size == 2 else np.zeros(2)
        return np.array([k[0], k[1], p[0], p[1], k[2]])

    def prop(self, name: str) -> Any:
        return {"FocalLength": row(self.focal), "PrincipalPoint": row(self.pp), "ImageSize": row(self.size),
                "RadialDistortion": row(self.radial), "TangentialDistortion": row(self.tang),
                "Skew": scalar(self.skew), "K": self.K}[name]

    # pixels <-> normalized camera coordinates
    def to_normalized(self, p: np.ndarray) -> np.ndarray:
        y = (p[:, 1] - self.pp[1]) / self.focal[1]
        x = (p[:, 0] - self.pp[0] - self.skew * y) / self.focal[0]
        return _undistort_iter(np.column_stack([x, y]), self.radial, self.tang)

    def to_pixels(self, xn: np.ndarray, distort: bool = True) -> np.ndarray:
        if distort:
            xn = distort_normalized(xn, self.radial, self.tang)
        u = self.focal[0] * xn[:, 0] + self.skew * xn[:, 1] + self.pp[0]
        v = self.focal[1] * xn[:, 1] + self.pp[1]
        return np.column_stack([u, v])

    def undistort_pixels(self, p: np.ndarray) -> np.ndarray:
        return self.to_pixels(self.to_normalized(p), distort=False)


def intrinsics_of(v: Any, fname: str) -> CameraIntrinsics:
    if isinstance(v, CameraIntrinsics):
        return v
    if isinstance(v, CameraParameters):
        return v.intr
    raise MatlabError(f"{fname}: expected a cameraIntrinsics or cameraParameters object.",
                      f"mylab:{fname}:invalidIntrinsics")


def undistort_normalized(p: np.ndarray, intr) -> np.ndarray:
    return intrinsics_of(intr, "undistort").to_normalized(p)


@builtin("cameraIntrinsics")
def _camera_intrinsics(interp, args, nargout):
    fname = "cameraIntrinsics"
    pos, opts = options(list(args), ("RadialDistortion", "TangentialDistortion", "Skew"), fname, 3)
    if len(pos) != 3:
        raise MatlabError("cameraIntrinsics: expected cameraIntrinsics(focalLength, principalPoint, imageSize, ...).",
                          "MATLAB:minrhs")
    f = fvec(pos[0], fname, "focalLength")
    f = np.repeat(f, 2) if f.size == 1 else f
    pp = fvec(pos[1], fname, "principalPoint", 2)
    sz = fvec(pos[2], fname, "imageSize", 2)
    if f.size != 2 or np.any(f <= 0):
        raise MatlabError("cameraIntrinsics: focalLength must be a positive scalar or [fx fy].")
    rad = fvec(opts["RadialDistortion"], fname, "RadialDistortion") if "RadialDistortion" in opts else np.zeros(2)
    if rad.size not in (2, 3):
        raise MatlabError("cameraIntrinsics: RadialDistortion must have two or three elements.")
    tan = fvec(opts["TangentialDistortion"], fname, "TangentialDistortion", 2) if "TangentialDistortion" in opts \
        else np.zeros(2)
    skew = fscalar(opts["Skew"], fname, "Skew") if "Skew" in opts else 0.0
    return CameraIntrinsics(f, pp, sz, rad, tan, skew)


# -- cameraParameters ------------------------------------------------------------------------------------

class CameraParameters(VisionObject):
    CLASS = "cameraParameters"
    PROPS = ("Intrinsics", "K", "IntrinsicMatrix", "FocalLength", "PrincipalPoint", "Skew", "RadialDistortion",
             "TangentialDistortion", "ImageSize", "NumPatterns", "PatternExtrinsics", "RotationVectors",
             "TranslationVectors", "RotationMatrices", "WorldPoints", "WorldUnits", "EstimateSkew",
             "NumRadialDistortionCoefficients", "EstimateTangentialDistortion", "ReprojectionErrors",
             "ReprojectedPoints", "DetectedKeypoints", "MeanReprojectionError")

    def __init__(self, intr: CameraIntrinsics, rvecs=None, tvecs=None, world=None, image_points=None,
                 units="mm", est_skew=False, n_radial=2, est_tan=False):
        super().__init__()
        self.intr = intr
        self.rvecs = np.zeros((0, 3)) if rvecs is None else np.asarray(rvecs, dtype=float).reshape(-1, 3)
        self.tvecs = np.zeros((0, 3)) if tvecs is None else np.asarray(tvecs, dtype=float).reshape(-1, 3)
        self.world = np.zeros((0, 2)) if world is None else np.asarray(world, dtype=float)
        self.image_points = image_points  # M-by-2-by-P (NaN where undetected) or None
        self.units = units
        self.est_skew = est_skew
        self.n_radial = n_radial
        self.est_tan = est_tan

    @property
    def n(self) -> int:
        return self.rvecs.shape[0]

    def extrinsic(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return rodrigues(self.rvecs[k]), self.tvecs[k]

    def reprojected(self) -> np.ndarray:
        if self.n == 0 or self.world.size == 0:
            return np.zeros((0, 2))
        W = self.world
        W3 = np.column_stack([W, np.zeros(len(W))]) if W.shape[1] == 2 else W
        out = np.zeros((W3.shape[0], 2, self.n))
        for k in range(self.n):
            R, t = self.extrinsic(k)
            C = W3 @ R.T + t
            out[:, :, k] = self.intr.to_pixels(C[:, :2] / C[:, 2:3])
        return out

    def errors(self) -> np.ndarray:
        if self.image_points is None or self.n == 0:
            return np.zeros((0, 2))
        return self.image_points - self.reprojected()

    def mean_error(self) -> float:
        e = self.errors()
        if e.size == 0:
            return 0.0
        d = np.sqrt((e ** 2).sum(axis=1))
        return float(np.nanmean(d))

    def prop(self, name: str) -> Any:
        i = self.intr
        if name in ("FocalLength", "PrincipalPoint", "Skew", "RadialDistortion", "TangentialDistortion",
                    "ImageSize", "K"):
            return i.prop(name)
        if name == "Intrinsics":
            return i
        if name == "IntrinsicMatrix":
            return i.K.T
        if name == "NumPatterns":
            return scalar(self.n)
        if name == "PatternExtrinsics":
            data = np.empty((self.n, 1), dtype=object)
            for k in range(self.n):
                A = np.eye(4)
                A[:3, :3], A[:3, 3] = self.extrinsic(k)
                data[k, 0] = GeoTform3("rigidtform3d", A)
            return Cell(data)
        if name == "RotationVectors":
            return self.rvecs.copy()
        if name == "TranslationVectors":
            return self.tvecs.copy()
        if name == "RotationMatrices":
            if self.n == 0:
                return np.zeros((3, 3, 0))
            return np.stack([self.extrinsic(k)[0].T for k in range(self.n)], axis=2)
        if name == "WorldPoints":
            return self.world.copy()
        if name == "WorldUnits":
            return char(self.units)
        if name == "EstimateSkew":
            return np.array([[self.est_skew]])
        if name == "NumRadialDistortionCoefficients":
            return scalar(self.n_radial)
        if name == "EstimateTangentialDistortion":
            return np.array([[self.est_tan]])
        if name == "ReprojectionErrors":
            return self.errors()
        if name == "ReprojectedPoints":
            return self.reprojected()
        if name == "DetectedKeypoints":
            if self.image_points is None:
                return np.zeros((0, self.n), dtype=bool)
            return ~np.isnan(self.image_points[:, 0, :])
        if name == "MeanReprojectionError":
            return scalar(self.mean_error())
        raise KeyError(name)


@method(CameraParameters, "pointsToWorld")
def _cp_points_to_world(self, interp, args, nargout):
    return _points_to_world(self.intr, args, "pointsToWorld")


@builtin("cameraParameters")
def _camera_parameters(interp, args, nargout):
    fname = "cameraParameters"
    names = ("K", "IntrinsicMatrix", "RadialDistortion", "TangentialDistortion", "Skew", "ImageSize",
             "WorldUnits", "EstimateSkew", "NumRadialDistortionCoefficients", "EstimateTangentialDistortion",
             "RotationVectors", "TranslationVectors", "WorldPoints", "ReprojectionErrors")
    pos, opts = options(list(args), names, fname, 0)
    K = np.eye(3)
    if "K" in opts:
        K = fmat(opts["K"], fname, "K", 3)
    elif "IntrinsicMatrix" in opts:
        K = fmat(opts["IntrinsicMatrix"], fname, "IntrinsicMatrix", 3).T
    rad = fvec(opts["RadialDistortion"], fname, "RadialDistortion") if "RadialDistortion" in opts else np.zeros(2)
    tan = fvec(opts["TangentialDistortion"], fname, "TangentialDistortion", 2) if "TangentialDistortion" in opts \
        else np.zeros(2)
    size = fvec(opts["ImageSize"], fname, "ImageSize", 2) if "ImageSize" in opts else np.zeros(2)
    skew = fscalar(opts["Skew"], fname, "Skew") if "Skew" in opts else K[0, 1]
    intr = CameraIntrinsics([K[0, 0], K[1, 1]], [K[0, 2], K[1, 2]], size, rad, tan, skew)
    rv = fmat(opts["RotationVectors"], fname, "RotationVectors", 3) if "RotationVectors" in opts else None
    tv = fmat(opts["TranslationVectors"], fname, "TranslationVectors", 3) if "TranslationVectors" in opts else None
    if (rv is None) != (tv is None) or (rv is not None and rv.shape != tv.shape):
        raise MatlabError("cameraParameters: give RotationVectors and TranslationVectors together, one row per "
                          "pattern.")
    world = fmat(opts["WorldPoints"], fname, "WorldPoints") if "WorldPoints" in opts else None
    pts = None
    if "ReprojectionErrors" in opts and rv is not None and world is not None:
        cp = CameraParameters(intr, rv, tv, world)
        pts = cp.reprojected() + np.asarray(opts["ReprojectionErrors"], dtype=float)
    units = text(opts["WorldUnits"]) if "WorldUnits" in opts else "mm"
    return CameraParameters(intr, rv, tv, world, pts, units,
                            flag(opts["EstimateSkew"]) if "EstimateSkew" in opts else False,
                            int(fscalar(opts["NumRadialDistortionCoefficients"], fname, "N"))
                            if "NumRadialDistortionCoefficients" in opts else rad.size,
                            flag(opts["EstimateTangentialDistortion"]) if "EstimateTangentialDistortion" in opts
                            else False)


# -- checkerboards ---------------------------------------------------------------------------------------

@builtin("generateCheckerboardPoints")
def _generate_checkerboard_points(interp, args, nargout):
    fname = "generateCheckerboardPoints"
    pos, _ = options(list(args), ("OriginCheckerColor",), fname, 2)
    if len(pos) != 2:
        raise MatlabError("generateCheckerboardPoints: expected generateCheckerboardPoints(boardSize, squareSize).",
                          "MATLAB:minrhs")
    bs = fvec(pos[0], fname, "boardSize", 2).astype(int)
    sq = fscalar(pos[1], fname, "squareSize")
    if np.any(bs < 2):
        raise MatlabError("generateCheckerboardPoints: boardSize must be at least [2 2] squares.")
    r, c = bs[0] - 1, bs[1] - 1
    jj, ii = np.meshgrid(np.arange(c), np.arange(r))
    return np.column_stack([jj.ravel(order="F") * sq, ii.ravel(order="F") * sq])


def _order_grid(G: np.ndarray) -> np.ndarray:
    """Arrange a corner grid so axis 0 runs down the image and axis 1 runs to the right."""
    best, best_score = G, -np.inf
    for t in (False, True):
        A = G.transpose(1, 0, 2) if t else G
        for f0 in (False, True):
            for f1 in (False, True):
                B = A[::-1] if f0 else A
                B = B[:, ::-1] if f1 else B
                d0 = np.diff(B, axis=0).reshape(-1, 2).mean(0) if B.shape[0] > 1 else np.zeros(2)
                d1 = np.diff(B, axis=1).reshape(-1, 2).mean(0) if B.shape[1] > 1 else np.zeros(2)
                n0, n1 = np.linalg.norm(d0) or 1.0, np.linalg.norm(d1) or 1.0
                score = d0[1] / n0 + d1[0] / n1
                if score > best_score + 1e-9:
                    best, best_score = B, score
    return best


def detect_board(g: np.ndarray):
    """Corner grid (rows, cols, 2) in 1-based pixels, or None."""
    cv2 = cv2mod("detectCheckerboardPoints")
    flags = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    try:
        ok, corners, meta = cv2.findChessboardCornersSBWithMeta(g, (3, 3), flags)
    except cv2.error:
        ok = False
    if ok and corners is not None and meta is not None:
        nr, nc = meta.shape[:2]
        c = corners.reshape(-1, 2)
        if c.shape[0] == nr * nc:
            G = c.reshape(nr, nc, 2).astype(float) + 1.0
            return _order_grid(G)
    return None


def _images_arg(v: Any, fname: str) -> list[np.ndarray]:
    if isinstance(v, Cell):
        return [_read_image(x, fname) for x in v.data.flatten(order="F")]
    if is_text(v) or not isinstance(v, np.ndarray) or v.dtype.kind == "U":
        names = text_list(v)
        return [_read_image(char(n), fname) for n in names]
    a = image_arg(v, fname)
    if a.ndim == 4:
        return [a[:, :, :, k] if a.shape[2] == 3 else a[:, :, 0, k] for k in range(a.shape[3])]
    if a.ndim == 3 and a.shape[2] != 3:
        return [a[:, :, k] for k in range(a.shape[2])]
    return [a]


def _read_image(v: Any, fname: str) -> np.ndarray:
    if isinstance(v, np.ndarray) and not is_char(v):
        return image_arg(v, fname)
    path = text(v)
    if not os.path.exists(path):
        raise MatlabError(f"{fname}: cannot find the image file '{path}'.")
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


@builtin("detectCheckerboardPoints")
def _detect_checkerboard_points(interp, args, nargout):
    fname = "detectCheckerboardPoints"
    pos, _ = options(list(args), ("MinCornerMetric", "PartialDetections", "HighDistortion", "ShowProgressBar"),
                     fname, 2)
    if not pos:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    sets = [_images_arg(p, fname) for p in pos]
    if len(sets) == 2 and len(sets[0]) != len(sets[1]):
        raise MatlabError("detectCheckerboardPoints: the two image sets must have the same number of images.")
    grids = [[detect_board(gray_u8(im, fname)) for im in s] for s in sets]
    sizes = [g.shape[:2] for gs in grids for g in gs if g is not None]
    if not sizes:
        empty = np.zeros((0, 2))
        used = np.zeros((len(sets[0]), 1), dtype=bool)
        return empty, np.zeros((1, 2)), used
    # the board size seen most often wins; images with other sizes are not used
    vals, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
    shape = tuple(vals[np.argmax(counts)])
    used = np.array([all(gs[k] is not None and gs[k].shape[:2] == shape for gs in grids)
                     for k in range(len(sets[0]))])
    M = shape[0] * shape[1]
    idx = np.nonzero(used)[0]
    out = np.zeros((M, 2, len(idx), len(sets)))
    for s, gs in enumerate(grids):
        for j, k in enumerate(idx):
            G = gs[k]
            out[:, 0, j, s] = G[:, :, 0].ravel(order="F")
            out[:, 1, j, s] = G[:, :, 1].ravel(order="F")
    if len(sets) == 1:
        out = out[:, :, :, 0]
        if out.shape[2] == 1:
            out = out[:, :, 0]
    board = np.array([[shape[0] + 1.0, shape[1] + 1.0]])
    return out, board, used.reshape(-1, 1)


# -- calibration ---------------------------------------------------------------------------------------------

def _calib_flags(cv2, n_radial: int, est_tan: bool) -> int:
    f = 0
    if n_radial < 3:
        f |= cv2.CALIB_FIX_K3
    if not est_tan:
        f |= cv2.CALIB_ZERO_TANGENT_DIST
    return f


def calibrate_single(P: np.ndarray, world: np.ndarray, size: tuple[int, int], n_radial: int, est_tan: bool,
                     units: str, est_skew: bool):
    cv2 = cv2mod("estimateCameraParameters")
    W3 = np.column_stack([world, np.zeros(len(world))]).astype(np.float32)
    obj, img, keep = [], [], []
    for k in range(P.shape[2]):
        p = P[:, :, k]
        ok = ~np.isnan(p).any(axis=1)
        if ok.sum() >= 4:
            obj.append(W3[ok])
            img.append((p[ok] - 1.0).astype(np.float32).reshape(-1, 1, 2))
            keep.append(k)
    if len(obj) < 2:
        raise MatlabError("estimateCameraParameters: at least two views of the pattern are needed.",
                          "mylab:estimateCameraParameters:notEnoughViews")
    H, W = size
    flags = _calib_flags(cv2, n_radial, est_tan)
    rms, K, dist, rvecs, tvecs, std_i, _, _ = cv2.calibrateCameraExtended(obj, img, (int(W), int(H)), None, None,
                                                                           flags=flags)
    dist = dist.ravel()
    radial = np.array([dist[0], dist[1], dist[4] if dist.size > 4 else 0.0])[:max(n_radial, 2)]
    intr = CameraIntrinsics([K[0, 0], K[1, 1]], [K[0, 2] + 1, K[1, 2] + 1], [H, W], radial,
                            dist[2:4] if est_tan else np.zeros(2), 0.0)
    rv = np.vstack([np.asarray(r).ravel() for r in rvecs])
    tv = np.vstack([np.asarray(t).ravel() for t in tvecs])
    params = CameraParameters(intr, rv, tv, world, P[:, :, keep], units, est_skew, max(n_radial, 2), est_tan)
    std = np.asarray(std_i).ravel()
    return params, np.array(keep), std


@builtin("estimateCameraParameters")
def _estimate_camera_parameters(interp, args, nargout):
    fname = "estimateCameraParameters"
    names = ("EstimateSkew", "NumRadialDistortionCoefficients", "EstimateTangentialDistortion", "WorldUnits",
             "InitialIntrinsicMatrix", "InitialRadialDistortion", "ImageSize", "ShowProgressBar",
             "InitialK")
    pos, opts = options(list(args), names, fname, 2)
    if len(pos) != 2:
        raise MatlabError("estimateCameraParameters: expected estimateCameraParameters(imagePoints, worldPoints).",
                          "MATLAB:minrhs")
    P = np.asarray(pos[0], dtype=float)
    world = fmat(pos[1], fname, "worldPoints", 2)
    if P.ndim == 2:
        P = P[:, :, None]
    if P.shape[0] != world.shape[0] or P.shape[1] != 2:
        raise MatlabError("estimateCameraParameters: imagePoints must be M-by-2-by-P with M = the number of world "
                          "points.", "mylab:estimateCameraParameters:sizeMismatch")
    n_radial = int(fscalar(opts["NumRadialDistortionCoefficients"], fname, "NumRadialDistortionCoefficients")) \
        if "NumRadialDistortionCoefficients" in opts else 2
    if n_radial not in (2, 3):
        raise MatlabError("estimateCameraParameters: NumRadialDistortionCoefficients must be 2 or 3.")
    est_tan = flag(opts["EstimateTangentialDistortion"]) if "EstimateTangentialDistortion" in opts else False
    est_skew = flag(opts["EstimateSkew"]) if "EstimateSkew" in opts else False
    units = text(opts["WorldUnits"]) if "WorldUnits" in opts else "mm"
    if "ImageSize" in opts:
        sz = fvec(opts["ImageSize"], fname, "ImageSize", 2)
    else:  # guess from the extent of the points
        sz = np.ceil(np.array([np.nanmax(P[:, 1]), np.nanmax(P[:, 0])]) * 1.1)
    size = (int(sz[0]), int(sz[1]))
    if P.ndim == 4:
        if P.shape[3] != 2:
            raise MatlabError("estimateCameraParameters: stereo image points must be M-by-2-by-P-by-2.")
        sp = calibrate_stereo(P[:, :, :, 0], P[:, :, :, 1], world, size, n_radial, est_tan, units, est_skew)
        used = np.ones((P.shape[2], 1), dtype=bool)
        return sp, used, Struct.scalar({})
    params, keep, std = calibrate_single(P, world, size, n_radial, est_tan, units, est_skew)
    used = np.zeros((P.shape[2], 1), dtype=bool)
    used[keep] = True
    errs = Struct.scalar({"FocalLengthError": row(std[0:2]), "PrincipalPointError": row(std[2:4]),
                   "RadialDistortionError": row(np.array([std[4], std[5], std[8]])[:n_radial]),
                   "TangentialDistortionError": row(std[6:8] if est_tan else np.zeros(2))})
    return params, used, errs


# -- stereo ------------------------------------------------------------------------------------------------

class StereoParameters(VisionObject):
    CLASS = "stereoParameters"
    PROPS = ("CameraParameters1", "CameraParameters2", "PoseCamera2", "RotationOfCamera2", "TranslationOfCamera2",
             "FundamentalMatrix", "EssentialMatrix", "MeanReprojectionError", "NumPatterns", "WorldPoints",
             "WorldUnits")
    stereo_cam1 = True

    def __init__(self, c1: CameraParameters, c2: CameraParameters, R: np.ndarray, t: np.ndarray, mre: float = 0.0):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.R = np.asarray(R, dtype=float)   # X2 = R X1 + t (camera-1 coordinates to camera-2)
        self.t = np.asarray(t, dtype=float).reshape(3)
        self.mre = mre

    @property
    def cam1(self) -> CameraIntrinsics:
        return self.c1.intr

    @property
    def cam2(self) -> CameraIntrinsics:
        return self.c2.intr

    def essential(self) -> np.ndarray:
        tx = np.array([[0, -self.t[2], self.t[1]], [self.t[2], 0, -self.t[0]], [-self.t[1], self.t[0], 0]])
        return tx @ self.R

    def fundamental(self) -> np.ndarray:
        F = np.linalg.inv(self.cam2.K).T @ self.essential() @ np.linalg.inv(self.cam1.K)
        return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F

    def projection_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        P1 = self.cam1.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.cam2.K @ np.hstack([self.R, self.t.reshape(3, 1)])
        return P1, P2

    def prop(self, name: str) -> Any:
        if name == "CameraParameters1":
            return self.c1
        if name == "CameraParameters2":
            return self.c2
        if name == "PoseCamera2":
            A = np.eye(4)
            A[:3, :3] = self.R.T
            A[:3, 3] = -self.R.T @ self.t
            return GeoTform3("rigidtform3d", A)
        if name == "RotationOfCamera2":
            return self.R.T.copy()
        if name == "TranslationOfCamera2":
            return row(self.t)
        if name == "FundamentalMatrix":
            return self.fundamental()
        if name == "EssentialMatrix":
            return self.essential()
        if name == "MeanReprojectionError":
            return scalar(self.mre)
        if name == "NumPatterns":
            return scalar(self.c1.n)
        if name == "WorldPoints":
            return self.c1.world.copy()
        if name == "WorldUnits":
            return char(self.c1.units)
        raise KeyError(name)


@builtin("stereoParameters")
def _stereo_parameters(interp, args, nargout):
    if len(args) != 4 or not all(isinstance(a, CameraParameters) for a in args[:2]):
        raise MatlabError("stereoParameters: expected stereoParameters(cameraParams1, cameraParams2, "
                          "rotationOfCamera2, translationOfCamera2).", "MATLAB:minrhs")
    R = fmat(args[2], "stereoParameters", "rotationOfCamera2", 3)
    t = fvec(args[3], "stereoParameters", "translationOfCamera2", 3)
    return StereoParameters(args[0], args[1], R.T, t)


def calibrate_stereo(P1, P2, world, size, n_radial, est_tan, units, est_skew) -> StereoParameters:
    cv2 = cv2mod("estimateCameraParameters")
    c1, k1, _ = calibrate_single(P1, world, size, n_radial, est_tan, units, est_skew)
    c2, k2, _ = calibrate_single(P2, world, size, n_radial, est_tan, units, est_skew)
    W3 = np.column_stack([world, np.zeros(len(world))]).astype(np.float32)
    obj, a, b = [], [], []
    for k in range(P1.shape[2]):
        ok = ~(np.isnan(P1[:, :, k]).any(1) | np.isnan(P2[:, :, k]).any(1))
        if ok.sum() >= 4:
            obj.append(W3[ok])
            a.append((P1[ok, :, k] - 1).astype(np.float32).reshape(-1, 1, 2))
            b.append((P2[ok, :, k] - 1).astype(np.float32).reshape(-1, 1, 2))
    # refine both cameras and their relative pose together, starting from the single-camera results
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | _calib_flags(cv2, n_radial, est_tan)
    res = cv2.stereoCalibrate(obj, a, b, c1.intr.K0(), c1.intr.dist_cv(), c2.intr.K0(), c2.intr.dist_cv(),
                              (int(size[1]), int(size[0])), flags=flags)
    rms, Ka, da, Kb, db, R, t = res[:7]
    for cam, Kn, dn in ((c1, Ka, da), (c2, Kb, db)):
        dn = np.asarray(dn).ravel()
        radial = np.array([dn[0], dn[1], dn[4] if dn.size > 4 else 0.0])[:max(n_radial, 2)]
        cam.intr = CameraIntrinsics([Kn[0, 0], Kn[1, 1]], [Kn[0, 2] + 1, Kn[1, 2] + 1], cam.intr.size, radial,
                                    dn[2:4] if est_tan else np.zeros(2), 0.0)
    return StereoParameters(c1, c2, R, np.asarray(t).ravel(), float(rms))


def _rectify_maps(sp: StereoParameters, size: tuple[int, int], alpha: float):
    cv2 = cv2mod("rectifyStereoImages")
    H, W = size
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(sp.cam1.K0(), sp.cam1.dist_cv(), sp.cam2.K0(),
                                                      sp.cam2.dist_cv(), (W, H), sp.R, sp.t.reshape(3, 1),
                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    m1 = cv2.initUndistortRectifyMap(sp.cam1.K0(), sp.cam1.dist_cv(), R1, P1, (W, H), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(sp.cam2.K0(), sp.cam2.dist_cv(), R2, P2, (W, H), cv2.CV_32FC1)
    return m1, m2, Q, roi1, roi2


def _q_matlab(Q: np.ndarray) -> np.ndarray:
    """OpenCV's Q (0-based, column vectors) as a matrix for [x y d 1] * Qm with 1-based x and y."""
    T = np.eye(4)
    T[0, 3] = -1.0
    T[1, 3] = -1.0
    return (Q @ T).T


def _remap(cv2, img: np.ndarray, m) -> np.ndarray:
    a = np.ascontiguousarray(img)
    work = a.astype(np.float32) if a.dtype not in (np.uint8, np.uint16, np.float32) else a
    if a.dtype == bool:
        work = a.astype(np.uint8)
    out = cv2.remap(work, m[0], m[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(a.dtype) if a.dtype != bool else out > 0


@builtin("rectifyStereoImages")
def _rectify_stereo_images(interp, args, nargout):
    fname = "rectifyStereoImages"
    pos, opts = options(list(args), ("OutputView", "FillValues"), fname, 4)
    if len(pos) < 3 or not isinstance(pos[2], StereoParameters):
        raise MatlabError("rectifyStereoImages: expected rectifyStereoImages(I1, I2, stereoParams).",
                          "MATLAB:minrhs")
    cv2 = cv2mod(fname)
    a, b = image_arg(pos[0], fname), image_arg(pos[1], fname)
    view = opt_text(opts, "OutputView", "valid", ("valid", "full"), fname)
    m1, m2, Q, roi1, roi2 = _rectify_maps(pos[2], a.shape[:2], 1.0 if view == "full" else 0.0)
    J1, J2 = _remap(cv2, a, m1), _remap(cv2, b, m2)
    if view == "valid":
        x0 = max(roi1[0], roi2[0])
        y0 = max(roi1[1], roi2[1])
        x1 = min(roi1[0] + roi1[2], roi2[0] + roi2[2])
        y1 = min(roi1[1] + roi1[3], roi2[1] + roi2[3])
        if x1 > x0 and y1 > y0:
            J1, J2 = J1[y0:y1, x0:x1], J2[y0:y1, x0:x1]
            Q = Q.copy()
            Q[0, 3] += x0
            Q[1, 3] += y0
    return J1, J2, _q_matlab(Q)


def _disparity_inputs(pos, fname):
    if len(pos) != 2:
        raise MatlabError(f"{fname}: expected {fname}(I1, I2, ...).", "MATLAB:minrhs")
    a, b = gray_u8(pos[0], fname), gray_u8(pos[1], fname)
    if image_arg(pos[0], fname).ndim != 2 or image_arg(pos[1], fname).ndim != 2:
        raise MatlabError(f"{fname}: the images must be 2-D grayscale images.", f"mylab:{fname}:invalidImage")
    if a.shape != b.shape:
        raise MatlabError(f"{fname}: the two images must have the same size.", f"mylab:{fname}:sizeMismatch")
    return a, b


def _disp_range(opts, default, fname):
    r = fvec(opts["DisparityRange"], fname, "DisparityRange", 2).astype(int) if "DisparityRange" in opts \
        else np.array(default)
    if r[1] <= r[0] or (r[1] - r[0]) % 8 != 0:
        raise MatlabError(f"{fname}: DisparityRange [min max] must have max - min a positive multiple of 8.",
                          f"mylab:{fname}:invalidDisparityRange")
    return int(r[0]), int(r[1] - r[0])


def _to_disp(raw: np.ndarray, dmin: int) -> np.ndarray:
    d = raw.astype(np.float32) / 16.0
    d[raw < dmin * 16] = np.nan
    return d


@builtin("disparitySGM")
def _disparity_sgm(interp, args, nargout):
    fname = "disparitySGM"
    pos, opts = options(list(args), ("DisparityRange", "UniquenessThreshold"), fname, 2)
    a, b = _disparity_inputs(pos, fname)
    dmin, nd = _disp_range(opts, (0, 128), fname)
    nd16 = int(np.ceil(nd / 16.0) * 16)
    uniq = int(fscalar(opts["UniquenessThreshold"], fname, "UniquenessThreshold")) if "UniquenessThreshold" in opts \
        else 15
    cv2 = cv2mod(fname)
    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=dmin, numDisparities=nd16, blockSize=bs, P1=8 * bs * bs,
                                 P2=32 * bs * bs, uniquenessRatio=uniq, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_HH)
    d = _to_disp(sgbm.compute(a, b), dmin)
    d[d > dmin + nd] = np.nan
    return d


@builtin("disparityBM")
def _disparity_bm(interp, args, nargout):
    fname = "disparityBM"
    pos, opts = options(list(args), ("DisparityRange", "BlockSize", "ContrastThreshold", "UniquenessThreshold",
                                     "DistanceThreshold", "TextureThreshold"), fname, 2)
    a, b = _disparity_inputs(pos, fname)
    dmin, nd = _disp_range(opts, (0, 64), fname)
    nd16 = int(np.ceil(nd / 16.0) * 16)
    bs = int(fscalar(opts["BlockSize"], fname, "BlockSize")) if "BlockSize" in opts else 15
    if bs < 5 or bs % 2 == 0:
        raise MatlabError(f"{fname}: BlockSize must be an odd integer of at least 5.")
    uniq = int(fscalar(opts["UniquenessThreshold"], fname, "UniquenessThreshold")) if "UniquenessThreshold" in opts \
        else 15
    tex = fscalar(opts["TextureThreshold"], fname, "TextureThreshold") if "TextureThreshold" in opts else 0.0002
    cv2 = cv2mod(fname)
    bm = cv2.StereoBM_create(numDisparities=nd16, blockSize=bs)
    bm.setMinDisparity(dmin)
    bm.setUniquenessRatio(uniq)
    bm.setTextureThreshold(int(round(tex * 255 * bs * bs)))
    if "DistanceThreshold" in opts and np.size(opts["DistanceThreshold"]):
        bm.setDisp12MaxDiff(int(fscalar(opts["DistanceThreshold"], fname, "DistanceThreshold")))
    d = _to_disp(bm.compute(a, b), dmin)
    d[d > dmin + nd] = np.nan
    return d


@builtin("reconstructScene")
def _reconstruct_scene(interp, args, nargout):
    fname = "reconstructScene"
    if len(args) != 2:
        raise MatlabError("reconstructScene: expected reconstructScene(disparityMap, reprojectionMatrix).",
                          "MATLAB:minrhs")
    D = np.asarray(args[0], dtype=float)
    if isinstance(args[1], StereoParameters):
        _, _, Q, _, _ = _rectify_maps(args[1], D.shape[:2], 0.0)
        Qm = _q_matlab(Q)
    else:
        Qm = fmat(args[1], fname, "reprojectionMatrix", 4)
    H, W = D.shape
    y, x = np.mgrid[1:H + 1, 1:W + 1].astype(float)
    v = np.stack([x, y, D, np.ones_like(D)], axis=-1) @ Qm
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = v[..., :3] / v[..., 3:4]
    xyz[~np.isfinite(D)] = np.nan
    return xyz.astype(np.float32 if args[0].dtype == np.float32 else float)


# -- undistortion and projection -----------------------------------------------------------------------------

@builtin("undistortImage")
def _undistort_image(interp, args, nargout):
    fname = "undistortImage"
    interp_methods = ("linear", "nearest", "cubic")
    pos, opts = options(list(args), ("OutputView", "FillValues"), fname, 3)
    if len(pos) < 2:
        raise MatlabError("undistortImage: expected undistortImage(I, intrinsics).", "MATLAB:minrhs")
    if len(pos) == 3 and not (is_text(pos[2])):
        raise MatlabError("undistortImage: the interpolation method must be text.")
    cv2 = cv2mod(fname)
    a = image_arg(pos[0], fname)
    intr = intrinsics_of(pos[1], fname)
    method_ = "linear"
    rest = list(args[2:])
    for r in rest:
        if is_text(r) and text(r).lower() in interp_methods:
            method_ = text(r).lower()
    view = opt_text(opts, "OutputView", "same", ("same", "full", "valid"), fname)
    fill = float(np.ravel(opts["FillValues"])[0]) if "FillValues" in opts else 0.0
    H, W = a.shape[:2]
    K0, dist = intr.K0(), intr.dist_cv()
    if view == "same":
        newK, origin, size = K0, np.array([1.0, 1.0]), (W, H)
    else:
        # bounds of the undistorted image border
        bx = np.concatenate([np.arange(W), np.arange(W), np.zeros(H), np.full(H, W - 1)])
        by = np.concatenate([np.zeros(W), np.full(W, H - 1), np.arange(H), np.arange(H)])
        und = intr.undistort_pixels(np.column_stack([bx, by]) + 1.0) - 1.0
        if view == "full":
            x0, y0 = np.floor(und.min(0))
            x1, y1 = np.ceil(und.max(0))
        else:
            x0 = np.ceil(max(und[2 * W:2 * W + H, 0].max(), und[:, 0].min()))
            x1 = np.floor(und[2 * W + H:, 0].min())
            y0 = np.ceil(und[:W, 1].max())
            y1 = np.floor(und[W:2 * W, 1].min())
        newK = K0.copy()
        newK[0, 2] -= x0
        newK[1, 2] -= y0
        size = (int(x1 - x0 + 1), int(y1 - y0 + 1))
        origin = np.array([x0 + 1, y0 + 1])
    flags = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}[method_]
    mx, my = cv2.initUndistortRectifyMap(K0, dist, None, newK, size, cv2.CV_32FC1)
    work = a.astype(np.float32) if a.dtype not in (np.uint8, np.uint16, np.float32) else a
    out = cv2.remap(np.ascontiguousarray(work), mx, my, flags, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=fill if a.dtype.kind == "f" else fill)
    out = out.astype(a.dtype) if a.dtype != bool else out > 0.5
    return (out, row(origin)) if nargout > 1 else out


@builtin("undistortPoints")
def _undistort_points(interp, args, nargout):
    fname = "undistortPoints"
    if len(args) != 2:
        raise MatlabError("undistortPoints: expected undistortPoints(points, intrinsics).", "MATLAB:minrhs")
    p = fmat(args[0], fname, "points", 2)
    intr = intrinsics_of(args[1], fname)
    out = intr.undistort_pixels(p)
    if nargout > 1:
        d = np.sqrt(((intr.to_pixels(intr.to_normalized(out), True) - p) ** 2).sum(1))
        return out, d.reshape(-1, 1)
    return out


def _world_points(v, fname: str) -> np.ndarray:
    W = fmat(v, fname, "worldPoints")
    if W.shape[1] == 2:
        W = np.column_stack([W, np.zeros(len(W))])
    if W.shape[1] != 3:
        raise MatlabError(f"{fname}: worldPoints must be M-by-3 (or M-by-2 points on the Z = 0 plane).")
    return W


def _project(intr: CameraIntrinsics, R: np.ndarray, t: np.ndarray, W: np.ndarray, distort: bool) -> np.ndarray:
    C = W @ R.T + t
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = C[:, :2] / C[:, 2:3]
    return intr.to_pixels(xn, distort)


@builtin("world2img")
def _world2img(interp, args, nargout):
    fname = "world2img"
    pos, opts = options(list(args), ("ApplyDistortion",), fname, 3)
    if len(pos) != 3:
        raise MatlabError("world2img: expected world2img(worldPoints, cameraExtrinsics, intrinsics).",
                          "MATLAB:minrhs")
    W = _world_points(pos[0], fname)
    tf = tform3_arg(pos[1], fname)
    intr = intrinsics_of(pos[2], fname)
    distort = flag(opts["ApplyDistortion"]) if "ApplyDistortion" in opts else False
    return _project(intr, tf.R, tf.t, W, distort)


@builtin("worldToImage")
def _world_to_image(interp, args, nargout):
    fname = "worldToImage"
    pos, opts = options(list(args), ("ApplyDistortion",), fname, 4)
    intr = intrinsics_of(pos[0], fname) if pos else None
    if len(pos) == 3 and isinstance(pos[1], GeoTform3):
        tf = pos[1]
        R, t, W = tf.R, tf.t, _world_points(pos[2], fname)
    elif len(pos) == 4:
        R = fmat(pos[1], fname, "rotationMatrix", 3).T
        t = fvec(pos[2], fname, "translationVector", 3)
        W = _world_points(pos[3], fname)
    else:
        raise MatlabError("worldToImage: expected worldToImage(intrinsics, rotationMatrix, translationVector, "
                          "worldPoints).", "MATLAB:minrhs")
    distort = flag(opts["ApplyDistortion"]) if "ApplyDistortion" in opts else False
    return _project(intr, R, t, W, distort)


def _points_to_world(intr: CameraIntrinsics, args: list, fname: str) -> np.ndarray:
    if len(args) == 2 and isinstance(args[0], GeoTform3):
        R, t, p = args[0].R, args[0].t, fmat(args[1], fname, "imagePoints", 2)
    elif len(args) == 3:
        R = fmat(args[0], fname, "rotationMatrix", 3).T
        t = fvec(args[1], fname, "translationVector", 3)
        p = fmat(args[2], fname, "imagePoints", 2)
    else:
        raise MatlabError(f"{fname}: expected {fname}(intrinsics, tform, imagePoints) or "
                          f"{fname}(intrinsics, R, t, imagePoints).")
    xn = intr.to_normalized(p)
    # intersect each viewing ray with the world plane Z = 0
    Hm = np.column_stack([R[:, 0], R[:, 1], t])
    w = np.column_stack([xn, np.ones(len(xn))]) @ np.linalg.inv(Hm).T
    return w[:, :2] / w[:, 2:3]


@builtin("img2world2d")
def _img2world2d(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("img2world2d: expected img2world2d(imagePoints, cameraExtrinsics, intrinsics).")
    return _points_to_world(intrinsics_of(args[2], "img2world2d"), [args[1], args[0]], "img2world2d")


@builtin("pointsToWorld")
def _points_to_world_b(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    return _points_to_world(intrinsics_of(args[0], "pointsToWorld"), list(args[1:]), "pointsToWorld")


@builtin("cameraProjection")
def _camera_projection(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("cameraProjection: expected cameraProjection(intrinsics, tform).", "MATLAB:minrhs")
    intr = intrinsics_of(args[0], "cameraProjection")
    tf = tform3_arg(args[1], "cameraProjection")
    return intr.K @ np.hstack([tf.R, tf.t.reshape(3, 1)])


@builtin("cameraMatrix")
def _camera_matrix(interp, args, nargout):
    if len(args) == 2:
        return _camera_projection(interp, args, nargout).T
    if len(args) != 3:
        raise MatlabError("cameraMatrix: expected cameraMatrix(intrinsics, rotationMatrix, translationVector).",
                          "MATLAB:minrhs")
    intr = intrinsics_of(args[0], "cameraMatrix")
    R = fmat(args[1], "cameraMatrix", "rotationMatrix", 3)
    t = fvec(args[2], "cameraMatrix", "translationVector", 3)
    return np.vstack([R, t.reshape(1, 3)]) @ intr.K.T


def _pnp(image_pts: np.ndarray, world: np.ndarray, intr: CameraIntrinsics, fname: str):
    cv2 = cv2mod(fname)
    W = _world_points(world, fname)
    p = fmat(image_pts, fname, "imagePoints", 2)
    if p.shape[0] != W.shape[0] or p.shape[0] < 4:
        raise MatlabError(f"{fname}: give at least four image points matching the world points.")
    planar = np.allclose(W[:, 2], 0)
    flags = cv2.SOLVEPNP_IPPE if planar and hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    ok, rv, tv = cv2.solvePnP(W.astype(np.float64), (p - 1.0).astype(np.float64), intr.K0(), intr.dist_cv(),
                              flags=flags)
    if not ok:
        raise MatlabError(f"{fname}: could not compute the camera pose.")
    return rodrigues(np.asarray(rv).ravel()), np.asarray(tv).ravel()


@builtin("estimateExtrinsics")
def _estimate_extrinsics(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("estimateExtrinsics: expected estimateExtrinsics(imagePoints, worldPoints, intrinsics).")
    R, t = _pnp(args[0], args[1], intrinsics_of(args[2], "estimateExtrinsics"), "estimateExtrinsics")
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, t
    return GeoTform3("rigidtform3d", A)


@builtin("extrinsics")
def _extrinsics(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("extrinsics: expected extrinsics(imagePoints, worldPoints, intrinsics).")
    R, t = _pnp(args[0], args[1], intrinsics_of(args[2], "extrinsics"), "extrinsics")
    return R.T, row(t)


@builtin("estworldpose")
def _estworldpose(interp, args, nargout):
    pos, _ = options(list(args), ("MaxNumTrials", "Confidence", "MaxReprojectionError"), "estworldpose", 3)
    if len(pos) != 3:
        raise MatlabError("estworldpose: expected estworldpose(imagePoints, worldPoints, intrinsics).")
    R, t = _pnp(pos[0], pos[1], intrinsics_of(pos[2], "estworldpose"), "estworldpose")
    A = np.eye(4)  # the camera pose in world coordinates (inverse of the extrinsics)
    A[:3, :3], A[:3, 3] = R.T, -R.T @ t
    inl = np.ones((fmat(pos[0], "estworldpose", "imagePoints", 2).shape[0], 1), dtype=bool)
    return GeoTform3("rigidtform3d", A), inl, scalar(0)

