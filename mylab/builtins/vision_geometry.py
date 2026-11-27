"""Multiple-view geometry: robust 2-D transform fitting, fundamental and essential matrices, relative
camera pose, triangulation, and the 3-D rigid/affine transform objects these return.

Image points are M-by-2 [x y] in MATLAB's 1-based pixel coordinates (or point objects). The 2-D
transforms are the rigidtform2d/simtform2d/affinetform2d/projtform2d objects of images_more.py, so
imwarp and transformPointsForward accept them directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import class_name
from ._cv import col, cv2mod, fmat, fscalar, fvec, opt_text, options, row, scalar
from ._registry import builtin
from ._util import is_text, text
from .images_more import GeoTform, make_tform

# -- 3-D transforms ----------------------------------------------------------------------------------------

_NEW3 = ("rigidtform3d", "simtform3d", "affinetform3d", "transltform3d")
_OLD3 = ("rigid3d", "affine3d")


class GeoTform3(MObject):
    """A 3-D geometric transformation; A (4x4) maps column vectors [x; y; z; 1]."""

    _METHODS = ("transformPointsForward", "transformPointsInverse", "invert")

    def __init__(self, kind: str, A: np.ndarray):
        self.kind = kind
        self.A = np.array(A, dtype=float)

    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        return name == self.kind

    @property
    def R(self) -> np.ndarray:
        return self.A[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.A[:3, 3]

    def property_names(self) -> list[str]:
        if self.kind == "rigid3d":
            return ["Rotation", "Translation", "T", "Dimensionality"]
        if self.kind == "affine3d":
            return ["T", "Dimensionality"]
        if self.kind == "rigidtform3d":
            return ["Dimensionality", "R", "Translation", "A"]
        if self.kind == "simtform3d":
            return ["Dimensionality", "Scale", "R", "Translation", "A"]
        if self.kind == "transltform3d":
            return ["Dimensionality", "Translation", "A"]
        return ["Dimensionality", "A"]

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def forward(self, p: np.ndarray) -> np.ndarray:
        h = np.column_stack([p, np.ones(p.shape[0])]) @ self.A.T
        return h[:, :3] / h[:, 3:4]

    def inverse(self, p: np.ndarray) -> np.ndarray:
        h = np.column_stack([p, np.ones(p.shape[0])]) @ np.linalg.inv(self.A).T
        return h[:, :3] / h[:, 3:4]

    def scale(self) -> float:
        return float(np.cbrt(abs(np.linalg.det(self.R))))

    def get_field(self, interp, name: str) -> Any:
        if name == "Dimensionality":
            return scalar(3)
        if name not in self.property_names():
            if name in self._METHODS:
                return self.call_method(interp, name, [self], 1)[0]
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.",
                              "MATLAB:noSuchMethodOrField")
        if name == "A":
            return self.A.copy()
        if name == "T":
            return self.A.T.copy()
        if name == "Translation":
            return row(self.t)
        if name == "Scale":
            return scalar(self.scale())
        if name == "R":
            return self.R / (self.scale() if self.kind == "simtform3d" else 1.0)
        if name == "Rotation":  # rigid3d keeps the transposed (row-vector) rotation
            return self.R.T.copy()
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")

    def set_field(self, interp, name: str, value: Any) -> "GeoTform3":
        A = self.A.copy()
        if name == "Translation":
            A[:3, 3] = fvec(value, self.kind, "Translation", 3)
        elif name == "R" and self.kind in ("rigidtform3d", "simtform3d"):
            A[:3, :3] = fmat(value, self.kind, "R", 3) * (self.scale() if self.kind == "simtform3d" else 1.0)
        elif name == "Rotation" and self.kind == "rigid3d":
            A[:3, :3] = fmat(value, self.kind, "Rotation", 3).T
        elif name == "A" and self.kind in _NEW3:
            A = fmat(value, self.kind, "A", 4)
        elif name == "T" and self.kind in _OLD3:
            A = fmat(value, self.kind, "T", 4).T
        else:
            raise MatlabError(f"Cannot set property '{name}' of class '{self.kind}'.")
        return make_tform3(self.kind, A)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "invert":
            inv = np.linalg.inv(self.A)
            return [GeoTform3(self.kind, inv / inv[3, 3])]
        fn = self.forward if name == "transformPointsForward" else self.inverse
        if len(rest) == 1:
            return [fn(fmat(rest[0], name, "points", 3))]
        if len(rest) == 3:
            u = [np.asarray(r, dtype=float) for r in rest]
            out = fn(np.column_stack([a.ravel(order="F") for a in u]))
            return [out[:, k].reshape(u[0].shape, order="F") for k in range(3)]
        raise MatlabError(f"{name}: expected (tform, points) with points M-by-3, or (tform, x, y, z).")

    def display_lines(self) -> list[str]:
        from ..display import summary
        out = [f"  {self.kind} with properties:", ""]
        for n in self.property_names():
            v = self.get_field(None, n)
            if v.size > 3:
                out.append(f"    {n}: ")
                out += ["        " + "  ".join(f"{x:10.4f}" for x in r) for r in v]
            else:
                out.append(f"    {n}: {summary(v, False)}")
        return out


def make_tform3(kind: str, A: np.ndarray) -> GeoTform3:
    A = np.asarray(A, dtype=float)
    if A.shape != (4, 4):
        raise MatlabError(f"{kind}: the transformation matrix must be 4-by-4.")
    if kind in ("rigidtform3d", "rigid3d"):
        R = A[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
            raise MatlabError(f"{kind}: the rotation part must be a rotation matrix (orthonormal, det = 1).",
                              f"mylab:{kind}:invalidRotation")
    if not np.allclose(A[3], [0, 0, 0, 1], atol=1e-8):
        raise MatlabError(f"{kind}: the last row of the matrix must be [0 0 0 1].")
    return GeoTform3(kind, A)


def _rot_from_euler_deg(e: np.ndarray) -> np.ndarray:
    """Rotation from [rx ry rz] angles in degrees, applied about x, then y, then z (R = Rz*Ry*Rx)."""
    rx, ry, rz = np.deg2rad(e)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tform3_arg(v: Any, fname: str) -> GeoTform3:
    if isinstance(v, GeoTform3):
        return v
    raise MatlabError(f"{fname}: expected a 3-D transformation (rigidtform3d, affinetform3d, rigid3d or affine3d), "
                      f"not a value of class '{class_name(v)}'.", f"mylab:{fname}:invalidTform")


@builtin("rigidtform3d")
def _rigidtform3d(interp, args, nargout):
    if not args:
        return GeoTform3("rigidtform3d", np.eye(4))
    if len(args) == 1:
        if isinstance(args[0], GeoTform3):
            return make_tform3("rigidtform3d", args[0].A)
        return make_tform3("rigidtform3d", fmat(args[0], "rigidtform3d", "A", 4))
    if len(args) == 2:
        r = np.asarray(args[0], dtype=float)
        A = np.eye(4)
        A[:3, :3] = _rot_from_euler_deg(r.ravel()) if r.size == 3 else fmat(r, "rigidtform3d", "R", 3)
        A[:3, 3] = fvec(args[1], "rigidtform3d", "translation", 3)
        return make_tform3("rigidtform3d", A)
    raise MatlabError("rigidtform3d: expected rigidtform3d(), rigidtform3d(A), rigidtform3d(R, t) or "
                      "rigidtform3d(eulerAnglesDeg, t).")


@builtin("simtform3d")
def _simtform3d(interp, args, nargout):
    if not args:
        return GeoTform3("simtform3d", np.eye(4))
    if len(args) == 1:
        return make_tform3("simtform3d", fmat(args[0], "simtform3d", "A", 4))
    if len(args) == 3:
        s = fscalar(args[0], "simtform3d", "the scale")
        r = np.asarray(args[1], dtype=float)
        A = np.eye(4)
        A[:3, :3] = s * (_rot_from_euler_deg(r.ravel()) if r.size == 3 else fmat(r, "simtform3d", "R", 3))
        A[:3, 3] = fvec(args[2], "simtform3d", "translation", 3)
        return make_tform3("simtform3d", A)
    raise MatlabError("simtform3d: expected simtform3d(A) or simtform3d(scale, R, t).")


@builtin("affinetform3d")
def _affinetform3d(interp, args, nargout):
    if not args:
        return GeoTform3("affinetform3d", np.eye(4))
    a = args[0].A if isinstance(args[0], GeoTform3) else fmat(args[0], "affinetform3d", "A", 4)
    return make_tform3("affinetform3d", a)


@builtin("transltform3d")
def _transltform3d(interp, args, nargout):
    A = np.eye(4)
    if args:
        t = np.concatenate([fvec(a, "transltform3d", "translation") for a in args])
        if t.size != 3:
            raise MatlabError("transltform3d: the translation must have three elements.")
        A[:3, 3] = t
    return GeoTform3("transltform3d", A)


@builtin("rigid3d")
def _rigid3d(interp, args, nargout):
    if not args:
        return GeoTform3("rigid3d", np.eye(4))
    if len(args) == 1:
        return make_tform3("rigid3d", fmat(args[0], "rigid3d", "T", 4).T)
    A = np.eye(4)
    A[:3, :3] = fmat(args[0], "rigid3d", "rotation", 3).T
    A[:3, 3] = fvec(args[1], "rigid3d", "translation", 3)
    return make_tform3("rigid3d", A)


@builtin("affine3d")
def _affine3d(interp, args, nargout):
    if not args:
        return GeoTform3("affine3d", np.eye(4))
    return make_tform3("affine3d", fmat(args[0], "affine3d", "T", 4).T)


def rodrigues(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    th = np.linalg.norm(v)
    if th < 1e-15:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rodrigues_inv(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        # near 180 degrees: the axis is the eigenvector of R with eigenvalue 1
        w, v = np.linalg.eigh((R + R.T) / 2)
        k = v[:, np.argmax(w)]
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


@builtin("rotvec2mat3d")
def _rotvec2mat3d(interp, args, nargout):
    v = fmat(args[0], "rotvec2mat3d", "rotation vector")
    if v.shape[1] != 3:
        raise MatlabError("rotvec2mat3d: the rotation vectors must be N-by-3.")
    mats = [rodrigues(r) for r in v]
    return mats[0] if len(mats) == 1 else np.stack(mats, axis=2)


@builtin("rotmat2vec3d")
def _rotmat2vec3d(interp, args, nargout):
    R = np.asarray(args[0], dtype=float)
    if R.shape[:2] != (3, 3):
        raise MatlabError("rotmat2vec3d: the input must be 3-by-3 (or 3-by-3-by-N).")
    R = R.reshape(3, 3, -1)
    return np.vstack([rodrigues_inv(R[:, :, k]) for k in range(R.shape[2])])


@builtin("rotationVectorToMatrix")
def _rotationvectortomatrix(interp, args, nargout):
    # the older function returns the transposed (row-vector convention) matrix
    return rodrigues(fvec(args[0], "rotationVectorToMatrix", "the rotation vector", 3)).T


@builtin("rotationMatrixToVector")
def _rotationmatrixtovector(interp, args, nargout):
    return row(rodrigues_inv(fmat(args[0], "rotationMatrixToVector", "rotation matrix", 3).T))


# -- robust fitting of 2-D transforms ----------------------------------------------------------------------

def _similarity_fit(p: np.ndarray, q: np.ndarray, scale: bool) -> np.ndarray:
    """Least-squares rotation (+ uniform scale) + translation taking p to q (Umeyama)."""
    mp, mq = p.mean(0), q.mean(0)
    P, Q = p - mp, q - mq
    U, S, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max((P ** 2).sum(), 1e-300) if scale else 1.0
    A = np.eye(3)
    A[:2, :2] = s * R
    A[:2, 2] = mq - s * R @ mp
    return A


def _affine_fit(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    X = np.column_stack([p, np.ones(p.shape[0])])
    M, *_ = np.linalg.lstsq(X, q, rcond=None)
    A = np.eye(3)
    A[:2, :] = M.T
    return A


def _normalize(p: np.ndarray) -> np.ndarray:
    m = p.mean(0)
    d = np.sqrt(((p - m) ** 2).sum(1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])


def _projective_fit(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    Tp, Tq = _normalize(p), _normalize(q)
    ph = np.column_stack([p, np.ones(len(p))]) @ Tp.T
    qh = np.column_stack([q, np.ones(len(q))]) @ Tq.T
    rows = []
    for (x, y, _), (u, v, _) in zip(ph, qh):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tq) @ H @ Tp
    return H / H[2, 2] if abs(H[2, 2]) > 1e-15 else H


_MIN_POINTS = {"rigid": 2, "similarity": 2, "affine": 3, "projective": 4}


def _fit(kind: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if kind == "rigid":
        return _similarity_fit(p, q, False)
    if kind == "similarity":
        return _similarity_fit(p, q, True)
    if kind == "affine":
        return _affine_fit(p, q)
    return _projective_fit(p, q)


def _errors(A: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    h = np.column_stack([p, np.ones(len(p))]) @ A.T
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = h[:, :2] / h[:, 2:3]
    d = np.sqrt(((proj - q) ** 2).sum(1))
    return np.where(np.isfinite(d), d, np.inf)


def _degenerate(kind: str, p: np.ndarray) -> bool:
    if kind in ("rigid", "similarity"):
        return np.linalg.norm(p[0] - p[1]) < 1e-9
    a = np.column_stack([p, np.ones(len(p))])
    if kind == "affine":
        return abs(np.linalg.det(a)) < 1e-9
    for i in range(4):  # projective: no three of the four collinear
        sub = np.delete(a, i, axis=0)
        if abs(np.linalg.det(sub)) < 1e-9:
            return True
    return False


def msac(kind: str, p: np.ndarray, q: np.ndarray, trials: int, conf: float, maxdist: float, rng=None):
    """MSAC: random minimal samples, truncated quadratic cost, adaptive trial count; refit on inliers."""
    n = p.shape[0]
    k = _MIN_POINTS[kind]
    rng = rng if rng is not None else np.random.default_rng(0)
    best_cost, best_A = np.inf, None
    t2 = maxdist ** 2
    needed = trials
    it = 0
    while it < min(needed, trials):
        it += 1
        idx = rng.choice(n, k, replace=False)
        if _degenerate(kind, p[idx]):
            continue
        A = _fit(kind, p[idx], q[idx])
        if not np.all(np.isfinite(A)) or abs(np.linalg.det(A)) < 1e-12:
            continue
        d2 = _errors(A, p, q) ** 2
        cost = np.minimum(d2, t2).sum()
        if cost < best_cost:
            best_cost, best_A = cost, A
            ratio = (d2 < t2).mean()
            if ratio > 0:
                denom = np.log(max(1 - ratio ** k, 1e-12))
                needed = int(np.ceil(np.log(max(1 - conf / 100.0, 1e-12)) / denom)) if denom < 0 else trials
    if best_A is None:
        return None, np.zeros(n, dtype=bool)
    inl = _errors(best_A, p, q) < maxdist
    for _ in range(3):  # refine on the inliers
        if inl.sum() < k:
            break
        A = _fit(kind, p[inl], q[inl])
        new = _errors(A, p, q) < maxdist
        if new.sum() < inl.sum():
            break
        best_A, inl = A, new
    return best_A, inl


_KINDS2D = ("rigid", "similarity", "affine", "projective")


def _global_rng(interp):
    from .core import _rng
    return _rng(interp) if interp is not None else None


def _estimate_2d(interp, args, nargout, fname: str, new_classes: bool, kinds=_KINDS2D):
    pos, opts = options(list(args[:2]) + list(args[3:]) if len(args) > 2 and is_text(args[2]) else list(args),
                        ("MaxNumTrials", "Confidence", "MaxDistance"), fname, 2)
    if len(pos) != 2 or len(args) < 3 or not is_text(args[2]):
        raise MatlabError(f"{fname}: expected {fname}(matchedPoints1, matchedPoints2, transformType).",
                          "MATLAB:minrhs")
    kind = text(args[2]).lower()
    if kind not in kinds:
        raise MatlabError(f"{fname}: the transformation type must be " + ", ".join(f"'{k}'" for k in kinds) + ".",
                          f"mylab:{fname}:invalidTransformType")
    p = fmat(pos[0], fname, "matchedPoints1", 2)
    q = fmat(pos[1], fname, "matchedPoints2", 2)
    if p.shape != q.shape:
        raise MatlabError(f"{fname}: matchedPoints1 and matchedPoints2 must have the same number of points.",
                          f"mylab:{fname}:sizeMismatch")
    trials = int(fscalar(opts["MaxNumTrials"], fname, "MaxNumTrials")) if "MaxNumTrials" in opts else 1000
    conf = fscalar(opts["Confidence"], fname, "Confidence") if "Confidence" in opts else 99.0
    maxd = fscalar(opts["MaxDistance"], fname, "MaxDistance") if "MaxDistance" in opts else 1.5
    if trials < 1 or not 0 < conf < 100 or maxd <= 0:
        raise MatlabError(f"{fname}: MaxNumTrials must be positive, Confidence in (0, 100) and MaxDistance positive.")
    need = _MIN_POINTS[kind]
    status = 0
    A, inl = None, np.zeros(p.shape[0], dtype=bool)
    if p.shape[0] < need:
        status = 1
    else:
        A, inl = msac(kind, p, q, trials, conf, maxd, _global_rng(interp))
        if A is None or inl.sum() < need:
            status = 2
    want_status = nargout >= (4 if fname == "estimateGeometricTransform" else 3)
    if status and not want_status:
        msg = {1: f"{fname}: at least {need} matched point pairs are needed for a {kind} transformation.",
               2: f"{fname}: could not find enough inliers; try a larger MaxDistance or better matches."}[status]
        raise MatlabError(msg, f"mylab:{fname}:{'notEnoughPts' if status == 1 else 'notEnoughInliers'}")
    if status:
        A = np.eye(3)
    if new_classes:
        cls = {"rigid": "rigidtform2d", "similarity": "simtform2d", "affine": "affinetform2d",
               "projective": "projtform2d"}[kind]
        if kind == "rigid":
            A = _clean_rigid(A)
        tf = make_tform(cls, A)
    else:
        tf = make_tform("projective2d" if kind == "projective" else "affine2d", A)
    if fname == "estimateGeometricTransform":
        return tf, p[inl], q[inl], scalar(status)
    return tf, inl.reshape(-1, 1), scalar(status)


def _clean_rigid(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A[:2, :2])
    out = A.copy()
    out[:2, :2] = U @ Vt
    return out


@builtin("estgeotform2d")
def _estgeotform2d(interp, args, nargout):
    return _estimate_2d(interp, args, nargout, "estgeotform2d", True)


@builtin("estimateGeometricTransform2D")
def _estimate_gt2d(interp, args, nargout):
    return _estimate_2d(interp, args, nargout, "estimateGeometricTransform2D", False)


@builtin("estimateGeometricTransform")
def _estimate_gt(interp, args, nargout):
    return _estimate_2d(interp, args, nargout, "estimateGeometricTransform", False,
                        ("similarity", "affine", "projective"))


# -- fundamental and essential matrices -------------------------------------------------------------------

def _pairs(pos: list, fname: str) -> tuple[np.ndarray, np.ndarray]:
    p = fmat(pos[0], fname, "matchedPoints1", 2)
    q = fmat(pos[1], fname, "matchedPoints2", 2)
    if p.shape != q.shape:
        raise MatlabError(f"{fname}: matchedPoints1 and matchedPoints2 must have the same number of points.",
                          f"mylab:{fname}:sizeMismatch")
    return p, q


def _eight_point(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized 8-point estimate of F with q^T F p = 0, rank 2 enforced."""
    Tp, Tq = _normalize(p), _normalize(q)
    ph = np.column_stack([p, np.ones(len(p))]) @ Tp.T
    qh = np.column_stack([q, np.ones(len(q))]) @ Tq.T
    A = np.einsum("ni,nj->nij", qh, ph).reshape(len(p), 9)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0]) @ Vt
    F = Tq.T @ F @ Tp
    return F / np.linalg.norm(F)


def sampson(F: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ph = np.column_stack([p, np.ones(len(p))])
    qh = np.column_stack([q, np.ones(len(q))])
    Fp = ph @ F.T
    Ftq = qh @ F
    num = np.sum(qh * Fp, axis=1) ** 2
    den = Fp[:, 0] ** 2 + Fp[:, 1] ** 2 + Ftq[:, 0] ** 2 + Ftq[:, 1] ** 2
    return num / np.maximum(den, 1e-300)


def _algebraic(F: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ph = np.column_stack([p, np.ones(len(p))])
    qh = np.column_stack([q, np.ones(len(q))])
    return np.abs(np.sum(qh * (ph @ F.T), axis=1))


@builtin("estimateFundamentalMatrix")
def _estimate_fundamental(interp, args, nargout):
    fname = "estimateFundamentalMatrix"
    pos, opts = options(list(args), ("Method", "OutputClass", "NumTrials", "DistanceType", "DistanceThreshold",
                                     "Confidence", "InlierPercentage", "ReportRuntimeError"), fname, 2)
    if len(pos) != 2:
        raise MatlabError(f"{fname}: expected {fname}(matchedPoints1, matchedPoints2, ...).", "MATLAB:minrhs")
    p, q = _pairs(pos, fname)
    method = opt_text(opts, "Method", "RANSAC", ("Norm8Point", "LMedS", "RANSAC", "MSAC", "LTS"), fname)
    dtype = opt_text(opts, "DistanceType", "Sampson", ("Sampson", "Algebraic"), fname)
    trials = int(fscalar(opts["NumTrials"], fname, "NumTrials")) if "NumTrials" in opts else 500
    th = fscalar(opts["DistanceThreshold"], fname, "DistanceThreshold") if "DistanceThreshold" in opts else 0.01
    conf = fscalar(opts["Confidence"], fname, "Confidence") if "Confidence" in opts else 99.0
    pct = fscalar(opts["InlierPercentage"], fname, "InlierPercentage") if "InlierPercentage" in opts else 50.0
    report = bool(np.ravel(opts["ReportRuntimeError"])[0]) if "ReportRuntimeError" in opts else True
    single = "OutputClass" in opts and text(opts["OutputClass"]).lower() == "single"
    n = p.shape[0]
    dist = sampson if dtype == "Sampson" else _algebraic
    status = 0
    F = np.zeros((3, 3))
    inl = np.zeros(n, dtype=bool)
    if n < 8:
        status = 1
    elif method == "Norm8Point":
        F = _eight_point(p, q)
        inl[:] = True
    else:
        rng = _global_rng(interp)
        best = (np.inf, None)
        needed = trials
        it = 0
        while it < min(needed, trials):
            it += 1
            idx = rng.choice(n, 8, replace=False)
            Fc = _eight_point(p[idx], q[idx])
            d = dist(Fc, p, q)
            if method in ("RANSAC",):
                cost = -(d < th).sum()
            elif method == "MSAC":
                cost = np.minimum(d, th).sum()
            elif method == "LMedS":
                cost = np.median(d)
            else:  # LTS: sum of the smallest InlierPercentage of the distances
                m = max(8, int(np.floor(n * pct / 100)))
                cost = np.sort(d)[:m].sum()
            if cost < best[0]:
                best = (cost, Fc)
                if method in ("RANSAC", "MSAC"):
                    ratio = (d < th).mean()
                    denom = np.log(max(1 - ratio ** 8, 1e-12)) if ratio > 0 else 0
                    if denom < 0:
                        needed = int(np.ceil(np.log(max(1 - conf / 100.0, 1e-12)) / denom))
        F = best[1]
        d = dist(F, p, q)
        if method in ("RANSAC", "MSAC"):
            inl = d < th
        elif method == "LMedS":
            inl = d <= np.median(d)
        else:
            m = max(8, int(np.floor(n * pct / 100)))
            inl = np.zeros(n, dtype=bool)
            inl[np.argsort(d)[:m]] = True
        if inl.sum() >= 8:
            F = _eight_point(p[inl], q[inl])
            if method in ("RANSAC", "MSAC"):
                inl = dist(F, p, q) < th
        else:
            status = 2
    if status and (report or nargout < 3):
        raise MatlabError(f"{fname}: " + ("at least 8 matched point pairs are needed." if status == 1 else
                                          "not enough inliers were found."),
                          f"mylab:{fname}:{'notEnoughPts' if status == 1 else 'notEnoughInliers'}")
    if F[2, 2] < 0:
        F = -F
    F = F.astype(np.float32) if single else F
    return F, inl.reshape(-1, 1), scalar(status)


def _intrinsics_K(v: Any, fname: str) -> np.ndarray:
    from .vision_calib import intrinsics_of
    return intrinsics_of(v, fname).K


def _normalized(p: np.ndarray, intr) -> np.ndarray:
    """Pixel points (1-based) to normalized camera coordinates, removing lens distortion."""
    from .vision_calib import undistort_normalized
    return undistort_normalized(p, intr)


def five_point_ransac(x1: np.ndarray, x2: np.ndarray, thresh: float, trials: int, conf: float):
    """Essential matrix from normalized coordinates with OpenCV's five-point RANSAC solver."""
    cv2 = cv2mod("estimateEssentialMatrix")
    E, mask = cv2.findEssentialMat(x1.astype(np.float64), x2.astype(np.float64), np.eye(3), method=cv2.RANSAC,
                                   prob=conf / 100.0, threshold=thresh, maxIters=trials)
    if E is None or E.shape[0] < 3:
        return None, np.zeros(len(x1), dtype=bool)
    E = E[:3]
    return E / np.linalg.norm(E), mask.ravel().astype(bool)


@builtin("estimateEssentialMatrix")
def _estimate_essential(interp, args, nargout):
    fname = "estimateEssentialMatrix"
    pos, opts = options(list(args), ("MaxNumTrials", "Confidence", "MaxDistance"), fname, 4)
    if len(pos) < 3:
        raise MatlabError(f"{fname}: expected {fname}(matchedPoints1, matchedPoints2, intrinsics).", "MATLAB:minrhs")
    from .vision_calib import intrinsics_of
    p, q = _pairs(pos, fname)
    i1 = intrinsics_of(pos[2], fname)
    i2 = intrinsics_of(pos[3], fname) if len(pos) > 3 else i1
    trials = int(fscalar(opts["MaxNumTrials"], fname, "MaxNumTrials")) if "MaxNumTrials" in opts else 500
    conf = fscalar(opts["Confidence"], fname, "Confidence") if "Confidence" in opts else 99.0
    maxd = fscalar(opts["MaxDistance"], fname, "MaxDistance") if "MaxDistance" in opts else 0.1
    status = 0
    E, inl = np.zeros((3, 3)), np.zeros(p.shape[0], dtype=bool)
    if p.shape[0] < 5:
        status = 1
    else:
        x1, x2 = _normalized(p, i1), _normalized(q, i2)
        f = (np.mean(i1.focal) + np.mean(i2.focal)) / 2
        E, inl = five_point_ransac(x1, x2, np.sqrt(maxd) / f, trials, conf)
        if E is None or inl.sum() < 5:
            status = 2
            E = np.zeros((3, 3))
    if status and nargout < 3:
        raise MatlabError(f"{fname}: " + ("at least 5 matched point pairs are needed." if status == 1 else
                                          "not enough inliers were found."),
                          f"mylab:{fname}:{'notEnoughPts' if status == 1 else 'notEnoughInliers'}")
    return E, inl.reshape(-1, 1), scalar(status)


# -- relative pose -----------------------------------------------------------------------------------------

def _triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    out = np.zeros((x1.shape[0], 3))
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(x1, x2)):
        A = np.vstack([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3] if abs(X[3]) > 1e-300 else np.full(3, np.inf)
    return out


def pose_from_essential(E: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """(R, t, fraction): camera 2 sees X2 = R X1 + t; picks the decomposition with most points in front."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    best = None
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            P2 = np.hstack([R, t.reshape(3, 1)])
            X = _triangulate_dlt(P1, P2, x1, x2)
            z1 = X[:, 2]
            z2 = (X @ R.T + t)[:, 2]
            good = np.isfinite(z1) & (z1 > 0) & (z2 > 0)
            frac = good.mean() if len(good) else 0.0
            if best is None or frac > best[2]:
                best = (R, t, frac)
    return best


def _pose_inputs(pos: list, fname: str):
    from .vision_calib import intrinsics_of
    M = fmat(pos[0], fname, "essential or fundamental matrix", 3)
    i1 = intrinsics_of(pos[1], fname)
    if len(pos) == 5:
        i2, a, b = intrinsics_of(pos[2], fname), pos[3], pos[4]
    elif len(pos) == 4:
        i2, a, b = i1, pos[2], pos[3]
    else:
        raise MatlabError(f"{fname}: expected {fname}(E, intrinsics, inlierPoints1, inlierPoints2).",
                          "MATLAB:minrhs")
    p, q = _pairs([a, b], fname)
    return M, i1, i2, p, q


def _to_essential(M: np.ndarray, i1, i2, fname: str, is_f: bool) -> np.ndarray:
    if is_f:
        M = i2.K.T @ M @ i1.K
    U, S, Vt = np.linalg.svd(M)
    s = (S[0] + S[1]) / 2
    return U @ np.diag([s, s, 0]) @ Vt


def _relpose(pos, fname: str, is_f: bool | None):
    M, i1, i2, p, q = _pose_inputs(pos, fname)
    if is_f is None:  # decide: an essential matrix has two equal singular values
        S = np.linalg.svd(M, compute_uv=False)
        is_f = not (S[0] > 0 and abs(S[0] - S[1]) / S[0] < 1e-2)
    E = _to_essential(M, i1, i2, fname, is_f)
    x1, x2 = _normalized(p, i1), _normalized(q, i2)
    R, t, frac = pose_from_essential(E, x1, x2)
    # pose of camera 2 in camera-1 coordinates: X1 = Rp X2 + tp
    Rp = R.T
    tp = -R.T @ t
    tp = tp / max(np.linalg.norm(tp), 1e-300)
    return Rp, tp, frac


@builtin("estrelpose")
def _estrelpose(interp, args, nargout):
    pos, _ = options(list(args), (), "estrelpose", 5)
    Rp, tp, frac = _relpose(pos, "estrelpose", None)
    A = np.eye(4)
    A[:3, :3] = Rp
    A[:3, 3] = tp
    return GeoTform3("rigidtform3d", A), scalar(frac)


@builtin("relativeCameraPose")
def _relative_camera_pose(interp, args, nargout):
    pos, _ = options(list(args), (), "relativeCameraPose", 5)
    Rp, tp, frac = _relpose(pos, "relativeCameraPose", None)
    if nargout <= 1:
        A = np.eye(4)
        A[:3, :3] = Rp
        A[:3, 3] = tp
        return GeoTform3("rigidtform3d", A)
    # older form: orientation in the row-vector convention, location as a row
    return Rp.T, row(tp), scalar(frac)


# -- triangulation -----------------------------------------------------------------------------------------

def _camera_matrix(v: Any, fname: str) -> np.ndarray:
    """A 3-by-4 projection matrix P with [x; y; 1] ~ P [X; Y; Z; 1] (4-by-3 matrices are transposed)."""
    if hasattr(v, "stereo_cam1"):
        raise MatlabError(f"{fname}: pass the two camera matrices, or use triangulate(p1, p2, stereoParams).")
    M = fmat(v, fname, "camera matrix")
    if M.shape == (4, 3):
        return M.T
    if M.shape == (3, 4):
        return M
    raise MatlabError(f"{fname}: a camera matrix must be 3-by-4 (camera projection) or 4-by-3.",
                      f"mylab:{fname}:invalidCameraMatrix")


@builtin("triangulate")
def _triangulate(interp, args, nargout):
    fname = "triangulate"
    if len(args) not in (3, 4):
        raise MatlabError("triangulate: expected triangulate(points1, points2, cameraMatrix1, cameraMatrix2) "
                          "or triangulate(points1, points2, stereoParams).", "MATLAB:minrhs")
    p, q = _pairs(list(args[:2]), fname)
    if len(args) == 3:
        sp = args[2]
        if not hasattr(sp, "stereo_cam1"):
            raise MatlabError("triangulate: with three inputs the third must be a stereoParameters object.")
        P1, P2 = sp.projection_matrices()
        p = sp.cam1.undistort_pixels(p)
        q = sp.cam2.undistort_pixels(q)
    else:
        P1, P2 = _camera_matrix(args[2], fname), _camera_matrix(args[3], fname)
    X = _triangulate_dlt(P1, P2, p, q)
    errs = np.zeros(p.shape[0])
    valid = np.ones(p.shape[0], dtype=bool)
    Xh = np.column_stack([X, np.ones(len(X))])
    for P, x in ((P1, p), (P2, q)):
        h = Xh @ P.T
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = h[:, :2] / h[:, 2:3]
        errs += np.sqrt(((proj - x) ** 2).sum(1))
        # in front of the camera: depth has the sign of det(M) (M = left 3x3 of P)
        valid &= h[:, 2] * np.sign(np.linalg.det(P[:, :3])) > 0
    return X, col(errs / 2), valid.reshape(-1, 1)


@builtin("epipolarLine")
def _epipolar_line(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("epipolarLine: expected epipolarLine(F, points).")
    F = fmat(args[0], "epipolarLine", "F", 3)
    p = fmat(args[1], "epipolarLine", "points", 2)
    lines = np.column_stack([p, np.ones(len(p))]) @ F.T
    return lines


@builtin("isEpipoleInImage")
def _is_epipole_in_image(interp, args, nargout):
    F = fmat(args[0], "isEpipoleInImage", "F", 3)
    sz = fvec(args[1], "isEpipoleInImage", "imageSize")
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]
    if abs(e[2]) < 1e-12:
        return (np.array([[False]]), np.full((1, 2), np.inf)) if nargout > 1 else np.array([[False]])
    e = e[:2] / e[2]
    inside = 0.5 <= e[0] <= sz[1] + 0.5 and 0.5 <= e[1] <= sz[0] + 0.5
    return (np.array([[inside]]), row(e)) if nargout > 1 else np.array([[inside]])


def is_tform2(v) -> bool:
    return isinstance(v, GeoTform)
