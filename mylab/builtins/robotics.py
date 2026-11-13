"""Rotations and rigid transforms (Euler angles, quaternions, axis-angle, 4x4 transforms), angle helpers,
and the small set of code-generation directives that ordinary code can simply ignore."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..values import char, is_char, to_str
from ._registry import builtin
from ._util import fnum, is_text, nargs, text


def _wrap(a: np.ndarray) -> np.ndarray:
    """Angles wrapped to [-pi, pi]; pi stays pi for positive inputs."""
    w = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return np.where((w == -np.pi) & (a > 0), np.pi, w)


@builtin("angdiff")
def _angdiff(interp, args, nargout):
    nargs(args, 1, 2, "angdiff")
    a = fnum(args[0])
    if len(args) == 1:
        v = a.reshape(-1)
        d = _wrap(np.diff(v))
        return d.reshape(1, -1) if a.shape[0] == 1 else d.reshape(-1, 1)
    try:
        return _wrap(fnum(args[1]) - a)
    except ValueError:
        raise MatlabError("angdiff: the two inputs must have the same size (or one must be a scalar).") from None


def _seq(args: list, k: int, name: str) -> str:
    s = text(args[k]).upper() if len(args) > k else "ZYX"
    if s not in ("ZYX", "ZYZ", "XYZ"):
        raise MatlabError(f"{name}: the rotation sequence must be 'ZYX', 'ZYZ' or 'XYZ'.")
    return s


def _axis(k: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if k == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if k == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rows(v, n: int, name: str, what: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim != 2 or (a.shape[1] != n and a.size != n):
        raise MatlabError(f"{name}: {what} must have {n} columns (one row per rotation).")
    return a.reshape(-1, n) if a.shape[1] == n else a.reshape(1, n)


def _mats(v, n: int, name: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[:2] != (n, n):
        raise MatlabError(f"{name}: the input must be {n}-by-{n} (or {n}-by-{n}-by-N).")
    return np.moveaxis(a, 2, 0)


def _stack(mats: list) -> np.ndarray:
    return mats[0] if len(mats) == 1 else np.stack(mats, axis=2)


def _eul2rotm(e: np.ndarray, seq: str) -> np.ndarray:
    return _axis(seq[0], e[0]) @ _axis(seq[1], e[1]) @ _axis(seq[2], e[2])


def _rotm2eul(r: np.ndarray, seq: str) -> np.ndarray:
    if seq == "ZYX":
        sy = np.hypot(r[0, 0], r[1, 0])
        if sy > 1e-10:
            return np.array([np.arctan2(r[1, 0], r[0, 0]), np.arctan2(-r[2, 0], sy), np.arctan2(r[2, 1], r[2, 2])])
        return np.array([0.0, np.arctan2(-r[2, 0], sy), np.arctan2(-r[1, 2], r[1, 1])])
    if seq == "XYZ":
        cy = np.hypot(r[0, 0], r[0, 1])
        if cy > 1e-10:
            return np.array([np.arctan2(-r[1, 2], r[2, 2]), np.arctan2(r[0, 2], cy), np.arctan2(-r[0, 1], r[0, 0])])
        return np.array([np.arctan2(r[2, 1], r[1, 1]), np.arctan2(r[0, 2], cy), 0.0])
    sy = np.hypot(r[0, 2], r[1, 2])  # ZYZ
    if sy > 1e-10:
        return np.array([np.arctan2(r[1, 2], r[0, 2]), np.arctan2(sy, r[2, 2]), np.arctan2(r[2, 1], -r[2, 0])])
    return np.array([0.0, np.arctan2(sy, r[2, 2]), np.arctan2(-r[0, 1], r[1, 1])])


def _quat2rotm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        raise MatlabError("quat2rotm: a quaternion must not be all zeros.")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotm2quat(r: np.ndarray) -> np.ndarray:
    # the largest of the four candidates keeps the division well conditioned
    k = np.array(
        [r[0, 0] + r[1, 1] + r[2, 2], r[0, 0] - r[1, 1] - r[2, 2], -r[0, 0] + r[1, 1] - r[2, 2], -r[0, 0] - r[1, 1] + r[2, 2]]
    )
    i = int(np.argmax(k))
    s = 2 * np.sqrt(max(1 + k[i], 0.0))
    if i == 0:
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif i == 1:
        q = [(r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif i == 2:
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s]
    else:
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4]
    q = np.array(q)
    return -q if q[0] < 0 else q


def _tform(r: np.ndarray, t=None) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    if t is not None:
        m[:3, 3] = t
    return m


@builtin("eul2rotm")
def _b_eul2rotm(interp, args, nargout):
    nargs(args, 1, 2, "eul2rotm")
    seq = _seq(args, 1, "eul2rotm")
    return _stack([_eul2rotm(e, seq) for e in _rows(args[0], 3, "eul2rotm", "the Euler angles")])


@builtin("eul2tform")
def _b_eul2tform(interp, args, nargout):
    nargs(args, 1, 2, "eul2tform")
    seq = _seq(args, 1, "eul2tform")
    return _stack([_tform(_eul2rotm(e, seq)) for e in _rows(args[0], 3, "eul2tform", "the Euler angles")])


@builtin("eul2quat")
def _b_eul2quat(interp, args, nargout):
    nargs(args, 1, 2, "eul2quat")
    seq = _seq(args, 1, "eul2quat")
    return np.array([_rotm2quat(_eul2rotm(e, seq)) for e in _rows(args[0], 3, "eul2quat", "the Euler angles")])


@builtin("rotm2eul")
def _b_rotm2eul(interp, args, nargout):
    nargs(args, 1, 2, "rotm2eul")
    seq = _seq(args, 1, "rotm2eul")
    return np.array([_rotm2eul(r, seq) for r in _mats(args[0], 3, "rotm2eul")])


@builtin("tform2eul")
def _b_tform2eul(interp, args, nargout):
    nargs(args, 1, 2, "tform2eul")
    seq = _seq(args, 1, "tform2eul")
    return np.array([_rotm2eul(m[:3, :3], seq) for m in _mats(args[0], 4, "tform2eul")])


@builtin("quat2eul")
def _b_quat2eul(interp, args, nargout):
    nargs(args, 1, 2, "quat2eul")
    seq = _seq(args, 1, "quat2eul")
    return np.array([_rotm2eul(_quat2rotm(q), seq) for q in _rows(args[0], 4, "quat2eul", "the quaternion")])


@builtin("quat2rotm")
def _b_quat2rotm(interp, args, nargout):
    nargs(args, 1, 1, "quat2rotm")
    return _stack([_quat2rotm(q) for q in _rows(args[0], 4, "quat2rotm", "the quaternion")])


@builtin("quat2tform")
def _b_quat2tform(interp, args, nargout):
    nargs(args, 1, 1, "quat2tform")
    return _stack([_tform(_quat2rotm(q)) for q in _rows(args[0], 4, "quat2tform", "the quaternion")])


@builtin("rotm2quat")
def _b_rotm2quat(interp, args, nargout):
    nargs(args, 1, 1, "rotm2quat")
    return np.array([_rotm2quat(r) for r in _mats(args[0], 3, "rotm2quat")])


@builtin("tform2quat")
def _b_tform2quat(interp, args, nargout):
    nargs(args, 1, 1, "tform2quat")
    return np.array([_rotm2quat(m[:3, :3]) for m in _mats(args[0], 4, "tform2quat")])


@builtin("rotm2tform")
def _b_rotm2tform(interp, args, nargout):
    nargs(args, 1, 1, "rotm2tform")
    return _stack([_tform(r) for r in _mats(args[0], 3, "rotm2tform")])


@builtin("tform2rotm")
def _b_tform2rotm(interp, args, nargout):
    nargs(args, 1, 1, "tform2rotm")
    return _stack([m[:3, :3].copy() for m in _mats(args[0], 4, "tform2rotm")])


@builtin("trvec2tform")
def _b_trvec2tform(interp, args, nargout):
    nargs(args, 1, 1, "trvec2tform")
    return _stack([_tform(np.eye(3), t) for t in _rows(args[0], 3, "trvec2tform", "the translation")])


@builtin("tform2trvec")
def _b_tform2trvec(interp, args, nargout):
    nargs(args, 1, 1, "tform2trvec")
    return np.array([m[:3, 3] for m in _mats(args[0], 4, "tform2trvec")])


@builtin("axang2rotm")
def _b_axang2rotm(interp, args, nargout):
    nargs(args, 1, 1, "axang2rotm")
    out = []
    for v in _rows(args[0], 4, "axang2rotm", "the axis and angle"):
        n = np.linalg.norm(v[:3])
        if n == 0:
            raise MatlabError("axang2rotm: the rotation axis must not be zero.")
        x, y, z = v[:3] / n
        c, s = np.cos(v[3]), np.sin(v[3])
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        out.append(np.eye(3) * c + s * k + (1 - c) * np.outer([x, y, z], [x, y, z]))
    return _stack(out)


@builtin("rotm2axang")
def _b_rotm2axang(interp, args, nargout):
    nargs(args, 1, 1, "rotm2axang")
    out = []
    for r in _mats(args[0], 3, "rotm2axang"):
        q = _rotm2quat(r)
        s = np.linalg.norm(q[1:])
        angle = 2 * np.arctan2(s, q[0])
        axis = q[1:] / s if s > 1e-12 else np.array([0.0, 0.0, 1.0])
        out.append([*axis, angle])
    return np.array(out)


def _rot_deg(axis: str, name: str):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        a = fnum(args[0])
        if a.size != 1:
            raise MatlabError(f"{name}: the angle (in degrees) must be a single number.")
        return _axis(axis, np.deg2rad(float(a.flat[0])))

    return f


builtin("rotx")(_rot_deg("X", "rotx"))
builtin("roty")(_rot_deg("Y", "roty"))
builtin("rotz")(_rot_deg("Z", "rotz"))


# -- code-generation directives: they do not change what the code computes -------------------------


@builtin("coder.extrinsic", "coder.inline", "coder.varsize", "coder.cinclude", "coder.updateBuildInfo", "coder.columnMajor")
def _coder_noop(interp, args, nargout):
    return None


@builtin("coder.const", "coder.nullcopy", "coder.ignoreConst", "coder.ignoreSize")
def _coder_same(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments.")
    return args[0]


@builtin("coder.target")
def _coder_target(interp, args, nargout):
    if not args:
        return char("")  # running as ordinary code
    name = to_str(args[0]) if is_text(args[0]) or is_char(args[0]) else ""
    return np.array([[name in ("MATLAB", "")]])


@builtin("coder.ceval")
def _coder_ceval(interp, args, nargout):
    raise MatlabError("coder.ceval calls compiled C code, which mylab cannot run.")


@builtin("simsizes")
def _simsizes(interp, args, nargout):
    """The size record of a level-1 S-function: a struct to fill in, or that struct as a vector."""
    from ..values import Struct

    fields = ["NumContStates", "NumDiscStates", "NumOutputs", "NumInputs", "DirFeedthrough", "NumSampleTimes"]
    if not args:
        return Struct.scalar({f: 0.0 for f in fields})
    s = args[0]
    if not isinstance(s, Struct):
        raise MatlabError("simsizes: pass the struct that simsizes returned, with its fields filled in.")
    v = [float(fnum(s.get_field(0, f)).flat[0]) if f in s.fields else 0.0 for f in fields]
    return np.array([[v[0], v[1], v[2], v[3], 0.0, v[4], v[5]]])
