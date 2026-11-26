"""Numerical integration: adaptive Gauss-Kronrod quadrature in one, two and three dimensions,
plus the classic adaptive Simpson and Lobatto rules."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..values import FunctionHandle, simplify
from ._registry import builtin
from ._util import fnum, is_text, nargs, text

# Gauss-Kronrod (7, 15) rule on [-1, 1]: positive nodes, then the centre.
_XK = np.array([0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                0.207784955007898467600689403773245, 0.0])
_WK = np.array([0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                0.204432940075298892414161999234649, 0.209482141084727828012999174891714])
_WG = np.array([0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                0.381830050505118944950369775488975, 0.417959183673469387755102040816327])
NODES = np.concatenate([-_XK[:7], [0.0], _XK[6::-1]])
W15 = np.concatenate([_WK[:7], [_WK[7]], _WK[6::-1]])
W7 = np.zeros(15)
W7[1:7:2] = _WG[:3]
W7[7] = _WG[3]
W7[13:7:-2] = _WG[:3]


def _is_fn(v) -> bool:
    return isinstance(v, FunctionHandle) or (is_text(v) and len(text(v)) > 0)


def _num(v, what: str) -> complex | float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"{what} must be a scalar.")
    z = a.flat[0]
    return float(np.real(z)) if np.imag(z) == 0 else complex(z)


def _options(rest: list, allowed: dict, what: str) -> dict:
    """Name-value pairs -> dict keyed by canonical lower-case names."""
    if len(rest) % 2:
        raise MatlabError(f"{what}: options must be name-value pairs.")
    out = {}
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"{what}: expected an option name at argument {k + 1} of the options.")
        key = text(rest[k]).lower()
        hits = [n for n in allowed if n == key] or [n for n in allowed if n.startswith(key)]
        if len(hits) != 1:
            raise MatlabError(f"{what}: unknown option '{text(rest[k])}'. Options are: "
                              + ", ".join(allowed.values()) + ".")
        out[hits[0]] = rest[k + 1]
    return out


def _tol(opts: dict, name: str, default: float, what: str) -> float:
    if name not in opts:
        return default
    a = fnum(opts[name])
    if a.size != 1 or not np.real(a.flat[0]) >= 0:
        raise MatlabError(f"{what}: {name} must be a nonnegative scalar.")
    return float(np.real(a.flat[0]))


def _flag(v, what: str) -> bool:
    if is_text(v):
        w = text(v).lower()
        if w in ("true", "on"):
            return True
        if w in ("false", "off"):
            return False
        raise MatlabError(f"{what} must be true or false.")
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"{what} must be true or false.")
    return bool(a.flat[0] != 0)


# -- the adaptive engine --------------------------------------------------------------------------

class _Segment:
    """One piece of the integration path with a change of variable t -> x."""

    def __init__(self, z0, z1):
        self.z0, self.z1 = z0, z1
        inf0, inf1 = np.isinf(z0), np.isinf(z1)
        if inf0 and inf1:
            self.kind, self.t0, self.t1 = "both", -1.0, 1.0
        elif inf1:
            self.kind, self.t0, self.t1 = "right", 0.0, 1.0
        elif inf0:
            self.kind, self.t0, self.t1 = "left", -1.0, 0.0
        else:
            self.kind, self.t0, self.t1 = "finite", -1.0, 1.0

    def map(self, t):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.kind == "finite":
                d = self.z1 - self.z0
                return 0.25 * d * t * (3 - t * t) + 0.5 * (self.z1 + self.z0), 0.75 * d * (1 - t * t)
            if self.kind == "right":
                u = t / (1 - t)
                return self.z0 + u * u, 2 * t / (1 - t) ** 3
            if self.kind == "left":
                u = t / (1 + t)
                return self.z1 - u * u, -2 * t / (1 + t) ** 3
            return t / (1 - t * t), (1 + t * t) / (1 - t * t) ** 2


def gk_adaptive(F, a, b, rtol=1e-6, atol=1e-10, max_intervals=650, waypoints=(), plain=False):
    """Integrate F along the path a -> waypoints -> b.

    ``F(x)`` takes a 1-D array of points and returns an array ``(nvals, npts)``.
    Returns (Q of shape (nvals,), error bound, message or None). ``plain`` skips the change of
    variable that weakens end-point singularities (used for inner integrals of smooth maps)."""
    pts = [a, *waypoints, b]
    cplx = any(isinstance(p, complex) for p in pts)
    sign = 1.0
    if not cplx:
        if a == b:
            return None, 0.0, None
        if a > b:
            pts = [b, *sorted(waypoints), a]
            sign = -1.0
        else:
            pts = [a, *sorted(waypoints), b]
        pts = [p for k, p in enumerate(pts) if k == 0 or p != pts[k - 1]]
        if any(not (pts[0] <= w <= pts[-1]) for w in pts[1:-1]):
            raise MatlabError("Waypoints must lie between the limits of integration.")
    else:
        if any(np.isinf(p) for p in pts):
            raise MatlabError("Infinite limits cannot be combined with a complex path.")
    segs = [_Segment(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    if plain:
        for sg in segs:
            if sg.kind == "finite":
                d, m = sg.z1 - sg.z0, 0.5 * (sg.z1 + sg.z0)
                sg.map = (lambda t, d=d, m=m: (0.5 * d * t + m, np.full(np.shape(t), 0.5 * d)))
    per = max(1, int(np.ceil(10 / len(segs))))
    seg_id, lo, hi = [], [], []
    for k, sg in enumerate(segs):
        edges = np.linspace(sg.t0, sg.t1, per + 1)
        seg_id += [k] * per
        lo += list(edges[:-1])
        hi += list(edges[1:])
    seg_id, lo, hi = np.array(seg_id), np.array(lo), np.array(hi)
    total_len = float(np.sum(hi - lo))
    q_ok = None
    err_ok = 0.0
    message = None
    while True:
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        t = mid[:, None] + half[:, None] * NODES[None, :]
        x = np.empty(t.shape, dtype=complex if cplx else float)
        w = np.empty(t.shape, dtype=complex if cplx else float)
        for k, sg in enumerate(segs):
            m = seg_id == k
            if m.any():
                x[m], w[m] = sg.map(t[m])
        fx = np.asarray(F(x.reshape(-1)))
        if fx.ndim == 1:
            fx = fx.reshape(1, -1)
        nv = fx.shape[0]
        fx = fx.reshape(nv, t.shape[0], 15) * w[None, :, :] * half[None, :, None]
        # a zero weight at an end point multiplies a possibly infinite value: that product is 0
        fx = np.where(w[None, :, :] == 0, 0.0, fx)
        q15 = fx @ W15
        q7 = fx @ W7
        if q_ok is None:
            q_ok = np.zeros(nv, dtype=q15.dtype)
        elif q15.dtype.kind == "c" and q_ok.dtype.kind != "c":
            q_ok = q_ok.astype(complex)
        errs = np.max(np.abs(q15 - q7), axis=0)
        q = q_ok + q15.sum(axis=1)
        if not np.all(np.isfinite(q)):
            message = "Infinite or Not-a-Number value encountered."
            return sign * q, np.inf, message
        tol = max(atol, rtol * float(np.max(np.abs(q))))
        total_err = err_ok + float(errs.sum())
        if total_err <= tol:
            return sign * q, total_err, None
        # keep the pieces that already meet their share of the tolerance; split the others
        share = tol * (hi - lo) / total_len
        bad = errs > share
        if not bad.any():
            return sign * q, total_err, None
        q_ok = q_ok + q15[:, ~bad].sum(axis=1)
        err_ok += float(errs[~bad].sum())
        lo_b, hi_b, id_b = lo[bad], hi[bad], seg_id[bad]
        mid_b = 0.5 * (lo_b + hi_b)
        if np.any((mid_b <= lo_b) | (mid_b >= hi_b)):
            message = "Minimum step size reached near a singularity; the result may be inaccurate."
            return sign * q, total_err, message
        if 2 * lo_b.size > max_intervals:
            message = ("Reached the limit on the maximum number of intervals in use. "
                       f"Approximate bound on error is {total_err:.1e}. The integral may not exist, or "
                       "it may be difficult to approximate numerically to the requested accuracy.")
            return sign * q, total_err, message
        lo = np.concatenate([lo_b, mid_b])
        hi = np.concatenate([mid_b, hi_b])
        seg_id = np.concatenate([id_b, id_b])


def _vector_fun(interp, fun, what: str):
    """Wrap a MATLAB integrand that maps a row vector of points to values of the same size."""
    def F(x):
        out = interp.call_function_value(fun, [x.reshape(1, -1)], 1)
        if not out:
            raise MatlabError(f"{what}: the integrand must return a value.")
        v = fnum(out[0])
        if v.size != x.size:
            if v.size == 1:
                raise MatlabError(f"{what}: the integrand returned a scalar for {x.size} points. It must "
                                  "accept a vector and return a vector of the same size (use .*, ./ and .^), "
                                  "or set 'ArrayValued' to true.")
            raise MatlabError(f"{what}: the integrand must return an output of the same size as its input "
                              f"({x.size} points, got {v.size} values).")
        return v.reshape(1, -1)
    return F


def _array_fun(interp, fun, what: str, shape_box: list):
    def F(x):
        cols = []
        for xv in x:
            out = interp.call_function_value(fun, [np.array([[xv]])], 1)
            if not out:
                raise MatlabError(f"{what}: the integrand must return a value.")
            v = fnum(out[0])
            if not shape_box:
                shape_box.append(v.shape)
            elif v.shape != shape_box[0]:
                raise MatlabError(f"{what}: the integrand must return arrays of the same size at every point.")
            cols.append(v.flatten(order="F"))
        return np.column_stack(cols) if cols else np.zeros((0, 0))
    return F


def _waypoints(opts: dict):
    if "waypoints" not in opts:
        return []
    w = fnum(opts["waypoints"]).flatten(order="F")
    if w.size and not np.all(np.isfinite(w)):
        raise MatlabError("Waypoints must be finite.")
    return [float(np.real(v)) if np.imag(v) == 0 else complex(v) for v in w]


def _finish(interp, q, msg, shape=None):
    if msg:
        interp.warn(msg)
    if shape is not None:
        return simplify(np.asarray(q).reshape(shape, order="F"))
    v = q[0]
    return complex(v) if np.iscomplexobj(q) and np.imag(v) != 0 else float(np.real(v))


@builtin("integral")
def _integral(interp, args, nargout):
    nargs(args, 3, None, "integral")
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError("integral: the first argument must be a function handle.")
    a, b = _num(args[1], "integral: the lower limit"), _num(args[2], "integral: the upper limit")
    opts = _options(list(args[3:]), {"reltol": "RelTol", "abstol": "AbsTol", "arrayvalued": "ArrayValued",
                                     "waypoints": "Waypoints"}, "integral")
    rtol = max(_tol(opts, "reltol", 1e-6, "integral"), 100 * np.finfo(float).eps)
    atol = _tol(opts, "abstol", 1e-10, "integral")
    way = _waypoints(opts)
    if "arrayvalued" in opts and _flag(opts["arrayvalued"], "integral: ArrayValued"):
        box: list = []
        F = _array_fun(interp, fun, "integral", box)
        if a == b:
            v = fnum(interp.call_function_value(fun, [np.array([[a]])], 1)[0])
            return np.zeros(v.shape)
        q, _, msg = gk_adaptive(F, a, b, rtol, atol, 650, way)
        return _finish(interp, q, msg, box[0])
    if a == b and not way:
        return 0.0
    q, _, msg = gk_adaptive(_vector_fun(interp, fun, "integral"), a, b, rtol, atol, 650, way)
    return _finish(interp, q, msg)


@builtin("quadgk")
def _quadgk(interp, args, nargout):
    nargs(args, 3, None, "quadgk")
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError("quadgk: the first argument must be a function handle.")
    a, b = _num(args[1], "quadgk: the lower limit"), _num(args[2], "quadgk: the upper limit")
    opts = _options(list(args[3:]), {"reltol": "RelTol", "abstol": "AbsTol", "waypoints": "Waypoints",
                                     "maxintervalcount": "MaxIntervalCount"}, "quadgk")
    rtol = max(_tol(opts, "reltol", 1e-6, "quadgk"), 100 * np.finfo(float).eps)
    atol = _tol(opts, "abstol", 1e-10, "quadgk")
    maxint = int(_tol(opts, "maxintervalcount", 650, "quadgk"))
    way = _waypoints(opts)
    if a == b and not way:
        return (0.0, 0.0) if nargout > 1 else 0.0
    q, err, msg = gk_adaptive(_vector_fun(interp, fun, "quadgk"), a, b, rtol, atol, maxint, way)
    val = _finish(interp, q, msg)
    return (val, float(err)) if nargout > 1 else val


# -- adaptive Simpson and Lobatto (quad, quadl, quadv) -----------------------------------------------

def _classic_args(args: list, what: str):
    nargs(args, 3, None, what)
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError(f"{what}: the first argument must be a function handle.")
    a = float(np.real(_num(args[1], f"{what}: the lower limit")))
    b = float(np.real(_num(args[2], f"{what}: the upper limit")))
    if not (np.isfinite(a) and np.isfinite(b)):
        raise MatlabError(f"{what}: the limits must be finite; use integral for infinite intervals.")
    tol = 1e-6
    if len(args) > 3 and not (isinstance(args[3], np.ndarray) and args[3].size == 0):
        tol = float(np.real(fnum(args[3]).flat[0]))
        if not tol > 0:
            raise MatlabError(f"{what}: the tolerance must be positive.")
    trace = len(args) > 4 and not (isinstance(args[4], np.ndarray) and args[4].size == 0) \
        and bool(np.any(fnum(args[4]) != 0))
    return fun, a, b, tol, trace, list(args[5:])


def _simpson(f, a, b, tol, trace_out=None):
    """Adaptive Simpson; ``f(x_array)`` returns (nvals, npts). Returns (Q, evaluations, warning)."""
    h = b - a
    # three starting panels of unequal width, so a symmetric integrand is not judged on a lucky sample
    edges = [a, a + h / np.pi, a + h * (1.0 - np.exp(-1.0)), b]
    mids = [0.5 * (edges[k] + edges[k + 1]) for k in range(3)]
    y = np.array(f(np.array(edges + mids)))
    count = [7]
    for k in (0, 3):
        y[:, k][~np.isfinite(y[:, k])] = 0.0  # an integrable end-point singularity contributes nothing
    hmin = np.finfo(float).eps / 1024 * abs(h)
    warn = [None]

    def step(a, b, fa, fm, fb, depth):
        h = b - a
        c = 0.5 * (a + b)
        if abs(h) < hmin or c == a or c == b:
            warn[0] = "Minimum step size reached; singularity possible."
            return h * fm
        if count[0] > 10000:
            warn[0] = "Maximum function count exceeded; singularity likely."
            return h / 6 * (fa + 4 * fm + fb)
        yy = f(np.array([0.5 * (a + c), 0.5 * (c + b)]))
        count[0] += 2
        fd, fe = yy[:, 0], yy[:, 1]
        q1 = h / 6 * (fa + 4 * fm + fb)
        q2 = h / 12 * (fa + 4 * fd + 2 * fm + 4 * fe + fb)
        q = q2 + (q2 - q1) / 15
        if trace_out is not None:
            trace_out(f"{count[0]:8d} {a:16.10f} {h:16.8e} {float(np.real(q.flat[0])):16.10f}\n")
        if not np.all(np.isfinite(q)):
            warn[0] = "Infinite or Not-a-Number function value encountered."
            return q
        if np.max(np.abs(q2 - q)) <= tol or depth > 60:
            return q
        return step(a, c, fa, fd, fm, depth + 1) + step(c, b, fm, fe, fb, depth + 1)

    total = sum(step(edges[k], edges[k + 1], y[:, k], y[:, 4 + k], y[:, k + 1], 0) for k in range(3))
    return total, count[0], warn[0]


def _lobatto(f, a, b, tol, trace_out=None):
    """Adaptive 4-point Gauss-Lobatto rule with its 7-point Kronrod extension.
    Returns (Q, evaluations, warning)."""
    alpha = np.sqrt(2.0 / 3.0)
    beta = 1.0 / np.sqrt(5.0)
    count = [0]
    warn = [None]

    def F(x):
        count[0] += x.size
        return np.array(f(x))

    m, h = 0.5 * (a + b), 0.5 * (b - a)
    xs = np.array([a, m - alpha * h, m - beta * h, m, m + beta * h, m + alpha * h, b])
    y = F(xs)
    for k in (0, 6):
        y[:, k][~np.isfinite(y[:, k])] = 0.0
    hmin = np.finfo(float).eps / 1024 * abs(b - a)

    def step(a, b, fa, fb, depth):
        m, h = 0.5 * (a + b), 0.5 * (b - a)
        pts = np.array([m - alpha * h, m - beta * h, m, m + beta * h, m + alpha * h])
        if abs(h) < hmin or pts[0] <= a or pts[-1] >= b:
            warn[0] = "Minimum step size reached; singularity possible."
            return h * (fa + fb)
        if count[0] > 10000:
            warn[0] = "Maximum function count exceeded; singularity likely."
            return h * (fa + fb)
        yy = F(pts)
        q1 = (h / 6) * (fa + fb + 5 * (yy[:, 1] + yy[:, 3]))
        q2 = (h / 1470) * (77 * (fa + fb) + 432 * (yy[:, 0] + yy[:, 4]) + 625 * (yy[:, 1] + yy[:, 3])
                           + 672 * yy[:, 2])
        if trace_out is not None:
            trace_out(f"{count[0]:8d} {a:16.10f} {2 * h:16.8e} {float(np.real(q2.flat[0])):16.10f}\n")
        if not np.all(np.isfinite(q2)):
            warn[0] = "Infinite or Not-a-Number function value encountered."
            return q2
        if np.max(np.abs(q1 - q2)) <= tol or depth > 60:
            return q2
        edges = [a, pts[0], pts[1], pts[2], pts[3], pts[4], b]
        vals = [fa, yy[:, 0], yy[:, 1], yy[:, 2], yy[:, 3], yy[:, 4], fb]
        return sum(step(edges[k], edges[k + 1], vals[k], vals[k + 1], depth + 1) for k in range(6))

    total = sum(step(xs[k], xs[k + 1], y[:, k], y[:, k + 1], 0) for k in range(6))
    return total, count[0], warn[0]


def _classic(kind: str):
    def f(interp, args, nargout):
        fun, a, b, tol, trace, extra = _classic_args(args, kind)
        shape: list = []

        def F(x):
            if kind == "quadv":
                cols = []
                for xv in x:
                    v = fnum(interp.call_function_value(fun, [np.array([[xv]]), *extra], 1)[0])
                    if not shape:
                        shape.append(v.shape)
                    elif v.shape != shape[0]:
                        raise MatlabError("quadv: the integrand must return arrays of the same size.")
                    cols.append(v.flatten(order="F"))
                return np.column_stack(cols)
            v = fnum(interp.call_function_value(fun, [x.reshape(1, -1), *extra], 1)[0])
            if v.size != x.size:
                raise MatlabError(f"{kind}: the integrand must return one value per point; write it with "
                                  "element-wise operators (.*, ./, .^).")
            return v.reshape(1, -1)

        trace_out = interp.out if trace else None
        if a == b:
            if kind == "quadv":
                v = fnum(interp.call_function_value(fun, [np.array([[a]]), *extra], 1)[0])
                q = np.zeros(v.shape)
            else:
                q = 0.0
            return (q, 0.0) if nargout > 1 else q
        sign = 1.0
        if a > b:
            a, b, sign = b, a, -1.0
        if kind == "quadl":
            q, count, msg = _lobatto(F, a, b, tol, trace_out)
        else:
            q, count, msg = _simpson(F, a, b, tol, trace_out)
        if msg:
            interp.warn(msg)
        q = sign * q
        if kind == "quadv":
            val = simplify(np.asarray(q).reshape(shape[0], order="F"))
        else:
            val = complex(q[0]) if np.iscomplexobj(q) and np.imag(q[0]) != 0 else float(np.real(q[0]))
        return (val, float(count)) if nargout > 1 else val

    return f


builtin("quad")(_classic("quad"))
builtin("quadl")(_classic("quadl"))
builtin("quadv")(_classic("quadv"))


# -- two and three dimensions ---------------------------------------------------------------------

def _limit(interp, v, what: str):
    """A limit given as a number or as a function of the outer variables -> callable on arrays."""
    if _is_fn(v):
        def lim(*xs):
            out = fnum(interp.call_function_value(v, [np.asarray(x, dtype=float) for x in xs], 1)[0])
            ref = np.asarray(xs[0])
            if out.size == 1 and ref.size != 1:
                return np.full(ref.shape, float(np.real(out.flat[0])))
            if out.size != ref.size:
                raise MatlabError(f"{what}: a limit function must return one value per point.")
            return np.real(out).reshape(ref.shape)
        return lim, False
    c = float(np.real(_num(v, f"{what}: a limit")))
    return (lambda *xs: np.full(np.shape(xs[0]), c)), True


def _call2(interp, fun, what, *arrays):
    out = fnum(interp.call_function_value(fun, list(arrays), 1)[0])
    if out.size != arrays[0].size:
        if out.size == 1:
            return np.full(arrays[0].shape, out.flat[0])
        raise MatlabError(f"{what}: the integrand must return an array of the same size as its inputs; "
                          "write it with element-wise operators (.*, ./, .^).")
    return out.reshape(arrays[0].shape)


def _inner_y(interp, g, xs, ylo, yhi, rtol, atol, what):
    """Integral over y of g(X, Y) for each x in xs, with limits ylo(x), yhi(x) (arrays)."""
    nx = xs.size
    finite = np.isfinite(ylo) & np.isfinite(yhi)
    out = np.zeros(nx, dtype=complex)
    if finite.all():
        width = yhi - ylo
        mid = 0.5 * (yhi + ylo)

        def F(s):
            # s in [-1, 1] with the end-point weakening map applied by the engine: y = mid + width/2 * s
            X = np.repeat(xs.reshape(-1, 1), s.size, axis=1)
            Y = mid.reshape(-1, 1) + 0.5 * width.reshape(-1, 1) * s.reshape(1, -1)
            return g(X, Y) * (0.5 * width.reshape(-1, 1))

        q, _, msg = gk_adaptive(F, -1.0, 1.0, rtol, atol, 650)
        return q, msg
    msg = None
    for k in range(nx):
        if ylo[k] == yhi[k]:
            continue

        def F1(y, k=k):
            return g(np.full((1, y.size), xs[k]), y.reshape(1, -1))

        q, _, m = gk_adaptive(F1, float(ylo[k]), float(yhi[k]), rtol, atol, 650)
        out[k] = q[0]
        msg = msg or m
    return out, msg


def _iterated2(interp, g, xmin, xmax, ylo_f, yhi_f, rtol, atol, what):
    msgs = []

    def outer(x):
        ylo, yhi = ylo_f(x), yhi_f(x)
        q, m = _inner_y(interp, g, x, ylo, yhi, rtol, 0.1 * atol, what)
        if m:
            msgs.append(m)
        return np.asarray(q).reshape(1, -1)

    if xmin == xmax:
        return np.zeros(1), 0.0, None
    q, err, msg = gk_adaptive(outer, xmin, xmax, rtol, atol, 650)
    return q, err, msg or (msgs[0] if msgs else None)


def _tiled2(interp, g, xmin, xmax, ylo_f, yhi_f, rtol, atol, max_evals, singular):
    """Adaptive tensor Gauss-Kronrod rule on rectangles of the mapped unit square."""
    if xmin == xmax:
        return np.zeros(1), 0.0, None
    sign = 1.0
    if xmin > xmax:
        xmin, xmax, sign = xmax, xmin, -1.0
    # tiles in (u, v) in [-1, 1]^2
    tiles = [(-1.0, 0.0, -1.0, 0.0), (0.0, 1.0, -1.0, 0.0), (-1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
    q_ok = 0.0
    err_ok = 0.0
    evals = 0
    dx, mx = 0.5 * (xmax - xmin), 0.5 * (xmax + xmin)
    while True:
        T = np.array(tiles)
        nt = T.shape[0]
        hu, mu = 0.5 * (T[:, 1] - T[:, 0]), 0.5 * (T[:, 1] + T[:, 0])
        hv, mv = 0.5 * (T[:, 3] - T[:, 2]), 0.5 * (T[:, 3] + T[:, 2])
        U = (mu[:, None] + hu[:, None] * NODES[None, :])[:, :, None] * np.ones((1, 1, 15))
        V = (mv[:, None] + hv[:, None] * NODES[None, :])[:, None, :] * np.ones((1, 15, 1))
        if singular:
            X = mx + dx * 0.5 * U * (3 - U * U)
            wx = dx * 1.5 * (1 - U * U)
            S = 0.5 * V * (3 - V * V)
            ws = 1.5 * (1 - V * V)
        else:
            X, wx, S, ws = mx + dx * U, np.full(U.shape, dx), V, np.ones(V.shape)
        ylo, yhi = ylo_f(X), yhi_f(X)
        Y = 0.5 * (yhi + ylo) + 0.5 * (yhi - ylo) * S
        vals = g(X.reshape(1, -1), Y.reshape(1, -1)).reshape(X.shape)
        evals += X.size
        jac = wx * ws * 0.5 * (yhi - ylo) * (hu * hv)[:, None, None]
        vals = np.where(jac == 0, 0.0, vals * jac)
        q15 = np.einsum("tij,i,j->t", vals, W15, W15)
        q7 = np.einsum("tij,i,j->t", vals, W7, W7)
        errs = np.abs(q15 - q7)
        q = q_ok + q15.sum()
        if not np.isfinite(q):
            return np.array([sign * q]), np.inf, "Infinite or Not-a-Number value encountered."
        total_err = err_ok + float(errs.sum())
        tol = max(atol, rtol * abs(q))
        if total_err <= tol:
            return np.array([sign * q]), total_err, None
        area = 4 * hu * hv
        bad = errs > tol * area / 4.0
        if not bad.any():
            return np.array([sign * q]), total_err, None
        q_ok += q15[~bad].sum()
        err_ok += float(errs[~bad].sum())
        if evals + 4 * int(bad.sum()) * 225 > max_evals:
            return np.array([sign * q]), total_err, (
                "Reached the maximum number of function evaluations. The result fails the global error "
                f"test; approximate bound on error is {total_err:.1e}.")
        new = []
        for k in np.nonzero(bad)[0]:
            u0, u1, v0, v1 = T[k]
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            if um in (u0, u1) or vm in (v0, v1):
                return np.array([sign * q]), total_err, "Minimum tile size reached; singularity possible."
            new += [(u0, um, v0, vm), (um, u1, v0, vm), (u0, um, vm, v1), (um, u1, vm, v1)]
        tiles = new
        del nt


def _method(opts: dict, what: str, finite: bool) -> str:
    if "method" not in opts:
        return "tiled" if finite else "iterated"
    if not is_text(opts["method"]):
        raise MatlabError(f"{what}: Method must be 'auto', 'tiled' or 'iterated'.")
    m = text(opts["method"]).lower()
    if m == "auto":
        return "tiled" if finite else "iterated"
    if m not in ("tiled", "iterated"):
        raise MatlabError(f"{what}: Method must be 'auto', 'tiled' or 'iterated'.")
    if m == "tiled" and not finite:
        raise MatlabError(f"{what}: the 'tiled' method needs finite limits; use 'iterated'.")
    return m


def _integrate2(interp, fun, xmin, xmax, ylo_f, yhi_f, const_y, opts, what, max_evals=200000):
    rtol = max(_tol(opts, "reltol", 1e-6, what), 100 * np.finfo(float).eps)
    atol = _tol(opts, "abstol", 1e-10, what)
    finite = bool(np.isfinite(xmin) and np.isfinite(xmax))
    if finite and const_y:
        probe = np.array([[0.5 * (xmin + xmax)]])
        finite = bool(np.isfinite(ylo_f(probe)).all() and np.isfinite(yhi_f(probe)).all())

    def g(X, Y):
        return _call2(interp, fun, what, X, Y)

    method = _method(opts, what, finite)
    if method == "tiled":
        return _tiled2(interp, g, xmin, xmax, ylo_f, yhi_f, rtol, atol, max_evals, True)
    return _iterated2(interp, g, xmin, xmax, ylo_f, yhi_f, rtol, atol, what)


_OPTS2 = {"reltol": "RelTol", "abstol": "AbsTol", "method": "Method"}


@builtin("integral2")
def _integral2(interp, args, nargout):
    nargs(args, 5, None, "integral2")
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError("integral2: the first argument must be a function handle.")
    xmin = float(np.real(_num(args[1], "integral2: xmin")))
    xmax = float(np.real(_num(args[2], "integral2: xmax")))
    ylo, c1 = _limit(interp, args[3], "integral2")
    yhi, c2 = _limit(interp, args[4], "integral2")
    opts = _options(list(args[5:]), _OPTS2, "integral2")
    q, _, msg = _integrate2(interp, fun, xmin, xmax, ylo, yhi, c1 and c2, opts, "integral2")
    return _finish(interp, q, msg)


@builtin("quad2d")
def _quad2d(interp, args, nargout):
    nargs(args, 5, None, "quad2d")
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError("quad2d: the first argument must be a function handle.")
    a = float(np.real(_num(args[1], "quad2d: a")))
    b = float(np.real(_num(args[2], "quad2d: b")))
    if not (np.isfinite(a) and np.isfinite(b)):
        raise MatlabError("quad2d: the limits must be finite; use integral2 for infinite regions.")
    ylo, _ = _limit(interp, args[3], "quad2d")
    yhi, _ = _limit(interp, args[4], "quad2d")
    opts = _options(list(args[5:]), {"reltol": "RelTol", "abstol": "AbsTol", "maxfunevals": "MaxFunEvals",
                                     "singular": "Singular", "failureplot": "FailurePlot"}, "quad2d")
    atol = _tol(opts, "abstol", 1e-5, "quad2d")
    rtol = max(_tol(opts, "reltol", 100 * np.finfo(float).eps, "quad2d"), 100 * np.finfo(float).eps)
    max_evals = int(_tol(opts, "maxfunevals", 2000, "quad2d"))
    singular = _flag(opts["singular"], "quad2d: Singular") if "singular" in opts else True
    if "failureplot" in opts and _flag(opts["failureplot"], "quad2d: FailurePlot"):
        raise MatlabError("quad2d: the FailurePlot option is not supported.")

    def g(X, Y):
        return _call2(interp, fun, "quad2d", X, Y)

    q, err, msg = _tiled2(interp, g, a, b, ylo, yhi, rtol, atol, max(max_evals, 900), singular)
    val = _finish(interp, q, msg)
    return (val, float(err)) if nargout > 1 else val


@builtin("integral3")
def _integral3(interp, args, nargout):
    nargs(args, 7, None, "integral3")
    fun = args[0]
    if not _is_fn(fun):
        raise MatlabError("integral3: the first argument must be a function handle.")
    xmin = float(np.real(_num(args[1], "integral3: xmin")))
    xmax = float(np.real(_num(args[2], "integral3: xmax")))
    ylo, cy1 = _limit(interp, args[3], "integral3")
    yhi, cy2 = _limit(interp, args[4], "integral3")
    zlo, cz1 = _limit(interp, args[5], "integral3")
    zhi, cz2 = _limit(interp, args[6], "integral3")
    opts = _options(list(args[7:]), _OPTS2, "integral3")
    rtol = max(_tol(opts, "reltol", 1e-6, "integral3"), 100 * np.finfo(float).eps)
    atol = _tol(opts, "abstol", 1e-10, "integral3")
    msgs: list = []
    inner_opts = dict(opts)

    def slab(x):
        """Integral over (y, z) for each x in the array x."""
        out = np.zeros(x.size, dtype=complex)
        for k, xv in enumerate(x):
            xa = np.array([[xv]])
            y0, y1 = float(ylo(xa).flat[0]), float(yhi(xa).flat[0])

            def g2(Y, Z, xv=xv):
                return interp.call_function_value(fun, [np.full(Y.shape, xv), Y, Z], 1)[0]

            def zl(Y, xv=xv):
                return zlo(np.full(np.shape(Y), xv), Y)

            def zh(Y, xv=xv):
                return zhi(np.full(np.shape(Y), xv), Y)

            def gg(Y, Z):
                v = fnum(g2(Y, Z))
                if v.size == 1 and Y.size != 1:
                    return np.full(Y.shape, v.flat[0])
                if v.size != Y.size:
                    raise MatlabError("integral3: the integrand must return an array of the same size as "
                                      "its inputs; write it with element-wise operators (.*, ./, .^).")
                return v.reshape(Y.shape)

            finite = bool(np.isfinite(y0) and np.isfinite(y1))
            if finite:
                pr = np.array([[0.5 * (y0 + y1)]])
                finite = bool(np.isfinite(zl(pr)).all() and np.isfinite(zh(pr)).all())
            method = _method(inner_opts, "integral3", finite)
            if y0 == y1:
                continue
            if method == "tiled":
                q, _, m = _tiled2(interp, gg, y0, y1, zl, zh, rtol, 0.1 * atol, 200000, True)
            else:
                q, _, m = _iterated2(interp, gg, y0, y1, zl, zh, rtol, 0.1 * atol, "integral3")
            if m:
                msgs.append(m)
            out[k] = q[0]
        return out.reshape(1, -1)

    if xmin == xmax:
        return 0.0
    q, _, msg = gk_adaptive(slab, xmin, xmax, rtol, atol, 650)
    return _finish(interp, q, msg or (msgs[0] if msgs else None))


def _nested_classic(kind: str):
    nlim = 4 if kind == "dblquad" else 6

    def f(interp, args, nargout):
        nargs(args, 1 + nlim, 3 + nlim, kind)
        fun = args[0]
        if not _is_fn(fun):
            raise MatlabError(f"{kind}: the first argument must be a function handle.")
        lims = [float(np.real(_num(a, f"{kind}: a limit"))) for a in args[1:1 + nlim]]
        if not all(np.isfinite(v) for v in lims):
            raise MatlabError(f"{kind}: the limits must be finite.")
        tol = 1e-6
        rest = list(args[1 + nlim:])
        if rest and not (isinstance(rest[0], np.ndarray) and rest[0].size == 0):
            tol = float(np.real(fnum(rest[0]).flat[0]))
        method = "quad"
        custom = None
        if len(rest) > 1 and not (isinstance(rest[1], np.ndarray) and rest[1].size == 0):
            mv = rest[1]
            name = mv.name if isinstance(mv, FunctionHandle) and not mv.is_anonymous else (
                text(mv) if is_text(mv) else None)
            if name in ("quad", "quadl"):
                method = name
            elif name in ("quadgk", "integral"):
                method = "gk"
            else:
                custom = mv
        msgs: list = []

        def one_d(F, a, b):
            """Integrate a Python function of a row of points over [a, b]."""
            if a == b:
                return 0.0
            if custom is not None:
                raise MatlabError(f"{kind}: only @quad, @quadl, @quadgk and @integral are supported as the "
                                  "quadrature method.")
            sign = 1.0
            if a > b:
                a, b, sign = b, a, -1.0
            if method == "gk":
                q, _, m = gk_adaptive(F, a, b, 0.0, tol, 650)
            elif method == "quadl":
                q, _, m = _lobatto(F, a, b, tol)
            else:
                q, _, m = _simpson(F, a, b, tol)
            if m:
                msgs.append(m)
            return sign * q[0]

        def inner_x(*outer):
            """Integral over x with the outer variables held at scalar values."""
            def F(x):
                v = fnum(interp.call_function_value(fun, [x.reshape(1, -1)] + [np.array([[o]]) for o in outer],
                                                    1)[0])
                if v.size != x.size:
                    if v.size == 1:
                        return np.full((1, x.size), v.flat[0])
                    raise MatlabError(f"{kind}: the integrand must return one value per element of x.")
                return v.reshape(1, -1)
            return one_d(F, lims[0], lims[1])

        if kind == "dblquad":
            def Fy(y):
                return np.array([[inner_x(yv) for yv in y]])
            q = one_d(Fy, lims[2], lims[3])
        else:
            def Fz(z):
                def Fy(y, zv):
                    return np.array([[inner_x(yv, zv) for yv in y]])
                return np.array([[one_d(lambda y, zv=zv: Fy(y, zv), lims[2], lims[3]) for zv in z]])
            q = one_d(Fz, lims[4], lims[5])
        if msgs:
            interp.warn(msgs[0])
        return complex(q) if np.imag(q) != 0 else float(np.real(q))

    return f


builtin("dblquad")(_nested_classic("dblquad"))
builtin("triplequad")(_nested_classic("triplequad"))
