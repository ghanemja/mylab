"""More image processing: colour spaces, contrast, local filters, features, deconvolution, morphology on
gray-scale images, region tools, geometric transformations and image comparison. The file formats, the
basic filters and the core morphology live in images.py."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char, to_int
from ._registry import BUILTINS, builtin
from ._util import is_text, nargs, text
from .images import (
    _cast, _class_range, _conn_structure, _correlate, _from_unit, _gauss1d, _hist_counts, _img, _is_rgb, _label,
    _morph, _nd, _nhood, _options, _pad, _resize_axis, _rgb2gray, _round_half_away, _scalar, _to_unit, _vec,
)


def _f(x) -> np.ndarray:
    return np.array([[float(x)]])


def _flag(v) -> bool:
    return text(v).lower() in ("true", "on", "yes") if is_text(v) else bool(np.ravel(v)[0])


def _like_class(out: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Put a double result back into the class of image a (round and saturate integers)."""
    if a.dtype == bool:
        return out > 0.5
    if a.dtype.kind in "iu":
        return _cast(out, a.dtype)
    return out.astype(a.dtype)


def _planes(a: np.ndarray, fn) -> np.ndarray:
    """Apply a 2-D operation to every plane of an image."""
    if a.ndim == 2:
        return fn(a)
    flat = a.reshape(a.shape[0], a.shape[1], -1)
    out = [fn(flat[..., k]) for k in range(flat.shape[2])]
    res = np.stack(out, axis=-1)
    return res.reshape(res.shape[:2] + a.shape[2:])


def _call(interp, f, args: list, nargout: int = 1):
    return interp.call_function_value(f, args, nargout)


# -- colour spaces ------------------------------------------------------------------------------------

_WHITE = {"d65": (0.95047, 1.0, 1.08883), "d50": (0.96422, 1.0, 0.82521), "icc": (0.9642, 1.0, 0.8249),
          "a": (1.09850, 1.0, 0.35585), "c": (0.98074, 1.0, 1.18232), "e": (1.0, 1.0, 1.0),
          "d55": (0.95682, 1.0, 0.92149), "f2": (0.99186, 1.0, 0.67393), "f7": (0.95041, 1.0, 1.08747),
          "f11": (1.00962, 1.0, 0.64350)}
_SRGB = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_ADOBE = np.array([[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741],
                   [0.0270343, 0.0706872, 0.9911085]])
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _white(v, name: str) -> np.ndarray:
    if v is None:
        return np.array(_WHITE["d65"])
    if is_text(v):
        key = text(v).lower()
        if key not in _WHITE:
            raise MatlabError(f"{name}: unknown white point '{text(v)}'. Use one of: " + ", ".join(_WHITE) + ".")
        return np.array(_WHITE[key])
    w = _vec(v, "white point")
    if w.size != 3:
        raise MatlabError(f"{name}: a white point needs three values [X Y Z].")
    return w


@builtin("whitepoint")
def _whitepoint(interp, args, nargout):
    nargs(args, 0, 1, "whitepoint")
    return _white(args[0] if args else char("icc"), "whitepoint").reshape(1, 3)


def _color_in(v, name: str, scale: bool = True) -> tuple[np.ndarray, tuple, np.dtype]:
    """Colours as an N-by-3 double array, with the original shape and class."""
    a = _img(v, name)
    if a.ndim == 3 and a.shape[2] == 3:
        flat = a.reshape(-1, 3)
    elif a.ndim == 2 and a.shape[1] == 3:
        flat = a
    else:
        raise MatlabError(f"{name}: expected an M-by-3 list of colours or an M-by-N-by-3 image.")
    data = _to_unit(flat) if scale else flat.astype(float)
    return data, a.shape, a.dtype


def _color_out(data: np.ndarray, shape: tuple, dtype=None) -> np.ndarray:
    out = data.reshape(shape)
    if dtype is not None and np.dtype(dtype) == np.float32:
        return out.astype(np.float32)
    return out


def _linearize(c: np.ndarray, space: str) -> np.ndarray:
    if space == "linear-rgb":
        return c
    if space == "adobe-rgb-1998":
        return np.sign(c) * np.abs(c) ** 2.19921875
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _delinearize(c: np.ndarray, space: str) -> np.ndarray:
    if space == "linear-rgb":
        return c
    if space == "adobe-rgb-1998":
        return np.sign(c) * np.abs(c) ** (1 / 2.19921875)
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def _adapt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Bradford chromatic adaptation matrix from white point src to dst."""
    if np.allclose(src, dst):
        return np.eye(3)
    s, d = _BRADFORD @ src, _BRADFORD @ dst
    return np.linalg.inv(_BRADFORD) @ np.diag(d / s) @ _BRADFORD


def _space_opts(args: list, name: str, extra: tuple = ()) -> tuple[list, str, np.ndarray, dict]:
    rest, opts = _options(list(args), ("ColorSpace", "WhitePoint") + extra, name)
    space = text(opts["colorspace"]).lower() if "colorspace" in opts else "srgb"
    if space not in ("srgb", "adobe-rgb-1998", "linear-rgb"):
        raise MatlabError(f"{name}: 'ColorSpace' must be 'srgb', 'adobe-rgb-1998' or 'linear-rgb'.")
    return rest, space, _white(opts.get("whitepoint"), name), opts


def _rgb_to_xyz(rgb: np.ndarray, space: str, wp: np.ndarray) -> np.ndarray:
    lin = _linearize(rgb, space)
    m = _ADOBE if space == "adobe-rgb-1998" else _SRGB
    xyz = lin @ m.T
    return xyz @ _adapt(np.array(_WHITE["d65"]), wp).T


def _xyz_to_rgb(xyz: np.ndarray, space: str, wp: np.ndarray) -> np.ndarray:
    m = _ADOBE if space == "adobe-rgb-1998" else _SRGB
    xyz = xyz @ _adapt(wp, np.array(_WHITE["d65"])).T
    return _delinearize(xyz @ np.linalg.inv(m).T, space)


def _xyz_to_lab(xyz: np.ndarray, wp: np.ndarray) -> np.ndarray:
    t = xyz / wp
    d = 6.0 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29)
    return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])])


def _lab_to_xyz(lab: np.ndarray, wp: np.ndarray) -> np.ndarray:
    fy = (lab[:, 0] + 16) / 116
    f = np.column_stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200])
    d = 6.0 / 29
    return np.where(f > d, f ** 3, 3 * d * d * (f - 4.0 / 29)) * wp


def _out_type(opts: dict, data: np.ndarray, shape: tuple, in_dtype, name: str) -> np.ndarray:
    kind = text(opts["outputtype"]).lower() if "outputtype" in opts else None
    out = data.reshape(shape)
    if kind is None:
        return out.astype(np.float32) if np.dtype(in_dtype) == np.float32 else out
    if kind in ("double", "single"):
        return out.astype(kind == "single" and np.float32 or np.float64)
    if kind in ("uint8", "uint16"):
        return _from_unit(np.clip(out, 0, 1), np.dtype(kind))
    raise MatlabError(f"{name}: 'OutputType' must be 'double', 'single', 'uint8' or 'uint16'.")


@builtin("rgb2xyz")
def _rgb2xyz(interp, args, nargout):
    rest, space, wp, _o = _space_opts(args, "rgb2xyz")
    nargs(rest, 1, 1, "rgb2xyz")
    rgb, shape, dt = _color_in(rest[0], "rgb2xyz")
    return _color_out(_rgb_to_xyz(rgb, space, wp), shape, dt)


@builtin("xyz2rgb")
def _xyz2rgb(interp, args, nargout):
    rest, space, wp, opts = _space_opts(args, "xyz2rgb", ("OutputType",))
    nargs(rest, 1, 1, "xyz2rgb")
    xyz, shape, dt = _color_in(rest[0], "xyz2rgb", scale=False)
    return _out_type(opts, _xyz_to_rgb(xyz, space, wp), shape, dt, "xyz2rgb")


@builtin("rgb2lab")
def _rgb2lab(interp, args, nargout):
    rest, space, wp, _o = _space_opts(args, "rgb2lab")
    nargs(rest, 1, 1, "rgb2lab")
    rgb, shape, dt = _color_in(rest[0], "rgb2lab")
    return _color_out(_xyz_to_lab(_rgb_to_xyz(rgb, space, wp), wp), shape, dt)


@builtin("lab2rgb")
def _lab2rgb(interp, args, nargout):
    rest, space, wp, opts = _space_opts(args, "lab2rgb", ("OutputType",))
    nargs(rest, 1, 1, "lab2rgb")
    lab, shape, dt = _color_in(rest[0], "lab2rgb", scale=False)
    return _out_type(opts, _xyz_to_rgb(_lab_to_xyz(lab, wp), space, wp), shape, dt, "lab2rgb")


@builtin("xyz2lab")
def _xyz2lab(interp, args, nargout):
    rest, opts = _options(list(args), ("WhitePoint",), "xyz2lab")
    nargs(rest, 1, 1, "xyz2lab")
    xyz, shape, dt = _color_in(rest[0], "xyz2lab", scale=False)
    return _color_out(_xyz_to_lab(xyz, _white(opts.get("whitepoint"), "xyz2lab")), shape, dt)


@builtin("lab2xyz")
def _lab2xyz(interp, args, nargout):
    rest, opts = _options(list(args), ("WhitePoint",), "lab2xyz")
    nargs(rest, 1, 1, "lab2xyz")
    lab, shape, dt = _color_in(rest[0], "lab2xyz", scale=False)
    return _color_out(_lab_to_xyz(lab, _white(opts.get("whitepoint"), "lab2xyz")), shape, dt)


_YCBCR = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
_YCBCR_OFF = np.array([16.0, 128.0, 128.0])


@builtin("rgb2ycbcr")
def _rgb2ycbcr(interp, args, nargout):
    nargs(args, 1, 1, "rgb2ycbcr")
    rgb, shape, dt = _color_in(args[0], "rgb2ycbcr")
    y = rgb @ _YCBCR.T + _YCBCR_OFF          # on the 0..255 scale
    if np.dtype(dt).kind in "iu":
        scale = (float(np.iinfo(dt).max)) / 255.0
        return _cast(y * scale, dt).reshape(shape)
    return _color_out(y / 255.0, shape, dt)


@builtin("ycbcr2rgb")
def _ycbcr2rgb(interp, args, nargout):
    nargs(args, 1, 1, "ycbcr2rgb")
    ycc, shape, dt = _color_in(args[0], "ycbcr2rgb")
    rgb = (ycc * 255.0 - _YCBCR_OFF) @ np.linalg.inv(_YCBCR).T
    if np.dtype(dt).kind in "iu":
        return _from_unit(np.clip(rgb, 0, 1), dt).reshape(shape)
    return _color_out(np.clip(rgb, 0, 1), shape, dt)


_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


@builtin("rgb2ntsc")
def _rgb2ntsc(interp, args, nargout):
    nargs(args, 1, 1, "rgb2ntsc")
    rgb, shape, _dt = _color_in(args[0], "rgb2ntsc")
    return (rgb @ _YIQ.T).reshape(shape)


@builtin("ntsc2rgb")
def _ntsc2rgb(interp, args, nargout):
    nargs(args, 1, 1, "ntsc2rgb")
    yiq, shape, _dt = _color_in(args[0], "ntsc2rgb", scale=False)
    return np.clip(yiq @ np.linalg.inv(_YIQ).T, 0, 1).reshape(shape)


# -- contrast and thresholds ----------------------------------------------------------------------------

@builtin("imhistmatch")
def _imhistmatch(interp, args, nargout):
    rest, opts = _options(list(args), ("Method",), "imhistmatch")
    nargs(rest, 2, 3, "imhistmatch")
    a, ref = _img(rest[0], "imhistmatch"), _img(rest[1], "imhistmatch", "reference image")
    nbins = to_int(rest[2], "number of bins") if len(rest) > 2 else 64
    if nbins < 2:
        raise MatlabError("imhistmatch: the number of bins must be at least 2.")
    if "method" in opts and text(opts["method"]).lower() not in ("uniform", "polynomial"):
        raise MatlabError("imhistmatch: 'Method' must be 'uniform' or 'polynomial'.")
    if "method" in opts and text(opts["method"]).lower() == "polynomial":
        raise MatlabError("imhistmatch: the 'polynomial' method is not supported.")
    ca = a.shape[2] if a.ndim == 3 else 1
    cr = ref.shape[2] if ref.ndim == 3 else 1
    if cr not in (1, ca):
        raise MatlabError("imhistmatch: the reference must be grayscale or have as many channels as the image.")
    out = np.empty(a.shape, dtype=a.dtype)
    hgrams = []
    for k in range(ca):
        plane = a[..., k] if a.ndim == 3 else a
        rplane = ref[..., min(k, cr - 1)] if ref.ndim == 3 else ref
        counts, _ = _hist_counts(rplane, nbins)
        hgrams.append(counts)
        res = BUILTINS["histeq"](interp, [plane, counts.reshape(1, -1).astype(float)], 1)
        if a.ndim == 3:
            out[..., k] = res
        else:
            out = res
    if nargout > 1:
        return out, np.array(hgrams, dtype=float)
    return out


def _clip_hist(h: np.ndarray, limit: float) -> np.ndarray:
    """Clip a histogram and spread the excess evenly over all bins (repeated until nothing overflows)."""
    h = h.astype(float).copy()
    for _ in range(100):
        over = np.maximum(h - limit, 0)
        excess = over.sum()
        if excess < 1e-9:
            break
        h = np.minimum(h, limit)
        room = h < limit
        if not np.any(room):
            h += excess / h.size
            break
        h[room] += excess / room.sum()
    return h


@builtin("adapthisteq")
def _adapthisteq(interp, args, nargout):
    rest, opts = _options(list(args), ("NumTiles", "ClipLimit", "NBins", "Range", "Distribution", "Alpha"),
                          "adapthisteq")
    nargs(rest, 1, 1, "adapthisteq")
    a = _img(rest[0], "adapthisteq")
    if a.ndim != 2 or a.dtype == bool:
        raise MatlabError("adapthisteq: the image must be a 2-D grayscale image.")
    tiles = _vec(opts["numtiles"], "NumTiles").astype(int) if "numtiles" in opts else np.array([8, 8])
    if tiles.size != 2 or np.any(tiles < 2):
        raise MatlabError("adapthisteq: 'NumTiles' must be [rows cols] with at least two tiles each way.")
    clip = _scalar(opts["cliplimit"], "ClipLimit") if "cliplimit" in opts else 0.01
    if not 0 <= clip <= 1:
        raise MatlabError("adapthisteq: 'ClipLimit' must lie between 0 and 1.")
    nbins = to_int(opts["nbins"], "NBins") if "nbins" in opts else 256
    dist = text(opts["distribution"]).lower() if "distribution" in opts else "uniform"
    if dist not in ("uniform", "rayleigh", "exponential"):
        raise MatlabError("adapthisteq: 'Distribution' must be 'uniform', 'rayleigh' or 'exponential'.")
    alpha = _scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.4
    rng_opt = text(opts["range"]).lower() if "range" in opts else "full"
    if rng_opt not in ("full", "original"):
        raise MatlabError("adapthisteq: 'Range' must be 'full' or 'original'.")
    u = np.clip(_to_unit(a), 0, 1)
    lo, hi = (0.0, 1.0) if rng_opt == "full" else (float(u.min()), float(u.max()))
    rows, cols = u.shape
    tr, tc = int(tiles[0]), int(tiles[1])
    th, tw = int(np.ceil(rows / tr)), int(np.ceil(cols / tc))
    th += th % 2
    tw += tw % 2
    padded = np.pad(u, ((0, th * tr - rows), (0, tw * tc - cols)), mode="symmetric")
    idx = np.minimum((padded * nbins).astype(int), nbins - 1)
    npix = th * tw
    minclip = np.ceil(npix / nbins)
    limit = minclip + np.round(clip * (npix - minclip))
    maps = np.empty((tr, tc, nbins))
    for i in range(tr):
        for j in range(tc):
            h = np.bincount(idx[i * th:(i + 1) * th, j * tw:(j + 1) * tw].ravel(), minlength=nbins)
            cdf = np.cumsum(_clip_hist(h, limit)) / npix
            if dist == "uniform":
                m = lo + (hi - lo) * cdf
            elif dist == "rayleigh":
                with np.errstate(divide="ignore"):
                    m = lo + np.sqrt(-2 * alpha ** 2 * np.log(np.maximum(1 - cdf, 1e-12)))
            else:
                with np.errstate(divide="ignore"):
                    m = lo - np.log(np.maximum(1 - cdf, 1e-12)) / alpha
            maps[i, j] = np.clip(m, lo, hi)
    # blend the mappings of the four nearest tile centres
    rr = (np.arange(padded.shape[0]) + 0.5) / th - 0.5
    cc = (np.arange(padded.shape[1]) + 0.5) / tw - 0.5
    r0 = np.clip(np.floor(rr).astype(int), 0, tr - 1)
    c0 = np.clip(np.floor(cc).astype(int), 0, tc - 1)
    r1 = np.clip(r0 + 1, 0, tr - 1)
    c1 = np.clip(c0 + 1, 0, tc - 1)
    fr = np.clip(rr - np.floor(rr), 0, 1)
    fc = np.clip(cc - np.floor(cc), 0, 1)
    fr = np.where(rr < 0, 0.0, np.where(rr > tr - 1, 0.0, fr))
    fc = np.where(cc < 0, 0.0, np.where(cc > tc - 1, 0.0, fc))
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.meshgrid(r1, c1, indexing="ij")
    FR, FC = np.meshgrid(fr, fc, indexing="ij")
    out = ((1 - FR) * (1 - FC) * maps[R0, C0, idx] + (1 - FR) * FC * maps[R0, C1, idx]
           + FR * (1 - FC) * maps[R1, C0, idx] + FR * FC * maps[R1, C1, idx])
    return _from_unit(out[:rows, :cols], a.dtype)


def _multi_otsu(counts: np.ndarray, n: int) -> tuple[list[int], float]:
    """Bin indices (last bin of each lower class) that maximise the between-class variance."""
    p = counts / counts.sum()
    L = p.size
    w = np.concatenate([[0.0], np.cumsum(p)])
    m = np.concatenate([[0.0], np.cumsum(p * np.arange(L))])
    mu_t = m[-1]

    S = np.zeros((L + 1, L + 1))
    for i in range(L):
        ww = w[i + 1:] - w[i]
        mm = m[i + 1:] - m[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            S[i, i + 1:] = np.where(ww > 0, mm * mm / np.where(ww > 0, ww, 1.0), 0.0)
    best = S[0].copy()               # best[j]: one class covering bins 0..j-1
    back = []
    for _k in range(n):
        cand = best[:, None] + S     # split point i then class i..j-1
        cand[np.tril_indices(L + 1)] = -np.inf
        arg = np.argmax(cand, axis=0)
        best = cand[arg, np.arange(L + 1)]
        back.append(arg)
    cuts = []
    j = L
    for arg in reversed(back):
        j = int(arg[j])
        cuts.append(j - 1)
    cuts = cuts[::-1]
    sigma_b = float(best[L] - mu_t ** 2)
    sigma_t = float(np.sum(p * (np.arange(L) - mu_t) ** 2))
    return cuts, (sigma_b / sigma_t if sigma_t > 0 else 0.0)


@builtin("multithresh")
def _multithresh(interp, args, nargout):
    nargs(args, 1, 2, "multithresh")
    a = _img(args[0], "multithresh")
    n = to_int(args[1], "number of thresholds") if len(args) > 1 else 1
    if not 1 <= n <= 20:
        raise MatlabError("multithresh: the number of thresholds must be between 1 and 20.")
    vals = a[np.isfinite(a)] if a.dtype.kind == "f" else a.ravel()
    if vals.size == 0:
        raise MatlabError("multithresh: the image has no finite values.")
    lo, hi = float(vals.min()), float(vals.max())
    nbins = 256
    if hi == lo:
        t = np.full(n, lo)
        out = _cast(t.reshape(1, -1), a.dtype) if a.dtype.kind in "iu" else t.reshape(1, -1).astype(a.dtype)
        return (out, 0.0) if nargout > 1 else out
    scaled = (vals.astype(float) - lo) / (hi - lo)
    counts = np.bincount(np.minimum(np.floor(scaled * (nbins - 1) + 0.5).astype(int), nbins - 1),
                         minlength=nbins).astype(float)
    cuts, metric = _multi_otsu(counts, n)
    t = lo + np.array(cuts, dtype=float) * (hi - lo) / (nbins - 1)
    if a.dtype.kind in "iu":
        out = _cast(t.reshape(1, -1), a.dtype)
    elif a.dtype == bool:
        out = t.reshape(1, -1)
    else:
        out = t.reshape(1, -1).astype(a.dtype)
    return (out, metric) if nargout > 1 else out


@builtin("imquantize")
def _imquantize(interp, args, nargout):
    nargs(args, 2, 3, "imquantize")
    a = _img(args[0], "imquantize")
    levels = _vec(args[1], "levels")
    if np.any(np.diff(levels) < 0):
        raise MatlabError("imquantize: the levels must be in increasing order.")
    idx = np.searchsorted(levels, a.astype(float), side="left")     # values equal to a level stay below it
    if len(args) > 2:
        values = args[2]
        v = _img(values, "imquantize", "values").flatten(order="F")
        if v.size != levels.size + 1:
            raise MatlabError("imquantize: there must be one more value than there are levels.")
        out = v[idx]
        return (out, (idx + 1).astype(float)) if nargout > 1 else out
    return (idx + 1).astype(float)


def _box_mean(g: np.ndarray, size: tuple[int, int], mode: str = "nearest") -> np.ndarray:
    return _nd().uniform_filter(g.astype(float), size=size, mode=mode)


@builtin("adaptthresh")
def _adaptthresh(interp, args, nargout):
    rest, opts = _options(list(args), ("NeighborhoodSize", "Statistic", "ForegroundPolarity"), "adaptthresh")
    nargs(rest, 1, 2, "adaptthresh")
    a = _img(rest[0], "adaptthresh")
    if a.ndim != 2:
        raise MatlabError("adaptthresh: the image must be 2-D.")
    sens = _scalar(rest[1], "sensitivity") if len(rest) > 1 else 0.5
    if not 0 <= sens <= 1:
        raise MatlabError("adaptthresh: the sensitivity must lie between 0 and 1.")
    g = _to_unit(a)
    if "neighborhoodsize" in opts:
        ns = _vec(opts["neighborhoodsize"], "NeighborhoodSize").astype(int)
        ns = (int(ns[0]), int(ns[0])) if ns.size == 1 else (int(ns[0]), int(ns[1]))
        if ns[0] % 2 == 0 or ns[1] % 2 == 0:
            raise MatlabError("adaptthresh: the neighbourhood size must be odd.")
    else:
        ns = (2 * (a.shape[0] // 16) + 1, 2 * (a.shape[1] // 16) + 1)
    stat = text(opts["statistic"]).lower() if "statistic" in opts else "mean"
    pol = text(opts["foregroundpolarity"]).lower() if "foregroundpolarity" in opts else "bright"
    if pol not in ("bright", "dark"):
        raise MatlabError("adaptthresh: 'ForegroundPolarity' must be 'bright' or 'dark'.")
    nd = _nd()
    if stat == "mean":
        local = _box_mean(g, ns)
    elif stat == "median":
        local = nd.median_filter(g, size=ns, mode="nearest")
    elif stat == "gaussian":
        sig = (ns[0] - 1) / 4, (ns[1] - 1) / 4
        local = nd.gaussian_filter(g, sigma=[max(s, 1e-6) for s in sig], mode="nearest", truncate=2.0)
    else:
        raise MatlabError("adaptthresh: 'Statistic' must be 'mean', 'median' or 'gaussian'.")
    scale = 0.6 + (1 - sens) if pol == "bright" else 0.4 + sens
    return np.clip(scale * local, 0, 1)


# -- local filters --------------------------------------------------------------------------------------

def _odd_size(v, default: int, name: str) -> tuple[int, int]:
    if v is None:
        return default, default
    s = _vec(v, "filter size").astype(int)
    s = (int(s[0]), int(s[0])) if s.size == 1 else (int(s[0]), int(s[1]))
    if s[0] < 1 or s[1] < 1 or s[0] % 2 == 0 or s[1] % 2 == 0:
        raise MatlabError(f"{name}: the filter size must be odd and positive.")
    return s


def _pad_spec(v, name: str):
    if v is None:
        return "replicate"
    if is_text(v):
        key = text(v).lower()
        if key not in ("replicate", "symmetric", "circular"):
            raise MatlabError(f"{name}: 'Padding' must be 'replicate', 'symmetric', 'circular' or a number.")
        return key
    return _scalar(v, "Padding")


@builtin("imboxfilt")
def _imboxfilt(interp, args, nargout):
    rest, opts = _options(list(args), ("Padding", "NormalizationFactor"), "imboxfilt")
    nargs(rest, 1, 2, "imboxfilt")
    a = _img(rest[0], "imboxfilt")
    size = _odd_size(rest[1] if len(rest) > 1 else None, 3, "imboxfilt")
    how = _pad_spec(opts.get("padding"), "imboxfilt")
    norm = _scalar(opts["normalizationfactor"], "NormalizationFactor") if "normalizationfactor" in opts \
        else 1.0 / (size[0] * size[1])
    h = np.ones(size) * norm
    out = _planes(a.astype(float), lambda p: _correlate(p, h, how, "same", False, "imboxfilt"))
    return _like_class(out, a)


def _gauss_blur(p: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    k = _gauss1d(sigma, radius)
    nd = _nd()
    return nd.correlate1d(nd.correlate1d(p, k, axis=0, mode="nearest"), k, axis=1, mode="nearest")


@builtin("imsharpen")
def _imsharpen(interp, args, nargout):
    rest, opts = _options(list(args), ("Radius", "Amount", "Threshold"), "imsharpen")
    nargs(rest, 1, 1, "imsharpen")
    a = _img(rest[0], "imsharpen")
    radius = _scalar(opts["radius"], "Radius") if "radius" in opts else 1.0
    amount = _scalar(opts["amount"], "Amount") if "amount" in opts else 0.8
    thresh = _scalar(opts["threshold"], "Threshold") if "threshold" in opts else 0.0
    if radius <= 0 or amount < 0 or not 0 <= thresh <= 1:
        raise MatlabError("imsharpen: 'Radius' must be positive, 'Amount' non-negative and 'Threshold' in [0, 1].")
    half = int(np.ceil(2 * radius))

    def sharpen(p, top):
        blur = _gauss_blur(p, radius, half)
        diff = p - blur
        if thresh > 0:
            diff = np.where(np.abs(diff) >= thresh * top, diff, 0.0)
        return p + amount * diff

    u = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    if _is_rgb(a):
        wp = np.array(_WHITE["d65"])
        lab = _xyz_to_lab(_rgb_to_xyz(u.reshape(-1, 3), "srgb", wp), wp).reshape(u.shape)
        lab[..., 0] = np.clip(sharpen(lab[..., 0], 100.0), 0, 100)
        out = _xyz_to_rgb(_lab_to_xyz(lab.reshape(-1, 3), wp), "srgb", wp).reshape(u.shape)
    else:
        out = _planes(u, lambda p: sharpen(p, 1.0))
    if a.dtype.kind == "f":
        return out.astype(a.dtype)
    return _from_unit(np.clip(out, 0, 1), a.dtype)


def _nhood_arg(v, default: int, name: str) -> np.ndarray:
    if v is None:
        return np.ones((default, default), dtype=bool)
    h = _img(v, name, "neighbourhood") != 0
    if any(s % 2 == 0 for s in h.shape):
        raise MatlabError(f"{name}: the neighbourhood must have odd dimensions.")
    return h


@builtin("stdfilt")
def _stdfilt(interp, args, nargout):
    nargs(args, 1, 2, "stdfilt")
    a = _img(args[0], "stdfilt")
    h = _nhood_arg(args[1] if len(args) > 1 else None, 3, "stdfilt").astype(float)
    n = h.sum()
    if n < 2:
        return np.zeros(a.shape)
    x = a.astype(float)

    def run(p):
        s1 = _correlate(p, h, "symmetric", "same", False, "stdfilt")
        s2 = _correlate(p * p, h, "symmetric", "same", False, "stdfilt")
        return np.sqrt(np.maximum(s2 - s1 * s1 / n, 0) / (n - 1))

    return _planes(x, run)


@builtin("rangefilt")
def _rangefilt(interp, args, nargout):
    nargs(args, 1, 2, "rangefilt")
    a = _img(args[0], "rangefilt")
    h = _nhood_arg(args[1] if len(args) > 1 else None, 3, "rangefilt")
    nd = _nd()
    work = a.astype(np.uint8) if a.dtype == bool else a

    def run(p):
        hi = nd.maximum_filter(p, footprint=h, mode="reflect")
        lo = nd.minimum_filter(p, footprint=h, mode="reflect")
        if p.dtype.kind in "iu":
            return (hi.astype(np.int64) - lo.astype(np.int64))
        return hi - lo

    out = _planes(work, run)
    if a.dtype == bool:
        return out.astype(np.uint8)
    if a.dtype.kind == "i":           # the range of a signed class needs the unsigned class of the same width
        return out.astype(np.dtype("u" + a.dtype.name))
    return out.astype(a.dtype)


@builtin("entropyfilt")
def _entropyfilt(interp, args, nargout):
    nargs(args, 1, 2, "entropyfilt")
    a = _img(args[0], "entropyfilt")
    h = _nhood_arg(args[1] if len(args) > 1 else None, 9, "entropyfilt").astype(float)
    if a.dtype == bool:
        levels = a.astype(np.int64)
    else:
        levels = np.clip(_round_half_away(_to_unit(a) * 255), 0, 255).astype(np.int64)
    n = h.sum()

    def run(p):
        ent = np.zeros(p.shape)
        for v in np.unique(p):
            cnt = _correlate((p == v).astype(float), h, "symmetric", "same", False, "entropyfilt")
            prob = cnt / n
            with np.errstate(divide="ignore", invalid="ignore"):
                ent -= np.where(prob > 0, prob * np.log2(prob), 0.0)
        return ent

    return _planes(levels, run)


_GRAD_KERNELS = {"sobel": np.array([[1.0, 2, 1], [0, 0, 0], [-1, -2, -1]]),
                 "prewitt": np.array([[1.0, 1, 1], [0, 0, 0], [-1, -1, -1]])}


def _gradient_xy(g: np.ndarray, method: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    if method in _GRAD_KERNELS:
        hy = -_GRAD_KERNELS[method]
        gx = _correlate(g, hy.T, "replicate", "same", False, name)
        gy = _correlate(g, hy, "replicate", "same", False, name)
        return gx, gy
    if method == "central":
        gy, gx = (np.gradient(g, axis=0) if g.shape[0] > 1 else np.zeros_like(g),
                  np.gradient(g, axis=1) if g.shape[1] > 1 else np.zeros_like(g))
        return gx, gy
    if method == "intermediate":
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, :-1] = np.diff(g, axis=1)
        gy[:-1, :] = np.diff(g, axis=0)
        return gx, gy
    raise MatlabError(f"{name}: the method must be 'sobel', 'prewitt', 'central' or 'intermediate'.")


def _grad_input(v, name: str) -> tuple[np.ndarray, type]:
    a = _img(v, name)
    if a.ndim != 2:
        raise MatlabError(f"{name}: the image must be 2-D.")
    return a.astype(float), (np.float32 if a.dtype == np.float32 else np.float64)


@builtin("imgradientxy")
def _imgradientxy(interp, args, nargout):
    nargs(args, 1, 2, "imgradientxy")
    g, cls = _grad_input(args[0], "imgradientxy")
    method = text(args[1]).lower() if len(args) > 1 else "sobel"
    gx, gy = _gradient_xy(g, method, "imgradientxy")
    return (gx.astype(cls), gy.astype(cls)) if nargout > 1 else gx.astype(cls)


@builtin("imgradient")
def _imgradient(interp, args, nargout):
    nargs(args, 1, 2, "imgradient")
    if len(args) == 2 and not is_text(args[1]):
        gx, cls = _grad_input(args[0], "imgradient")
        gy, _ = _grad_input(args[1], "imgradient")
        if gx.shape != gy.shape:
            raise MatlabError("imgradient: the two gradient images must have the same size.")
        roberts = False
    else:
        g, cls = _grad_input(args[0], "imgradient")
        method = text(args[1]).lower() if len(args) > 1 else "sobel"
        roberts = method == "roberts"
        if roberts:
            gx = _correlate(g, np.array([[1.0, 0], [0, -1]]), "replicate", "same", False, "imgradient")
            gy = _correlate(g, np.array([[0, 1.0], [-1, 0]]), "replicate", "same", False, "imgradient")
        else:
            gx, gy = _gradient_xy(g, method, "imgradient")
    mag = np.hypot(gx, gy).astype(cls)
    if nargout > 1:
        if roberts:
            ang = np.degrees(np.arctan2(gy, -gx)) - 180 / 4 * 3
            ang = np.where(ang < -180, ang + 360, ang)
            ang = np.where((gx == 0) & (gy == 0), 0.0, ang)
        else:
            ang = np.degrees(np.arctan2(-gy, gx))
        return mag, ang.astype(cls)
    return mag


@builtin("imbilatfilt")
def _imbilatfilt(interp, args, nargout):
    rest, opts = _options(list(args), ("NeighborhoodSize", "Padding"), "imbilatfilt")
    nargs(rest, 1, 3, "imbilatfilt")
    a = _img(rest[0], "imbilatfilt")
    if a.ndim > 3 or (a.ndim == 3 and a.shape[2] != 3):
        raise MatlabError("imbilatfilt: the image must be grayscale or RGB.")
    lo, hi = _class_range(a)
    degree = _scalar(rest[1], "degree of smoothing") if len(rest) > 1 and rest[1].size else 0.01 * (hi - lo) ** 2
    sigma = _scalar(rest[2], "spatial sigma") if len(rest) > 2 else 1.0
    if degree <= 0 or sigma <= 0:
        raise MatlabError("imbilatfilt: the degree of smoothing and the spatial sigma must be positive.")
    if "neighborhoodsize" in opts:
        size = _odd_size(opts["neighborhoodsize"], 3, "imbilatfilt")
    else:
        w = 2 * int(np.ceil(2 * sigma)) + 1
        size = (w, w)
    how = _pad_spec(opts.get("padding"), "imbilatfilt")
    x = a.astype(float)
    x3 = x if x.ndim == 3 else x[..., None]
    rr, rc = size[0] // 2, size[1] // 2
    padded = _pad(x3, [(rr, rr), (rc, rc)], how, "imbilatfilt")
    num = np.zeros_like(x3)
    den = np.zeros(x3.shape[:2])
    H, W = x3.shape[:2]
    for dr in range(-rr, rr + 1):
        for dc in range(-rc, rc + 1):
            nb = padded[rr + dr:rr + dr + H, rc + dc:rc + dc + W]
            d2 = np.sum((nb - x3) ** 2, axis=2)
            wgt = np.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma) - d2 / (2 * degree))
            num += wgt[..., None] * nb
            den += wgt
    out = (num / den[..., None]).reshape(x.shape)
    return _like_class(out, a)


@builtin("imguidedfilter")
def _imguidedfilter(interp, args, nargout):
    rest, opts = _options(list(args), ("NeighborhoodSize", "DegreeOfSmoothing"), "imguidedfilter")
    nargs(rest, 1, 2, "imguidedfilter")
    a = _img(rest[0], "imguidedfilter")
    guide = _img(rest[1], "imguidedfilter", "guidance image") if len(rest) > 1 else a
    if guide.shape[:2] != a.shape[:2] or guide.ndim != 2 and not _is_rgb(guide):
        raise MatlabError("imguidedfilter: the guidance image must have the size of the image.")
    if guide.ndim == 3:
        guide = _rgb2gray(guide)
    size = _odd_size(opts.get("neighborhoodsize"), 5, "imguidedfilter")
    lo, hi = _class_range(guide)
    eps_ = _scalar(opts["degreeofsmoothing"], "DegreeOfSmoothing") if "degreeofsmoothing" in opts \
        else 0.01 * (hi - lo) ** 2
    G = guide.astype(float)

    def run(p):
        mg, mp = _box_mean(G, size), _box_mean(p, size)
        cov = _box_mean(G * p, size) - mg * mp
        var = _box_mean(G * G, size) - mg * mg
        aa = cov / (var + eps_)
        bb = mp - aa * mg
        return _box_mean(aa, size) * G + _box_mean(bb, size)

    return _like_class(_planes(a.astype(float), run), a)


@builtin("integralBoxFilter")
def _integral_box_filter(interp, args, nargout):
    rest, opts = _options(list(args), ("NormalizationFactor",), "integralBoxFilter")
    nargs(rest, 1, 2, "integralBoxFilter")
    ii = _img(rest[0], "integralBoxFilter", "integral image").astype(float)
    size = _odd_size(rest[1] if len(rest) > 1 else None, 3, "integralBoxFilter")
    norm = _scalar(opts["normalizationfactor"], "NormalizationFactor") if "normalizationfactor" in opts \
        else 1.0 / (size[0] * size[1])
    m, n = size
    if ii.shape[0] <= m or ii.shape[1] <= n:
        raise MatlabError("integralBoxFilter: the filter is larger than the image.")
    out = ii[m:, n:] - ii[:-m, n:] - ii[m:, :-n] + ii[:-m, :-n]
    return out * norm


# -- block and neighbourhood operations -----------------------------------------------------------------

def _block_size(v, name: str) -> tuple[int, int]:
    s = _vec(v, "block size").astype(int)
    if s.size != 2 or np.any(s < 1):
        raise MatlabError(f"{name}: the block size must be [rows cols].")
    return int(s[0]), int(s[1])


@builtin("blockproc")
def _blockproc(interp, args, nargout):
    rest, opts = _options(list(args), ("BorderSize", "TrimBorder", "PadPartialBlocks", "PadMethod",
                                       "UseParallel", "DisplayWaitbar"), "blockproc")
    nargs(rest, 3, 3, "blockproc")
    a = rest[0]
    if not isinstance(a, np.ndarray):
        raise MatlabError("blockproc: the image must be a numeric or logical array (file names are not supported).")
    bs = _block_size(rest[1], "blockproc")
    fun = rest[2]
    if not isinstance(fun, FunctionHandle):
        raise MatlabError("blockproc: the third input must be a function handle.")
    border = _vec(opts["bordersize"], "BorderSize").astype(int) if "bordersize" in opts else np.array([0, 0])
    trim = _flag(opts["trimborder"]) if "trimborder" in opts else True
    padpartial = _flag(opts["padpartialblocks"]) if "padpartialblocks" in opts else False
    how: object = 0.0
    if "padmethod" in opts:
        pm = opts["padmethod"]
        how = text(pm).lower() if is_text(pm) else _scalar(pm, "PadMethod")
    rows, cols = a.shape[:2]
    vb, hb = int(border[0]), int(border[1])
    work = a
    if padpartial:
        work = _pad(a, [(0, (-rows) % bs[0]), (0, (-cols) % bs[1])], how, "blockproc")
    prow, pcol = work.shape[:2]
    padded = _pad(work, [(vb, vb), (hb, hb)], how, "blockproc") if (vb or hb) else work
    row_out = []
    for r0 in range(0, prow, bs[0]):
        line = []
        for c0 in range(0, pcol, bs[1]):
            r1, c1 = min(r0 + bs[0], prow), min(c0 + bs[1], pcol)
            data = padded[r0:r1 + 2 * vb, c0:c1 + 2 * hb]
            block = Struct.scalar({"border": np.array([[float(vb), float(hb)]]),
                                   "blockSize": np.array([[float(r1 - r0), float(c1 - c0)]]),
                                   "data": data,
                                   "imageSize": np.array([[float(rows), float(cols)]]),
                                   "location": np.array([[float(r0 + 1), float(c0 + 1)]])})
            res = _call(interp, fun, [block], 1)
            if not res:
                raise MatlabError("blockproc: the function must return a value.")
            out = res[0]
            if not isinstance(out, np.ndarray):
                raise MatlabError("blockproc: the function must return a numeric, logical or char array.")
            if trim and (vb or hb) and out.shape[0] > 2 * vb and out.shape[1] > 2 * hb \
                    and out.shape[:2] == data.shape[:2]:
                out = out[vb:out.shape[0] - vb, hb:out.shape[1] - hb]
            line.append(out)
        try:
            row_out.append(np.concatenate(line, axis=1))
        except ValueError:
            raise MatlabError("blockproc: the results of the blocks in one row do not fit together.")
    try:
        return np.concatenate(row_out, axis=0)
    except ValueError:
        raise MatlabError("blockproc: the results of the block rows do not fit together.")


@builtin("nlfilter")
def _nlfilter(interp, args, nargout):
    rest = [a for a in args if not (is_text(a) and text(a).lower() == "indexed")]
    nargs(rest, 3, None, "nlfilter")
    a = _img(rest[0], "nlfilter")
    if a.ndim != 2:
        raise MatlabError("nlfilter: the image must be 2-D.")
    m, n = _block_size(rest[1], "nlfilter")
    fun = rest[2]
    extra = list(rest[3:])
    top, left = (m - 1) // 2, (n - 1) // 2
    padded = np.pad(a, ((top, m - 1 - top), (left, n - 1 - left)), mode="constant")
    out = None
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            res = _call(interp, fun, [padded[r:r + m, c:c + n], *extra], 1)[0]
            val = np.asarray(res)
            if val.size != 1:
                raise MatlabError("nlfilter: the function must return a scalar.")
            if out is None:
                out = np.zeros(a.shape, dtype=val.dtype)
            out[r, c] = val.flat[0]
    return out if out is not None else np.zeros(a.shape)


def _sliding_columns(a: np.ndarray, m: int, n: int, pad: bool) -> np.ndarray:
    """Columns holding every m-by-n neighbourhood (column-major inside a neighbourhood and over positions)."""
    if pad:
        top, left = (m - 1) // 2, (n - 1) // 2
        a = np.pad(a, ((top, m - 1 - top), (left, n - 1 - left)), mode="constant")
    rows, cols = a.shape[0] - m + 1, a.shape[1] - n + 1
    out = np.empty((m * n, rows * cols), dtype=a.dtype)
    k = 0
    for dc in range(n):
        for dr in range(m):
            out[k] = a[dr:dr + rows, dc:dc + cols].ravel(order="F")
            k += 1
    return out


@builtin("colfilt")
def _colfilt(interp, args, nargout):
    rest = [a for a in args if not (is_text(a) and text(a).lower() == "indexed")]
    nargs(rest, 4, None, "colfilt")
    a = _img(rest[0], "colfilt")
    if a.ndim != 2:
        raise MatlabError("colfilt: the image must be 2-D.")
    m, n = _block_size(rest[1], "colfilt")
    pos = 2
    if not is_text(rest[pos]):
        pos += 1                      # an optional [mblock nblock] memory hint
    kind = text(rest[pos]).lower()
    fun = rest[pos + 1]
    extra = list(rest[pos + 2:])
    if kind == "sliding":
        cols = _sliding_columns(a, m, n, True)
        res = np.asarray(_call(interp, fun, [cols, *extra], 1)[0])
        if res.size != a.size:
            raise MatlabError("colfilt: a 'sliding' function must return one value per column.")
        return res.reshape(a.shape, order="F")
    if kind == "distinct":
        rows, ncol = a.shape
        pr, pc = (-rows) % m, (-ncol) % n
        padded = np.pad(a, ((0, pr), (0, pc)), mode="constant")
        cols = _distinct_columns(padded, m, n)
        res = np.asarray(_call(interp, fun, [cols, *extra], 1)[0])
        if res.shape != cols.shape:
            raise MatlabError("colfilt: a 'distinct' function must return a matrix of the same size.")
        return _columns_to_image(res, m, n, padded.shape)[:rows, :ncol]
    raise MatlabError("colfilt: the block type must be 'sliding' or 'distinct'.")


def _distinct_columns(a: np.ndarray, m: int, n: int) -> np.ndarray:
    rows, cols = a.shape
    br, bc = rows // m, cols // n
    blocks = a.reshape(br, m, bc, n).transpose(1, 3, 0, 2)       # (m, n, block row, block col)
    return blocks.reshape(m * n, br * bc, order="F")


def _columns_to_image(cols: np.ndarray, m: int, n: int, shape: tuple[int, int]) -> np.ndarray:
    br, bc = shape[0] // m, shape[1] // n
    blocks = cols.reshape((m, n, br, bc), order="F")
    return blocks.transpose(2, 0, 3, 1).reshape(shape)


@builtin("col2im")
def _col2im(interp, args, nargout):
    nargs(args, 3, 4, "col2im")
    b = args[0]
    if not isinstance(b, np.ndarray) or b.ndim != 2:
        raise MatlabError("col2im: the first input must be a matrix.")
    m, n = _block_size(args[1], "col2im")
    size = _vec(args[2], "image size").astype(int)
    kind = text(args[3]).lower() if len(args) > 3 else "sliding"
    rows, cols = int(size[0]), int(size[1])
    if kind == "distinct":
        if rows % m or cols % n or b.shape != (m * n, (rows // m) * (cols // n)):
            raise MatlabError("col2im: the matrix does not match the block and image sizes.")
        return _columns_to_image(b, m, n, (rows, cols))
    if kind == "sliding":
        if b.size != (rows - m + 1) * (cols - n + 1):
            raise MatlabError("col2im: a 'sliding' rearrangement needs one value per neighbourhood.")
        return b.reshape((rows - m + 1, cols - n + 1), order="F")
    raise MatlabError("col2im: the block type must be 'sliding' or 'distinct'.")


# -- test images ----------------------------------------------------------------------------------------

@builtin("checkerboard")
def _checkerboard(interp, args, nargout):
    nargs(args, 0, 3, "checkerboard")
    n = to_int(args[0], "square size") if len(args) > 0 else 10
    p = to_int(args[1], "rows") if len(args) > 1 else 4
    q = to_int(args[2], "columns") if len(args) > 2 else p
    if n < 1 or p < 1 or q < 1:
        raise MatlabError("checkerboard: the sizes must be positive integers.")
    black, white = np.zeros((n, n)), np.ones((n, n))
    tile = np.block([[black, white], [white, black]])
    board = np.tile(tile, (p, q))
    ncols = board.shape[1]
    right = board[:, ncols // 2:]
    right[right > 0] = 0.7            # the light squares of the right half are gray
    return board


_SHEPP = [(0.69, 0.92, 0.0, 0.0, 0.0), (0.6624, 0.8740, 0.0, -0.0184, 0.0), (0.11, 0.31, 0.22, 0.0, -18.0),
          (0.16, 0.41, -0.22, 0.0, 18.0), (0.21, 0.25, 0.0, 0.35, 0.0), (0.046, 0.046, 0.0, 0.1, 0.0),
          (0.046, 0.046, 0.0, -0.1, 0.0), (0.046, 0.023, -0.08, -0.605, 0.0), (0.023, 0.023, 0.0, -0.606, 0.0),
          (0.023, 0.046, 0.06, -0.605, 0.0)]
_SHEPP_GRAY = {"shepp-logan": [1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
               "modified shepp-logan": [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}


@builtin("phantom")
def _phantom(interp, args, nargout):
    nargs(args, 0, 2, "phantom")
    rest = list(args)
    ell = None
    name = "modified shepp-logan"
    if rest and is_text(rest[0]):
        name = text(rest.pop(0)).lower()
        if name not in _SHEPP_GRAY:
            raise MatlabError("phantom: the name must be 'Shepp-Logan' or 'Modified Shepp-Logan'.")
    elif rest and isinstance(rest[0], np.ndarray) and rest[0].ndim == 2 and rest[0].shape[1] == 6:
        ell = rest.pop(0).astype(float)
    n = to_int(rest[0], "size") if rest else 256
    if n < 1:
        raise MatlabError("phantom: the size must be a positive integer.")
    if ell is None:
        ell = np.array([(g,) + e for g, e in zip(_SHEPP_GRAY[name], _SHEPP)], dtype=float)
    ax = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2) if n > 1 else np.zeros(1)
    xg = np.tile(ax, (n, 1))
    yg = np.rot90(xg)
    p = np.zeros((n, n))
    for gray, a, b, x0, y0, phi in ell:
        t = np.deg2rad(phi)
        x, y = xg - x0, yg - y0
        inside = ((x * np.cos(t) + y * np.sin(t)) ** 2 / a ** 2 + (y * np.cos(t) - x * np.sin(t)) ** 2 / b ** 2) <= 1
        p[inside] += gray
    return (p, ell) if nargout > 1 else p


# -- matching and features --------------------------------------------------------------------------------

@builtin("normxcorr2")
def _normxcorr2(interp, args, nargout):
    nargs(args, 2, 2, "normxcorr2")
    t = _img(args[0], "normxcorr2", "template").astype(float)
    a = _img(args[1], "normxcorr2").astype(float)
    if t.ndim != 2 or a.ndim != 2:
        raise MatlabError("normxcorr2: the template and the image must be 2-D.")
    if t.shape[0] > a.shape[0] or t.shape[1] > a.shape[1]:
        raise MatlabError("normxcorr2: the template must not be larger than the image.")
    if t.size < 2 or np.all(t == t.flat[0]):
        raise MatlabError("normxcorr2: the values of the template must not all be the same.")
    from scipy.signal import fftconvolve
    m, n = t.shape
    xcorr = fftconvolve(a, t[::-1, ::-1], mode="full")
    if np.all(a == np.round(a)) and np.all(t == np.round(t)):
        xcorr = np.round(xcorr)

    def local_sum(z):
        s = np.cumsum(np.pad(z, ((m, m - 1), (n, n - 1))), axis=0)
        s = s[m:] - s[:-m]
        s = np.cumsum(s, axis=1)
        return s[:, n:] - s[:, :-n]

    ls, ls2 = local_sum(a), local_sum(a * a)
    mn = m * n
    num = xcorr - ls * t.sum() / mn
    var_a = np.maximum(ls2 - ls ** 2 / mn, 0)
    den = np.sqrt(var_a) * np.sqrt((mn - 1)) * t.std(ddof=1)
    out = np.zeros(num.shape)
    tol = np.sqrt(np.finfo(float).eps * max(float(np.max(np.abs(den))), 1e-300))
    ok = den > tol
    out[ok] = num[ok] / den[ok]
    out[np.abs(out) > 1 + 1e-9] = 0.0
    return np.clip(out, -1, 1)


@builtin("corner")
def _corner(interp, args, nargout):
    rest, opts = _options(list(args), ("QualityLevel", "SensitivityFactor", "FilterCoefficients"), "corner")
    nargs(rest, 1, 3, "corner")
    a = _img(rest[0], "corner")
    if a.ndim != 2:
        raise MatlabError("corner: the image must be 2-D grayscale.")
    method = "harris"
    nmax = 200
    for extra in rest[1:]:
        if is_text(extra):
            method = text(extra).lower()
        else:
            nmax = to_int(extra, "number of corners")
    if method not in ("harris", "minimumeigenvalue"):
        raise MatlabError("corner: the method must be 'Harris' or 'MinimumEigenvalue'.")
    quality = _scalar(opts["qualitylevel"], "QualityLevel") if "qualitylevel" in opts else 0.01
    k = _scalar(opts["sensitivityfactor"], "SensitivityFactor") if "sensitivityfactor" in opts else 0.04
    if "filtercoefficients" in opts:
        w = _vec(opts["filtercoefficients"], "FilterCoefficients")
    else:
        x = np.arange(-2, 3, dtype=float)
        w = np.exp(-x * x / (2 * 1.5 ** 2))
        w /= w.sum()
    g = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    nd = _nd()
    d = np.array([-1.0, 0.0, 1.0])
    ix = nd.correlate1d(g, d, axis=1, mode="constant")
    iy = nd.correlate1d(g, d, axis=0, mode="constant")
    ix, iy = ix[1:-1, 1:-1], iy[1:-1, 1:-1]            # drop the border the derivative cannot see

    def smooth(z):
        return nd.correlate1d(nd.correlate1d(z, w, axis=0, mode="constant"), w, axis=1, mode="constant")

    aa, bb, cc = smooth(ix * ix), smooth(iy * iy), smooth(ix * iy)
    if method == "harris":
        resp = aa * bb - cc * cc - k * (aa + bb) ** 2
    else:
        resp = (aa + bb - np.sqrt((aa - bb) ** 2 + 4 * cc * cc)) / 2
    full = np.zeros(g.shape)
    full[1:-1, 1:-1] = resp
    peak = float(full.max()) if full.size else 0.0
    if peak <= 0:
        return np.zeros((0, 2))
    is_max = _regional_max(full, np.ones((3, 3), dtype=bool)) & (full > quality * peak)
    half = w.size // 2
    is_max[:half + 1, :] = False
    is_max[-(half + 1):, :] = False
    is_max[:, :half + 1] = False
    is_max[:, -(half + 1):] = False
    rr, cc_ = np.nonzero(is_max)
    order = np.argsort(-full[rr, cc_], kind="stable")[:nmax]
    return np.column_stack([cc_[order] + 1, rr[order] + 1]).astype(float)


@builtin("hough")
def _hough(interp, args, nargout):
    rest, opts = _options(list(args), ("RhoResolution", "Theta", "ThetaResolution"), "hough")
    nargs(rest, 1, 1, "hough")
    bw = _img(rest[0], "hough")
    if bw.ndim != 2:
        raise MatlabError("hough: the image must be 2-D.")
    res = _scalar(opts["rhoresolution"], "RhoResolution") if "rhoresolution" in opts else 1.0
    if "theta" in opts:
        theta = _vec(opts["theta"], "Theta")
    elif "thetaresolution" in opts:
        step = _scalar(opts["thetaresolution"], "ThetaResolution")
        theta = np.arange(-90.0, 90.0, step)
    else:
        theta = np.arange(-90.0, 90.0)
    if res <= 0 or np.any(theta < -90) or np.any(theta >= 90):
        raise MatlabError("hough: 'RhoResolution' must be positive and 'Theta' must lie in [-90, 90).")
    rows, cols = bw.shape
    diag = np.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)
    q = int(np.ceil(diag / res))
    nrho = 2 * q + 1
    rho = np.linspace(-q * res, q * res, nrho)
    r, c = np.nonzero(bw)
    H = np.zeros((nrho, theta.size))
    t = np.deg2rad(theta)
    cos_t, sin_t = np.cos(t), np.sin(t)
    slope = (nrho - 1) / (rho[-1] - rho[0]) if nrho > 1 else 0.0
    for j in range(theta.size):
        val = c * cos_t[j] + r * sin_t[j]
        bins = np.floor(slope * (val - rho[0]) + 0.5).astype(int)
        H[:, j] = np.bincount(bins, minlength=nrho)[:nrho]
    out = (H, theta.reshape(1, -1), rho.reshape(1, -1))
    return out[:nargout] if nargout > 1 else H


@builtin("houghpeaks")
def _houghpeaks(interp, args, nargout):
    rest, opts = _options(list(args), ("Threshold", "NHoodSize", "Theta"), "houghpeaks")
    nargs(rest, 1, 2, "houghpeaks")
    H = _img(rest[0], "houghpeaks", "Hough matrix").astype(float)
    if H.ndim != 2:
        raise MatlabError("houghpeaks: the Hough matrix must be 2-D.")
    npeaks = to_int(rest[1], "number of peaks") if len(rest) > 1 else 1
    thresh = _scalar(opts["threshold"], "Threshold") if "threshold" in opts else 0.5 * float(H.max(initial=0.0))
    if "nhoodsize" in opts:
        nh = _vec(opts["nhoodsize"], "NHoodSize").astype(int)
        if nh.size != 2 or np.any(nh % 2 == 0) or np.any(nh < 1):
            raise MatlabError("houghpeaks: 'NHoodSize' must be two odd positive integers.")
    else:
        nh = np.maximum(2 * np.ceil(np.array(H.shape) / 100.0).astype(int) + 1, 1)
    work = H.copy()
    peaks = []
    nr, nt = H.shape
    hr, ht = int(nh[0]) // 2, int(nh[1]) // 2
    while len(peaks) < npeaks:
        idx = int(np.argmax(work.ravel(order="F")))
        j, i = divmod(idx, nr)
        if work[i, j] < thresh or work[i, j] <= 0:
            break
        peaks.append((i + 1, j + 1))
        for dj in range(-ht, ht + 1):
            jj = j + dj
            flip = False
            if jj < 0:
                jj += nt
                flip = True          # theta wraps round with rho changing sign
            elif jj >= nt:
                jj -= nt
                flip = True
            for di in range(-hr, hr + 1):
                ii = i + di
                if flip:
                    ii = nr - 1 - ii
                if 0 <= ii < nr:
                    work[ii, jj] = 0
    return np.array(peaks, dtype=float).reshape(-1, 2)


@builtin("houghlines")
def _houghlines(interp, args, nargout):
    rest, opts = _options(list(args), ("FillGap", "MinLength"), "houghlines")
    nargs(rest, 4, 4, "houghlines")
    bw = _img(rest[0], "houghlines")
    theta, rho = _vec(rest[1], "theta"), _vec(rest[2], "rho")
    peaks = _img(rest[3], "houghlines", "peaks").astype(int).reshape(-1, 2)
    fill = _scalar(opts["fillgap"], "FillGap") if "fillgap" in opts else 20.0
    minlen = _scalar(opts["minlength"], "MinLength") if "minlength" in opts else 40.0
    r, c = np.nonzero(bw.T)          # column-major order of the pixels
    r, c = c, r
    nrho = rho.size
    slope = (nrho - 1) / (rho[-1] - rho[0]) if nrho > 1 else 0.0
    items = []
    for pr, pt in peaks:
        th = np.deg2rad(theta[pt - 1])
        val = c * np.cos(th) + r * np.sin(th)
        bins = np.floor(slope * (val - rho[0]) + 0.5).astype(int)
        sel = bins == pr - 1
        if not np.any(sel):
            continue
        pr_, pc_ = r[sel], c[sel]
        # walk along the dominant direction of the line
        span_r, span_c = np.ptp(pr_), np.ptp(pc_)
        order = np.lexsort((pc_, pr_)) if span_r > span_c else np.lexsort((pr_, pc_))
        pr_, pc_ = pr_[order], pc_[order]
        gaps = np.hypot(np.diff(pr_), np.diff(pc_))
        breaks = np.concatenate([[0], np.flatnonzero(gaps > fill) + 1, [pr_.size]])
        for b0, b1 in zip(breaks[:-1], breaks[1:]):
            p1 = (pc_[b0] + 1, pr_[b0] + 1)
            p2 = (pc_[b1 - 1] + 1, pr_[b1 - 1] + 1)
            if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= minlen:
                items.append({"point1": np.array([p1], dtype=float), "point2": np.array([p2], dtype=float),
                              "theta": _f(theta[pt - 1]), "rho": _f(rho[pr - 1])})
    data = np.empty((1, len(items)), dtype=object)
    for i, it in enumerate(items):
        data[0, i] = it
    return Struct(data, ["point1", "point2", "theta", "rho"])


@builtin("radon")
def _radon(interp, args, nargout):
    nargs(args, 1, 3, "radon")
    a = _img(args[0], "radon").astype(float)
    if a.ndim != 2:
        raise MatlabError("radon: the image must be 2-D.")
    theta = _vec(args[1], "theta") if len(args) > 1 and args[1].size else np.arange(0.0, 180.0)
    rows, cols = a.shape
    cr, cc = (rows - 1) // 2, (cols - 1) // 2          # zero-based centre pixel
    rsize = 2 * int(np.ceil(np.hypot(rows - cr - 1, cols - cc - 1))) + 3
    half = (rsize - 1) // 2
    xp = np.arange(-half, half + 1, dtype=float)
    R = np.zeros((rsize, theta.size))
    rr, cc_idx = np.nonzero(a)
    vals = a[rr, cc_idx] / 4.0
    x = (cc_idx - cc).astype(float)
    y = (cr - rr).astype(float)
    # every pixel is split into four sub-pixels whose values are shared between the two nearest bins
    xs = np.concatenate([x - 0.25, x + 0.25, x - 0.25, x + 0.25])
    ys = np.concatenate([y - 0.25, y - 0.25, y + 0.25, y + 0.25])
    vs = np.tile(vals, 4)
    for j, th in enumerate(np.deg2rad(theta)):
        pos = xs * np.cos(th) + ys * np.sin(th) + half
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        R[:, j] = (np.bincount(lo, weights=vs * (1 - frac), minlength=rsize + 1)[:rsize]
                   + np.bincount(lo + 1, weights=vs * frac, minlength=rsize + 1)[:rsize])
    return (R, xp.reshape(-1, 1)) if nargout > 1 else R


@builtin("iradon")
def _iradon(interp, args, nargout):
    nargs(args, 1, 6, "iradon")
    R = _img(args[0], "iradon", "projection data").astype(float)
    if R.ndim != 2:
        raise MatlabError("iradon: the projections must be a matrix with one column per angle.")
    nproj = R.shape[1]
    theta = None
    interp_m, filt, d, n_out = "linear", "ram-lak", 1.0, None
    numeric = []
    for extra in args[1:]:
        if is_text(extra):
            s = text(extra).lower()
            if s in ("nearest", "linear", "spline", "pchip", "cubic", "v5cubic"):
                interp_m = s
            elif s in ("ram-lak", "shepp-logan", "cosine", "hamming", "hann", "none"):
                filt = s
            else:
                raise MatlabError(f"iradon: unknown option '{text(extra)}'.")
        else:
            numeric.append(extra)
    if numeric:
        t = numeric[0]
        if t.size == 0:
            theta = None
        elif t.size == 1 and nproj > 1:
            theta = np.arange(nproj) * float(t.flat[0])
        else:
            theta = _vec(t, "theta")
    if len(numeric) > 1:
        d = _scalar(numeric[1], "frequency scaling")
    if len(numeric) > 2:
        n_out = to_int(numeric[2], "output size")
    if theta is None:
        theta = np.arange(nproj) * 180.0 / nproj
    if theta.size != nproj:
        raise MatlabError("iradon: there must be one angle per projection (column).")
    if interp_m not in ("nearest", "linear"):
        raise MatlabError(f"iradon: the interpolation '{interp_m}' is not supported; use 'linear' or 'nearest'.")
    length = R.shape[0]
    if n_out is None:
        n_out = 2 * int(np.floor(length / (2 * np.sqrt(2))))
    order = max(64, int(2 ** np.ceil(np.log2(2 * length))))
    if filt == "none":
        proj = R
        H = None
    else:
        # band-limited ramp filter built from its impulse response (Kak and Slaney)
        nvec = np.concatenate([np.arange(0, order // 2 + 1), np.arange(order // 2 - 1, 0, -1)])
        h = np.zeros(order)
        h[0] = 0.25
        odd = nvec % 2 == 1
        h[odd] = -1.0 / (np.pi * nvec[odd]) ** 2
        H = 2 * np.real(np.fft.fft(h))
        w = 2 * np.pi * np.fft.fftfreq(order)
        aw = np.abs(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            if filt == "shepp-logan":
                H[1:] *= (np.sin(aw[1:] / (2 * d)) / (aw[1:] / (2 * d)))
            elif filt == "cosine":
                H[1:] *= np.cos(aw[1:] / (2 * d))
            elif filt == "hamming":
                H[1:] *= 0.54 + 0.46 * np.cos(aw[1:] / d)
            elif filt == "hann":
                H[1:] *= (1 + np.cos(aw[1:] / d)) / 2
        H[aw > np.pi * d] = 0
        P = np.fft.fft(R, n=order, axis=0) * H[:, None]
        proj = np.real(np.fft.ifft(P, axis=0))[:length]
    centre = int(np.ceil(length / 2)) - 1                      # zero-based centre of a projection
    mid = (n_out + 1) // 2
    xax = np.arange(n_out) - mid + 1.0
    yax = mid - 1.0 - np.arange(n_out)
    X, Y = np.meshgrid(xax, yax)
    img = np.zeros((n_out, n_out))
    for j, th in enumerate(np.deg2rad(theta)):
        t = X * np.cos(th) + Y * np.sin(th) + centre
        if interp_m == "nearest":
            idx = np.floor(t + 0.5).astype(int)
            ok = (idx >= 0) & (idx < length)
            img[ok] += proj[idx[ok], j]
        else:
            lo = np.floor(t).astype(int)
            frac = t - lo
            ok = (lo >= 0) & (lo < length - 1)
            img[ok] += (1 - frac[ok]) * proj[lo[ok], j] + frac[ok] * proj[lo[ok] + 1, j]
    img *= np.pi / (2 * nproj)
    if nargout > 1:
        return img, (H.reshape(-1, 1) if H is not None else np.ones((order, 1)))
    return img


# -- deconvolution ----------------------------------------------------------------------------------------

def _psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pad = np.zeros(shape)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def _deconv_inputs(args: list, name: str):
    a = _img(args[0], name)
    psf = _img(args[1], name, "point-spread function").astype(float)
    if psf.ndim != 2 or psf.sum() == 0:
        raise MatlabError(f"{name}: the point-spread function must be a matrix that does not sum to zero.")
    if psf.shape[0] > a.shape[0] or psf.shape[1] > a.shape[1]:
        raise MatlabError(f"{name}: the point-spread function must not be larger than the image.")
    return a, psf


def _deconv_out(out: np.ndarray, a: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "f":
        return out.astype(a.dtype)
    return _from_unit(np.clip(out, 0, 1), a.dtype)


@builtin("deconvwnr")
def _deconvwnr(interp, args, nargout):
    nargs(args, 2, 4, "deconvwnr")
    a, psf = _deconv_inputs(args, "deconvwnr")
    u = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    if len(args) == 4:
        raise MatlabError("deconvwnr: the noise and image autocorrelation form is not supported; give a "
                          "noise-to-signal ratio.")
    nsr = args[2].astype(float) if len(args) > 2 else np.zeros((1, 1))

    def run(p):
        H = _psf2otf(psf, p.shape)
        s = nsr if nsr.size == 1 else nsr
        if s.size != 1 and s.shape != p.shape:
            raise MatlabError("deconvwnr: the noise-to-signal ratio must be a scalar or have the image size.")
        den = np.abs(H) ** 2 + (float(s.flat[0]) if s.size == 1 else s)
        tiny = np.sqrt(np.finfo(float).eps)
        den = np.where(np.abs(den) < tiny, tiny, den)
        return np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(p) / den))

    return _deconv_out(_planes(u, run), a)


@builtin("deconvlucy")
def _deconvlucy(interp, args, nargout):
    nargs(args, 2, 7, "deconvlucy")
    a, psf = _deconv_inputs(args, "deconvlucy")
    numit = to_int(args[2], "number of iterations") if len(args) > 2 and args[2].size else 10
    damp = _scalar(args[3], "damping threshold") if len(args) > 3 and args[3].size else 0.0
    if len(args) > 4 and any(x.size for x in args[4:]):
        raise MatlabError("deconvlucy: the weight, read-out noise and sub-sampling inputs are not supported.")
    u = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    psf = psf / psf.sum()

    def run(p):
        H = _psf2otf(psf, p.shape)
        est = np.maximum(p, 0).copy()
        tiny = np.finfo(float).eps
        for _ in range(numit):
            blur = np.maximum(np.real(np.fft.ifft2(H * np.fft.fft2(est))), tiny)
            ratio = p / blur
            if damp > 0:
                # ignore deviations smaller than the damping threshold (in units of the noise)
                dev = np.abs(p - blur) / np.sqrt(np.maximum(blur, tiny))
                ratio = np.where(dev < damp, 1.0, ratio)
            est = np.maximum(est * np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(ratio))), 0)
        return est

    return _deconv_out(_planes(u, run), a)


@builtin("deconvreg")
def _deconvreg(interp, args, nargout):
    nargs(args, 2, 5, "deconvreg")
    a, psf = _deconv_inputs(args, "deconvreg")
    u = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    npow = _scalar(args[2], "noise power") if len(args) > 2 and args[2].size else 0.0
    lrange = _vec(args[3], "search range") if len(args) > 3 and args[3].size else np.array([1e-9, 1e9])
    regop = args[4].astype(float) if len(args) > 4 else np.array([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lagra_used = []

    def run(p):
        H = _psf2otf(psf, p.shape)
        L = _psf2otf(regop, p.shape)
        G = np.fft.fft2(p)
        h2, l2 = np.abs(H) ** 2, np.abs(L) ** 2

        def resid(lam):
            den = h2 + lam * l2
            den = np.where(den == 0, np.finfo(float).eps, den)
            r = G - H * (np.conj(H) * G / den)
            return float(np.sum(np.abs(r) ** 2)) / p.size

        if lrange.size == 1:
            lam = float(lrange[0])
        else:
            from scipy.optimize import fminbound
            lo, hi = np.log10(lrange[0]), np.log10(lrange[1])
            lam = 10 ** float(fminbound(lambda e: abs(resid(10 ** e) - npow), lo, hi, xtol=1e-6))
        lagra_used.append(lam)
        den = h2 + lam * l2
        den = np.where(den == 0, np.finfo(float).eps, den)
        return np.real(np.fft.ifft2(np.conj(H) * G / den))

    out = _deconv_out(_planes(u, run), a)
    return (out, float(lagra_used[0])) if nargout > 1 else out


# -- colour filter arrays -----------------------------------------------------------------------------------

_MHC_G = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]) / 8.0
_MHC_RB_ROW = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0],
                        [0, 0, 0.5, 0, 0]]) / 8.0
_MHC_RB_DIAG = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0],
                         [0, 0, -1.5, 0, 0]]) / 8.0


@builtin("demosaic")
def _demosaic(interp, args, nargout):
    nargs(args, 2, 2, "demosaic")
    a = _img(args[0], "demosaic")
    if a.ndim != 2 or a.dtype.kind not in "iu" and a.dtype != np.float32:
        raise MatlabError("demosaic: the image must be a 2-D uint8, uint16, uint32 or single array.")
    pattern = text(args[1]).lower()
    if pattern not in ("gbrg", "grbg", "bggr", "rggb"):
        raise MatlabError("demosaic: the sensor alignment must be 'gbrg', 'grbg', 'bggr' or 'rggb'.")
    x = a.astype(float)
    rows, cols = np.indices(a.shape)
    pos = {ch: [] for ch in "rgb"}
    for k, ch in enumerate(pattern):
        pos[ch].append((k // 2, k % 2))
    mask = {ch: np.zeros(a.shape, dtype=bool) for ch in "rgb"}
    for ch, lst in pos.items():
        for r0, c0 in lst:
            mask[ch] |= (rows % 2 == r0) & (cols % 2 == c0)
    (rr, rc), (br, bc) = pos["r"][0], pos["b"][0]

    def filt(h):
        return _correlate(x, h, "symmetric", "same", False, "demosaic")

    g_est, row_est, col_est, diag_est = filt(_MHC_G), filt(_MHC_RB_ROW), filt(_MHC_RB_ROW.T), filt(_MHC_RB_DIAG)
    G = np.where(mask["g"], x, g_est)
    out = {}
    for ch, (r0, c0), other in (("r", (rr, rc), "b"), ("b", (br, bc), "r")):
        same_row = (rows % 2 == r0) & mask["g"]          # green pixel in a row that holds this colour
        same_col = (cols % 2 == c0) & mask["g"]
        est = np.where(mask[ch], x, 0.0)
        est = np.where(same_row, row_est, est)
        est = np.where(same_col, col_est, est)
        est = np.where(mask[other], diag_est, est)
        out[ch] = est
    rgb = np.stack([out["r"], G, out["b"]], axis=-1)
    if a.dtype.kind == "f":
        return rgb.astype(a.dtype)
    return _cast(rgb, a.dtype)


# -- connectivity -----------------------------------------------------------------------------------------

def _conn(v, ndim: int, name: str) -> np.ndarray:
    """A 3x3(x3) neighbourhood for a connectivity given as 4, 8, 6, 18, 26 or as a matrix."""
    nd = _nd()
    if v is None:
        return np.ones((3,) * ndim, dtype=bool)
    a = _img(v, name, "connectivity")
    if a.size == 1:
        c = int(a.flat[0])
        if ndim == 2 and c in (4, 8):
            return nd.generate_binary_structure(2, 1 if c == 4 else 2)
        if ndim == 3 and c in (6, 18, 26):
            return nd.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[c])
        if ndim == 3 and c in (4, 8):
            st = np.zeros((3, 3, 3), dtype=bool)
            st[:, :, 1] = nd.generate_binary_structure(2, 1 if c == 4 else 2)
            return st
        if ndim == 2 and c in (6, 18, 26):
            return nd.generate_binary_structure(2, 1 if c == 6 else 2)
        raise MatlabError(f"{name}: the connectivity must be 4 or 8 (6, 18 or 26 for volumes).")
    st = a != 0
    if any(s != 3 for s in st.shape) or st.ndim > ndim:
        raise MatlabError(f"{name}: a connectivity matrix must be 3-by-3 (or 3-by-3-by-3).")
    st = st.reshape(st.shape + (1,) * (ndim - st.ndim))
    if not st[(1,) * st.ndim] or not np.array_equal(st, st[(slice(None, None, -1),) * st.ndim]):
        raise MatlabError(f"{name}: a connectivity matrix must be symmetric about its centre, which must be set.")
    if st.shape != (3,) * ndim:
        full = np.zeros((3,) * ndim, dtype=bool)
        full[(slice(None),) * st.ndim] = st
        st = full
    return st


def _label_cm(bw: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components numbered in column-major order of their first element (any dimension)."""
    axes = tuple(range(bw.ndim - 1, -1, -1))
    lab, n = _nd().label(bw.transpose(axes), structure=st.transpose(axes))
    return lab.transpose(axes), int(n)


def _low_value(a: np.ndarray):
    if a.dtype == bool:
        return False
    if a.dtype.kind in "iu":
        return np.iinfo(a.dtype).min
    return -np.inf


# -- gray-scale reconstruction and extrema ----------------------------------------------------------------

def _reconstruct(marker: np.ndarray, mask: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Morphological reconstruction by dilation of marker under mask."""
    nd = _nd()
    if marker.dtype == bool and mask.dtype == bool:
        return nd.binary_propagation(marker & mask, structure=st, mask=mask)
    if marker.dtype.kind == "f" and (np.isinf(marker).any() or np.isinf(mask).any()):
        # work with finite stand-ins for the infinities and put them back afterwards
        finite = np.concatenate([marker[np.isfinite(marker)], mask[np.isfinite(mask)]])
        lo = (finite.min() if finite.size else 0.0) - 1.0
        hi = (finite.max() if finite.size else 0.0) + 1.0
        res = _reconstruct(np.clip(marker, lo, hi), np.clip(mask, lo, hi), st)
        return np.where(res <= lo, -np.inf, np.where(res >= hi, np.inf, res))
    cur = np.minimum(marker, mask)
    levels = np.unique(np.concatenate([cur.ravel(), mask.ravel()]))
    if levels.size <= 600:
        # threshold decomposition: one binary propagation per gray level
        out = np.full(cur.shape, levels[0] if levels.size else 0, dtype=cur.dtype)
        for t in levels[1:]:
            region = mask >= t
            seeds = (cur >= t) & region
            if not seeds.any():
                break
            out[nd.binary_propagation(seeds, structure=st, mask=region)] = t
        return out
    low = _low_value(cur)
    for _ in range(cur.size + 1):
        nxt = np.minimum(nd.grey_dilation(cur, footprint=st, mode="constant", cval=low), mask)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def _same_class_pair(args: list, name: str) -> tuple[np.ndarray, np.ndarray]:
    marker, mask = _img(args[0], name, "marker"), _img(args[1], name, "mask")
    if marker.shape != mask.shape:
        raise MatlabError(f"{name}: the marker and the mask must have the same size.")
    if marker.dtype != mask.dtype:
        raise MatlabError(f"{name}: the marker and the mask must have the same class.")
    return marker, mask


@builtin("imreconstruct")
def _imreconstruct(interp, args, nargout):
    nargs(args, 2, 3, "imreconstruct")
    marker, mask = _same_class_pair(args, "imreconstruct")
    st = _conn(args[2] if len(args) > 2 else None, max(marker.ndim, 2), "imreconstruct")
    return _reconstruct(marker, mask, st)


def _regional_max(a: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Plateaus none of whose neighbours is higher."""
    nd = _nd()
    if a.size == 0:
        return np.zeros(a.shape, dtype=bool)
    work = a.astype(np.int64) if a.dtype.kind in "biu" else a.astype(float)
    low = np.iinfo(np.int64).min if work.dtype.kind == "i" else -np.inf
    cand = work == nd.maximum_filter(work, footprint=st, mode="constant", cval=low)
    if work.dtype.kind == "f":
        cand &= ~np.isnan(work)
    # a candidate plateau is spoilt when it touches an equal pixel that itself has a higher neighbour
    outside = np.where(cand, low, work)
    spoil = cand & (nd.maximum_filter(outside, footprint=st, mode="constant", cval=low) >= work)
    if not spoil.any():
        return cand
    lab, n = nd.label(cand, structure=st)
    bad = np.zeros(n + 1, dtype=bool)
    bad[np.unique(lab[spoil])] = True
    bad[0] = False
    return cand & ~bad[lab]


def _negate(a: np.ndarray) -> np.ndarray:
    """Order-reversing copy of an image (complement for integers, negation for floats)."""
    if a.dtype == bool:
        return ~a
    if a.dtype.kind == "u":
        return np.iinfo(a.dtype).max - a
    if a.dtype.kind == "i":
        return (-1 - a.astype(np.int64)).astype(a.dtype)
    return -a


def _extrema_builtin(name: str, minimum: bool):
    def fn(interp, args, nargout):
        nargs(args, 1, 2, name)
        a = _img(args[0], name)
        st = _conn(args[1] if len(args) > 1 else None, max(a.ndim, 2), name)
        return _regional_max(_negate(a) if minimum else a, st)
    builtin(name)(fn)


_extrema_builtin("imregionalmax", False)
_extrema_builtin("imregionalmin", True)


def _minus_h(a: np.ndarray, h: float) -> np.ndarray:
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        return np.clip(a.astype(np.int64) - int(round(h)), info.min, info.max).astype(a.dtype)
    return a - h


def _hmax(a: np.ndarray, h: float, st: np.ndarray) -> np.ndarray:
    return _reconstruct(_minus_h(a, h), a, st)


def _h_builtin(name: str, minimum: bool, extended: bool):
    def fn(interp, args, nargout):
        nargs(args, 2, 3, name)
        a = _img(args[0], name)
        if a.dtype == bool:
            raise MatlabError(f"{name}: the image must be numeric, not logical.")
        h = _scalar(args[1], "height")
        if h < 0:
            raise MatlabError(f"{name}: the height must not be negative.")
        st = _conn(args[2] if len(args) > 2 else None, max(a.ndim, 2), name)
        work = _negate(a) if minimum else a
        res = _hmax(work, h, st)
        if extended:
            return _regional_max(res, st)
        return _negate(res) if minimum else res
    builtin(name)(fn)


_h_builtin("imhmax", False, False)
_h_builtin("imhmin", True, False)
_h_builtin("imextendedmax", False, True)
_h_builtin("imextendedmin", True, True)


@builtin("imimposemin")
def _imimposemin(interp, args, nargout):
    nargs(args, 2, 3, "imimposemin")
    a = _img(args[0], "imimposemin")
    bw = _img(args[1], "imimposemin", "marker image") != 0
    if bw.shape != a.shape:
        raise MatlabError("imimposemin: the marker image must have the size of the image.")
    st = _conn(args[2] if len(args) > 2 else None, max(a.ndim, 2), "imimposemin")
    x = a.astype(float)
    if a.dtype.kind == "f":
        span = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
        h = span / 1000 if span > 0 else 0.1
    else:
        h = 1.0
    fm = np.where(bw, -np.inf, np.inf)
    g = np.minimum(x + h, fm)
    out = -_reconstruct(-fm, -g, st)          # reconstruction by erosion of fm above g
    if a.dtype.kind == "f":
        return out.astype(a.dtype)
    info = np.iinfo(a.dtype) if a.dtype.kind in "iu" else None
    if info is None:
        return out
    return np.clip(np.where(np.isneginf(out), info.min, out), info.min, info.max).astype(a.dtype)


@builtin("watershed")
def _watershed(interp, args, nargout):
    nargs(args, 1, 2, "watershed")
    a = _img(args[0], "watershed")
    ndim = max(a.ndim, 2)
    st = _conn(args[1] if len(args) > 1 else None, ndim, "watershed")
    if a.size == 0:
        return np.zeros(a.shape)
    import heapq
    vals = a.astype(float)
    labels, _n = _label_cm(_regional_max(_negate(a), st), st)
    padded_shape = tuple(s + 2 for s in vals.shape)
    lab = np.full(padded_shape, -2, dtype=np.int64)          # -2 marks the border
    inner = tuple(slice(1, -1) for _ in vals.shape)
    lab[inner] = labels
    val = np.zeros(padded_shape)
    val[inner] = vals
    strides = np.array([int(np.prod(padded_shape[k + 1:])) for k in range(ndim)])
    offs = [int(np.dot(np.array(idx) - 1, strides)) for idx in np.argwhere(st) if not np.all(idx == 1)]
    flat_lab = lab.ravel()
    flat_val = val.ravel()
    queued = np.zeros(flat_lab.size, dtype=bool)
    heap: list = []
    count = 0
    seeds = np.flatnonzero(flat_lab > 0)
    for p in seeds:
        for o in offs:
            q = p + o
            if flat_lab[q] == 0 and not queued[q]:
                queued[q] = True
                heapq.heappush(heap, (flat_val[q], count, q))
                count += 1
    while heap:
        level, _c, p = heapq.heappop(heap)
        found = 0
        ridge = False
        for o in offs:
            l = flat_lab[p + o]
            if l > 0:
                if found == 0:
                    found = l
                elif l != found:
                    ridge = True
                    break
        if ridge or found == 0:
            flat_lab[p] = -1                                 # a watershed pixel
            continue
        flat_lab[p] = found
        for o in offs:
            q = p + o
            if flat_lab[q] == 0 and not queued[q]:
                queued[q] = True
                heapq.heappush(heap, (max(flat_val[q], level), count, q))
                count += 1
    out = lab[inner].copy()
    out[out < 0] = 0
    return out.astype(float)


@builtin("imclearborder")
def _imclearborder(interp, args, nargout):
    nargs(args, 1, 2, "imclearborder")
    a = _img(args[0], "imclearborder")
    st = _conn(args[1] if len(args) > 1 else None, max(a.ndim, 2), "imclearborder")
    edge = np.zeros(a.shape, dtype=bool)
    for ax in range(a.ndim):
        if a.shape[ax] > 0:
            sl = [slice(None)] * a.ndim
            sl[ax] = 0
            edge[tuple(sl)] = True
            sl[ax] = -1
            edge[tuple(sl)] = True
    if a.dtype == bool:
        return a & ~_reconstruct(a & edge, a, st)
    low = _low_value(a)
    marker = np.where(edge, a, low).astype(a.dtype) if a.dtype.kind != "f" else np.where(edge, a, low)
    rec = _reconstruct(marker, a, st)
    if a.dtype.kind in "iu":
        return (a.astype(np.int64) - rec.astype(np.int64)).clip(np.iinfo(a.dtype).min,
                                                               np.iinfo(a.dtype).max).astype(a.dtype)
    return a - np.where(np.isneginf(rec), 0, rec)


# -- binary image tools ---------------------------------------------------------------------------------

def _bw(v, name: str, ndim: int | None = 2) -> np.ndarray:
    a = _img(v, name)
    if ndim is not None and a.ndim != ndim:
        raise MatlabError(f"{name}: the image must be {ndim}-D.")
    return a != 0


@builtin("bwlabeln")
def _bwlabeln(interp, args, nargout):
    nargs(args, 1, 2, "bwlabeln")
    bw = _bw(args[0], "bwlabeln", None)
    st = _conn(args[1] if len(args) > 1 else None, max(bw.ndim, 2), "bwlabeln")
    lab, n = _label_cm(bw, st)
    return (lab.astype(float), float(n)) if nargout > 1 else lab.astype(float)


@builtin("bwulterode")
def _bwulterode(interp, args, nargout):
    nargs(args, 1, 3, "bwulterode")
    bw = _bw(args[0], "bwulterode")
    method = "euclidean"
    conn_arg = None
    for extra in args[1:]:
        if is_text(extra):
            method = text(extra).lower()
        else:
            conn_arg = extra
    st = _conn(conn_arg, 2, "bwulterode")
    dist = BUILTINS["bwdist"](interp, [~bw, char(method)], 1)
    return _regional_max(np.asarray(dist, dtype=float), st) & bw


@builtin("bwhitmiss")
def _bwhitmiss(interp, args, nargout):
    nargs(args, 2, 3, "bwhitmiss")
    bw = _bw(args[0], "bwhitmiss")
    if len(args) == 2:
        interval = _img(args[1], "bwhitmiss", "interval")
        se1, se2 = interval == 1, interval == -1
    else:
        se1 = _nhood(args[1], "bwhitmiss", 2)
        se2 = _nhood(args[2], "bwhitmiss", 2)
    if se1.shape == se2.shape and np.any(se1 & se2):
        raise MatlabError("bwhitmiss: the two structuring elements must not overlap.")

    def erode(img, se):
        return _morph(img, se, False) if se.any() else np.ones(img.shape, dtype=bool)

    return erode(bw, se1) & erode(~bw, se2)


@builtin("bwselect")
def _bwselect(interp, args, nargout):
    nargs(args, 3, 4, "bwselect")
    bw = _bw(args[0], "bwselect")
    c, r = _vec(args[1], "columns"), _vec(args[2], "rows")
    if c.size != r.size:
        raise MatlabError("bwselect: the column and row coordinates must have the same number of elements.")
    st, _ = _conn_structure(args[3] if len(args) > 3 else None, 8, "bwselect")
    ri = np.floor(r + 0.5).astype(int) - 1
    ci = np.floor(c + 0.5).astype(int) - 1
    ok = (ri >= 0) & (ri < bw.shape[0]) & (ci >= 0) & (ci < bw.shape[1])
    seed = np.zeros(bw.shape, dtype=bool)
    seed[ri[ok], ci[ok]] = True
    out = _nd().binary_propagation(seed & bw, structure=st, mask=bw)
    if nargout > 1:
        idx = np.flatnonzero(out.ravel(order="F")) + 1
        return out, idx.astype(float).reshape(-1, 1)
    return out


_DIRS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
_DIR_STEPS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]       # clockwise from north


@builtin("bwtraceboundary")
def _bwtraceboundary(interp, args, nargout):
    nargs(args, 3, 6, "bwtraceboundary")
    bw = _bw(args[0], "bwtraceboundary")
    p = _vec(args[1], "starting point").astype(int)
    if p.size != 2:
        raise MatlabError("bwtraceboundary: the starting point must be [row col].")
    first = text(args[2]).lower()
    conn = int(_scalar(args[3], "connectivity")) if len(args) > 3 and args[3].size else 8
    limit = _scalar(args[4], "number of points") if len(args) > 4 and args[4].size else np.inf
    direction = text(args[5]).lower() if len(args) > 5 else "clockwise"
    if conn not in (4, 8):
        raise MatlabError("bwtraceboundary: the connectivity must be 4 or 8.")
    if first not in _DIRS or (conn == 4 and len(first) == 2):
        raise MatlabError("bwtraceboundary: the first step must be 'N', 'E', 'S' or 'W' (with 'NE', 'SE', 'SW' "
                          "and 'NW' for 8-connected tracing).")
    if direction not in ("clockwise", "counterclockwise"):
        raise MatlabError("bwtraceboundary: the direction must be 'clockwise' or 'counterclockwise'.")
    r0, c0 = int(p[0]) - 1, int(p[1]) - 1
    if not (0 <= r0 < bw.shape[0] and 0 <= c0 < bw.shape[1]) or not bw[r0, c0]:
        return np.zeros((0, 2))
    padded = np.pad(bw, 1, mode="constant")
    turn = 1 if direction == "clockwise" else -1
    step = 1 if conn == 8 else 2
    nsteps = 8 // step
    pos = (r0 + 1, c0 + 1)
    scan = _DIRS.index(first)
    pts = [pos]
    first_move = None
    while len(pts) < limit:
        found = None
        for k in range(nsteps):
            d = (scan + turn * step * k) % 8
            if padded[pos[0] + _DIR_STEPS[d][0], pos[1] + _DIR_STEPS[d][1]]:
                found = d
                break
        if found is None:
            break                                            # an isolated pixel
        if first_move is None:
            first_move = found
        elif pos == pts[0] and found == first_move:
            break                                            # about to repeat the first move
        pos = (pos[0] + _DIR_STEPS[found][0], pos[1] + _DIR_STEPS[found][1])
        pts.append(pos)
        # resume the search just past the background pixel examined before this move
        back = (found + 4) % 8
        scan = (back + turn * step) % 8
    return np.array(pts, dtype=float)


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points (Andrew's monotone chain), counter-clockwise, without repeats."""
    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return pts

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) > 1e-12:
                    break
                out.pop()
            out.append((float(q[0]), float(q[1])))
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _corner_points(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """[x y] corners of the given pixels (one-based pixel centres)."""
    x, y = cols + 1.0, rows + 1.0
    return np.concatenate([np.column_stack([x + dx, y + dy]) for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])


def _hull_mask(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull polygon (closed, [x y]) of some pixels and the pixels it covers."""
    from matplotlib.path import Path
    hull = _hull(_corner_points(rows, cols))
    if hull.shape[0] < 3:
        mask = np.zeros(shape, dtype=bool)
        mask[rows, cols] = True
        return hull, mask
    cc, rr = np.meshgrid(np.arange(1, shape[1] + 1, dtype=float), np.arange(1, shape[0] + 1, dtype=float))
    inside = Path(hull).contains_points(np.column_stack([cc.ravel(), rr.ravel()]), radius=1e-9)
    mask = inside.reshape(shape)
    mask[rows, cols] = True
    return np.vstack([hull, hull[:1]]), mask


@builtin("bwconvhull")
def _bwconvhull(interp, args, nargout):
    nargs(args, 1, 3, "bwconvhull")
    bw = _bw(args[0], "bwconvhull")
    method = text(args[1]).lower() if len(args) > 1 else "union"
    if method not in ("union", "objects"):
        raise MatlabError("bwconvhull: the method must be 'union' or 'objects'.")
    st, _ = _conn_structure(args[2] if len(args) > 2 else None, 8, "bwconvhull")
    out = np.zeros(bw.shape, dtype=bool)
    if not bw.any():
        return out
    if method == "union":
        r, c = np.nonzero(bw)
        return _hull_mask(r, c, bw.shape)[1]
    lab, n = _label(bw, st)
    for k in range(1, n + 1):
        r, c = np.nonzero(lab == k)
        out |= _hull_mask(r, c, bw.shape)[1]
    return out


def _quad_counts(bw: np.ndarray) -> np.ndarray:
    """Number of set pixels and the diagonal flag of every 2-by-2 window of the zero-padded image."""
    p = np.pad(bw.astype(np.int64), 1, mode="constant")
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    total = a + b + c + d
    diag = (total == 2) & (a == d)
    return total, diag


@builtin("bweuler")
def _bweuler(interp, args, nargout):
    nargs(args, 1, 2, "bweuler")
    bw = _bw(args[0], "bweuler")
    conn = int(_scalar(args[1], "connectivity")) if len(args) > 1 else 8
    if conn not in (4, 8):
        raise MatlabError("bweuler: the connectivity must be 4 or 8.")
    total, diag = _quad_counts(bw)
    q1, q3, qd = int((total == 1).sum()), int((total == 3).sum()), int(diag.sum())
    return float((q1 - q3 - 2 * qd) / 4 if conn == 8 else (q1 - q3 + 2 * qd) / 4)


@builtin("bwarea")
def _bwarea(interp, args, nargout):
    nargs(args, 1, 1, "bwarea")
    bw = _bw(args[0], "bwarea")
    total, diag = _quad_counts(bw)
    weights = np.array([0.0, 0.25, 0.5, 0.875, 1.0])
    area = weights[total].sum() + 0.25 * diag.sum()
    return float(area)


@builtin("bwskel")
def _bwskel(interp, args, nargout):
    rest, opts = _options(list(args), ("MinBranchLength",), "bwskel")
    nargs(rest, 1, 1, "bwskel")
    bw = _bw(rest[0], "bwskel")
    if "minbranchlength" in opts and _scalar(opts["minbranchlength"], "MinBranchLength") > 0:
        raise MatlabError("bwskel: pruning with 'MinBranchLength' is not supported.")
    return BUILTINS["bwmorph"](interp, [bw, char("thin"), _f(np.inf)], 1)


def _props_of(interp, lab: np.ndarray, names: list[str], values=None) -> Struct:
    a = [lab.astype(float)] + ([values] if values is not None else []) + [char(n) for n in names]
    return BUILTINS["regionprops"](interp, a, 1)


def _prop_filter(interp, args: list, name: str, area_only: bool):
    rest = list(args)
    bw = _bw(rest.pop(0), name)
    values = None
    if not area_only and rest and not is_text(rest[0]):
        values = _img(rest.pop(0), name, "grayscale image")
    if area_only:
        attrib = "Area"
    else:
        if not rest or not is_text(rest[0]):
            raise MatlabError(f"{name}: the name of a property is required.")
        attrib = text(rest.pop(0))
    if not rest:
        raise MatlabError(f"{name}: a range [low high] or a number of objects is required.")
    spec = _vec(rest.pop(0), "range")
    keep_which = "largest"
    conn_arg = None
    for extra in rest:
        if is_text(extra):
            keep_which = text(extra).lower()
        else:
            conn_arg = extra
    if keep_which not in ("largest", "smallest"):
        raise MatlabError(f"{name}: the keep option must be 'largest' or 'smallest'.")
    st, _ = _conn_structure(conn_arg, 8, name)
    lab, n = _label(bw, st)
    props = _props_of(interp, lab, [attrib], values)
    key = props.fields[0]
    vals = np.array([float(np.ravel(props.data.flat[k][key])[0]) for k in range(n)])
    if spec.size == 2:
        keep = (vals >= spec[0]) & (vals <= spec[1])
    elif spec.size == 1:
        count = int(spec[0])
        order = np.argsort(-vals if keep_which == "largest" else vals, kind="stable")
        keep = np.zeros(n, dtype=bool)
        keep[order[:count]] = True
    else:
        raise MatlabError(f"{name}: give a range [low high] or a number of objects.")
    table = np.concatenate([[False], keep])
    return table[lab]


@builtin("bwpropfilt")
def _bwpropfilt(interp, args, nargout):
    nargs(args, 3, 6, "bwpropfilt")
    return _prop_filter(interp, args, "bwpropfilt", False)


@builtin("bwareafilt")
def _bwareafilt(interp, args, nargout):
    nargs(args, 2, 4, "bwareafilt")
    return _prop_filter(interp, args, "bwareafilt", True)


@builtin("makelut")
def _makelut(interp, args, nargout):
    nargs(args, 2, None, "makelut")
    fun = args[0]
    n = to_int(args[1], "neighbourhood size")
    if n not in (2, 3):
        raise MatlabError("makelut: the neighbourhood size must be 2 or 3.")
    extra = list(args[2:])
    count = 2 ** (n * n)
    lut = np.zeros((count, 1))
    for i in range(count):
        bits = np.array([(i >> k) & 1 for k in range(n * n)], dtype=float).reshape((n, n), order="F")
        res = np.asarray(_call(interp, fun, [bits, *extra], 1)[0])
        if res.size != 1:
            raise MatlabError("makelut: the function must return a scalar.")
        lut[i, 0] = float(np.real(res.flat[0]))
    return lut


@builtin("applylut")
def _applylut(interp, args, nargout):
    nargs(args, 2, 2, "applylut")
    bw = _bw(args[0], "applylut")
    lut = _img(args[1], "applylut", "lookup table").flatten(order="F")
    if lut.size == 16:
        n = 2
    elif lut.size == 512:
        n = 3
    else:
        raise MatlabError("applylut: the lookup table must have 16 or 512 elements (from makelut).")
    b = bw.astype(np.int64)
    if n == 3:
        p = np.pad(b, 1, mode="constant")
        rows, cols = b.shape
        idx = np.zeros(b.shape, dtype=np.int64)
        k = 0
        for dc in range(3):
            for dr in range(3):
                idx += p[dr:dr + rows, dc:dc + cols] << k
                k += 1
    else:
        p = np.pad(b, ((0, 1), (0, 1)), mode="constant")
        rows, cols = b.shape
        idx = p[:rows, :cols] + (p[1:, :cols] << 1) + (p[:rows, 1:] << 2) + (p[1:, 1:] << 3)
    out = lut[idx]
    if np.all((out == 0) | (out == 1)):
        return out.astype(bool)
    return out.astype(float)


_P3 = ["Volume", "Centroid", "BoundingBox", "EquivDiameter", "Extent", "VoxelIdxList", "VoxelList", "Image"]
_P3_VALUE = ["MeanIntensity", "MinIntensity", "MaxIntensity", "VoxelValues"]


@builtin("regionprops3")
def _regionprops3(interp, args, nargout):
    nargs(args, 1, None, "regionprops3")
    rest = list(args)
    src = rest.pop(0)
    if isinstance(src, Struct):
        s = src.data.flat[0]
        shape = tuple(int(v) for v in np.ravel(s["ImageSize"]))
        shape = shape + (1,) * (3 - len(shape))
        lab = np.zeros(int(np.prod(shape)), dtype=np.int64)
        for k, idx in enumerate(s["PixelIdxList"].values_f()):
            lab[np.ravel(idx).astype(np.int64) - 1] = k + 1
        lab = lab.reshape(shape, order="F")
    else:
        a = _img(src, "regionprops3")
        if a.ndim > 3:
            raise MatlabError("regionprops3: the volume must have at most three dimensions.")
        a = a.reshape(a.shape + (1,) * (3 - a.ndim))
        if a.dtype == bool:
            lab, _n = _label_cm(a, np.ones((3, 3, 3), dtype=bool))
        else:
            lab = a.astype(np.int64)
    n = int(lab.max()) if lab.size else 0
    values = None
    if rest and not is_text(rest[0]):
        values = _img(rest.pop(0), "regionprops3", "intensity volume")
        values = values.reshape(values.shape + (1,) * (3 - values.ndim))
        if values.shape != lab.shape:
            raise MatlabError("regionprops3: the intensity volume must have the size of the label volume.")
    known = {p.lower(): p for p in _P3 + _P3_VALUE}
    names: list[str] = []
    for r in rest or [char("basic")]:
        key = text(r).lower()
        if key == "basic":
            group = ["Volume", "Centroid", "BoundingBox"]
        elif key == "all":
            group = _P3 + (_P3_VALUE if values is not None else [])
        elif key in known:
            group = [known[key]]
        else:
            raise MatlabError(f"regionprops3: the property '{text(r)}' is not supported. Supported: "
                              + ", ".join(_P3 + _P3_VALUE) + ".")
        names.extend(g for g in group if g not in names)
    flat = lab.ravel(order="F")
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(1, n + 2))
    cols: dict[str, list] = {p: [] for p in names}
    for k in range(n):
        idx = order[bounds[k]:bounds[k + 1]]
        r, c, z = np.unravel_index(idx, lab.shape, order="F")
        vol = float(idx.size)
        for p in names:
            if p in _P3_VALUE and values is None:
                raise MatlabError(f"regionprops3: '{p}' needs the intensity volume as the second input.")
            if idx.size == 0:
                cols[p].append(0.0 if p == "Volume" else np.zeros((0, 0)))
            elif p == "Volume":
                cols[p].append(vol)
            elif p == "Centroid":
                cols[p].append(np.array([c.mean() + 1, r.mean() + 1, z.mean() + 1]))
            elif p == "BoundingBox":
                cols[p].append(np.array([c.min() + 0.5, r.min() + 0.5, z.min() + 0.5, np.ptp(c) + 1.0, np.ptp(r) + 1.0,
                                         np.ptp(z) + 1.0]))
            elif p == "EquivDiameter":
                cols[p].append(float((6 * vol / np.pi) ** (1 / 3)))
            elif p == "Extent":
                cols[p].append(vol / float((np.ptp(c) + 1) * (np.ptp(r) + 1) * (np.ptp(z) + 1)))
            elif p == "VoxelIdxList":
                cols[p].append((idx + 1).astype(float).reshape(-1, 1))
            elif p == "VoxelList":
                cols[p].append(np.column_stack([c + 1, r + 1, z + 1]).astype(float))
            elif p == "Image":
                im = np.zeros((np.ptp(r) + 1, np.ptp(c) + 1, np.ptp(z) + 1), dtype=bool)
                im[r - r.min(), c - c.min(), z - z.min()] = True
                cols[p].append(im)
            else:
                v = values.ravel(order="F")[idx]
                if p == "MeanIntensity":
                    cols[p].append(float(v.astype(float).mean()))
                elif p == "MinIntensity":
                    cols[p].append(float(v.min()))
                elif p == "MaxIntensity":
                    cols[p].append(float(v.max()))
                else:
                    cols[p].append(v.reshape(-1, 1))
    return _props_table(names, cols, n)


_TABLE_NUMERIC = {"Area", "Volume", "Centroid", "BoundingBox", "MajorAxisLength", "MinorAxisLength",
                  "Eccentricity", "Orientation", "EquivDiameter", "Extent", "Perimeter", "Circularity",
                  "FilledArea", "EulerNumber", "MeanIntensity", "MinIntensity", "MaxIntensity",
                  "WeightedCentroid", "ConvexArea", "Solidity"}


def _props_table(names: list[str], cols: dict, n: int):
    """A table with one row per region: numbers and fixed-length vectors as matrices, the rest as cells."""
    from .table import TableObject
    columns = []
    for p in names:
        items = cols[p]
        if n == 0:
            columns.append(np.zeros((0, 1)))
        elif p in _TABLE_NUMERIC and len({np.size(v) for v in items}) == 1:
            columns.append(np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in items]))
        else:
            data = np.empty((n, 1), dtype=object)
            for i, v in enumerate(items):
                data[i, 0] = v if isinstance(v, (np.ndarray, Cell, Struct)) else _f(v)
            columns.append(Cell(data))
    return TableObject(names, columns)


# -- geometric transformation objects -------------------------------------------------------------------

_OLD_KINDS = ("affine2d", "projective2d")
_AFFINE_KINDS = ("affine2d", "affinetform2d", "rigidtform2d", "simtform2d", "transltform2d")


class GeoTform(MObject):
    """A 2-D geometric transformation. A maps column vectors: [x; y; 1] ~ A * [u; v; 1]."""

    _METHODS = {"transformPointsForward", "transformPointsInverse", "invert", "outputLimits", "isRigid",
                "isSimilarity", "isTranslation"}

    def __init__(self, kind: str, A: np.ndarray):
        self.kind = kind
        self.A = np.array(A, dtype=float)

    def class_name(self) -> str:
        return self.kind

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def is_affine(self) -> bool:
        return self.kind in _AFFINE_KINDS

    def forward(self, pts: np.ndarray) -> np.ndarray:
        h = np.column_stack([pts, np.ones(pts.shape[0])]) @ self.A.T
        return h[:, :2] / h[:, 2:3]

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        h = np.column_stack([pts, np.ones(pts.shape[0])]) @ np.linalg.inv(self.A).T
        return h[:, :2] / h[:, 2:3]

    def _parts(self) -> tuple[float, float, np.ndarray]:
        """Scale, rotation angle in degrees and translation of a similarity."""
        a, b = self.A[0, 0], self.A[1, 0]
        return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a))), self.A[:2, 2].copy()

    def get_field(self, interp, name: str):
        if name == "Dimensionality":
            return _f(2)
        if name == "T" and self.kind in _OLD_KINDS:
            return self.A.T.copy()
        if name == "A" and self.kind not in _OLD_KINDS:
            return self.A.copy()
        if self.kind in ("rigidtform2d", "simtform2d", "transltform2d"):
            scale, ang, tr = self._parts()
            if name == "Translation":
                return tr.reshape(1, 2)
            if name == "RotationAngle" and self.kind != "transltform2d":
                return _f(ang)
            if name == "R" and self.kind != "transltform2d":
                t = np.deg2rad(ang)
                return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            if name == "Scale" and self.kind == "simtform2d":
                return _f(scale)
        if name in self._METHODS:
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")

    def set_field(self, interp, name: str, value):
        m = np.asarray(value, dtype=float)
        if name == "T" and self.kind in _OLD_KINDS and m.shape == (3, 3):
            return make_tform(self.kind, m.T)
        if name == "A" and self.kind not in _OLD_KINDS and m.shape == (3, 3):
            return make_tform(self.kind, m)
        raise MatlabError(f"Cannot set property '{name}' of class '{self.kind}'.")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        me = next(i for i, a in enumerate(args) if a is self)
        rest = [a for i, a in enumerate(args) if i != me]
        if name in ("transformPointsForward", "transformPointsInverse"):
            fn = self.forward if name.endswith("Forward") else self.inverse
            if len(rest) == 1:
                pts = _img(rest[0], name, "points").astype(float)
                if pts.ndim != 2 or pts.shape[1] != 2:
                    raise MatlabError(f"{name}: the points must be an n-by-2 matrix [x y].")
                return [fn(pts)]
            if len(rest) == 2:
                u, v = rest[0].astype(float), rest[1].astype(float)
                if u.shape != v.shape:
                    raise MatlabError(f"{name}: the two coordinate arrays must have the same size.")
                out = fn(np.column_stack([u.ravel(), v.ravel()]))
                return [out[:, 0].reshape(u.shape), out[:, 1].reshape(u.shape)]
            raise MatlabError(f"{name}: expected (tform, points) or (tform, u, v).")
        if name == "invert":
            inv = np.linalg.inv(self.A)
            return [make_tform(self.kind, inv / inv[2, 2])]
        if name == "outputLimits":
            xl, yl = _vec(rest[0], "x limits"), _vec(rest[1], "y limits")
            xo, yo = _out_limits(self, xl, yl)
            return [xo.reshape(1, 2), yo.reshape(1, 2)]
        lin = self.A[:2, :2]
        affine = np.allclose(self.A[2], [0, 0, 1])
        if name == "isTranslation":
            return [np.array([[bool(affine and np.allclose(lin, np.eye(2)))]])]
        s = np.linalg.svd(lin, compute_uv=False)
        similar = affine and np.isclose(s[0], s[1], rtol=1e-8, atol=1e-10)
        if name == "isSimilarity":
            return [np.array([[bool(similar)]])]
        if name == "isRigid":
            return [np.array([[bool(similar and np.isclose(s[0], 1.0, atol=1e-8) and np.linalg.det(lin) > 0)]])]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        mat = self.A.T if self.kind in _OLD_KINDS else self.A
        rows = ["    " + "  ".join(f"{v:10.4f}" for v in r) for r in mat]
        label = "T" if self.kind in _OLD_KINDS else "A"
        return [f"  {self.kind} with properties:", "", "    Dimensionality: 2", f"    {label}:"] + rows


def make_tform(kind: str, A: np.ndarray) -> GeoTform:
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise MatlabError(f"{kind}: the transformation matrix must be 3-by-3.")
    if abs(np.linalg.det(A)) < 1e-14 * max(1.0, float(np.max(np.abs(A))) ** 3):
        raise MatlabError(f"{kind}: the transformation matrix must be invertible.")
    if kind in _AFFINE_KINDS and not np.allclose(A[2], [0, 0, 1], atol=1e-8 * max(1.0, float(np.max(np.abs(A))))):
        raise MatlabError(f"{kind}: the last {'column' if kind in _OLD_KINDS else 'row'} of an affine "
                          "transformation must be [0 0 1].")
    return GeoTform(kind, A)


def _matrix_arg(v, name: str) -> np.ndarray:
    if isinstance(v, GeoTform):
        return v.A
    m = _img(v, name, "transformation matrix").astype(float)
    if m.shape != (3, 3):
        raise MatlabError(f"{name}: the transformation matrix must be 3-by-3.")
    return m


def _old_tform(kind: str):
    def fn(interp, args, nargout):
        nargs(args, 0, 1, kind)
        if not args:
            return GeoTform(kind, np.eye(3))
        if isinstance(args[0], GeoTform):
            return make_tform(kind, args[0].A)
        return make_tform(kind, _matrix_arg(args[0], kind).T)
    builtin(kind)(fn)


_old_tform("affine2d")
_old_tform("projective2d")


def _new_tform(kind: str):
    def fn(interp, args, nargout):
        nargs(args, 0, 1, kind)
        return make_tform(kind, _matrix_arg(args[0], kind) if args else np.eye(3))
    builtin(kind)(fn)


_new_tform("affinetform2d")
_new_tform("projtform2d")


def _rot(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


@builtin("rigidtform2d")
def _rigidtform2d(interp, args, nargout):
    nargs(args, 0, 2, "rigidtform2d")
    A = np.eye(3)
    if len(args) == 1:
        A = _matrix_arg(args[0], "rigidtform2d").copy()
    elif len(args) == 2:
        r = _img(args[0], "rigidtform2d", "rotation").astype(float)
        A[:2, :2] = _rot(float(r.flat[0])) if r.size == 1 else r
        A[:2, 2] = _vec(args[1], "translation")[:2]
    tf = make_tform("rigidtform2d", A)
    lin = A[:2, :2]
    if not np.allclose(lin @ lin.T, np.eye(2), atol=1e-6) or np.linalg.det(lin) < 0:
        raise MatlabError("rigidtform2d: the rotation part must be a rotation matrix.")
    return tf


@builtin("simtform2d")
def _simtform2d(interp, args, nargout):
    nargs(args, 0, 3, "simtform2d")
    A = np.eye(3)
    if len(args) == 1:
        A = _matrix_arg(args[0], "simtform2d").copy()
    elif len(args) == 3:
        scale = _scalar(args[0], "scale")
        r = _img(args[1], "simtform2d", "rotation").astype(float)
        A[:2, :2] = scale * (_rot(float(r.flat[0])) if r.size == 1 else r)
        A[:2, 2] = _vec(args[2], "translation")[:2]
    elif len(args) == 2:
        raise MatlabError("simtform2d: expected simtform2d(scale, rotation, translation).")
    return make_tform("simtform2d", A)


@builtin("transltform2d")
def _transltform2d(interp, args, nargout):
    nargs(args, 0, 2, "transltform2d")
    A = np.eye(3)
    if len(args) == 1:
        v = _img(args[0], "transltform2d", "translation").astype(float)
        if v.shape == (3, 3):
            A = v.copy()
        else:
            A[:2, 2] = v.ravel()[:2]
    elif len(args) == 2:
        A[:2, 2] = [_scalar(args[0], "x translation"), _scalar(args[1], "y translation")]
    return make_tform("transltform2d", A)


def _tform_arg(v, name: str) -> GeoTform:
    if isinstance(v, GeoTform):
        return v
    if isinstance(v, Struct) and "tdata" in v.fields:
        td = v.data.flat[0]["tdata"]
        T = np.asarray(td.data.flat[0]["T"], dtype=float)
        kind = "projective2d" if not np.allclose(T[:, 2], [0, 0, 1]) else "affine2d"
        return GeoTform(kind, T.T)
    raise MatlabError(f"{name}: expected a geometric transformation such as affine2d or projective2d.")


def _points_call(name: str, forward: bool):
    def fn(interp, args, nargout):
        nargs(args, 2, 3, name)
        tf = _tform_arg(args[0], name)
        method = "transformPointsForward" if forward else "transformPointsInverse"
        out = tf.call_method(interp, method, [tf, *args[1:]], nargout)
        if len(out) == 1 and nargout > 1:
            return out[0][:, 0:1], out[0][:, 1:2]
        return tuple(out) if len(out) > 1 else out[0]
    builtin(name)(fn)


_points_call("tformfwd", True)
_points_call("tforminv", False)


def _fit_similarity(u: np.ndarray, x: np.ndarray, reflect: bool) -> np.ndarray:
    uu = u.copy()
    if reflect:
        uu[:, 1] = -uu[:, 1]
    n = u.shape[0]
    M = np.zeros((2 * n, 4))
    M[:n] = np.column_stack([uu[:, 0], -uu[:, 1], np.ones(n), np.zeros(n)])
    M[n:] = np.column_stack([uu[:, 1], uu[:, 0], np.zeros(n), np.ones(n)])
    a, b, tx, ty = np.linalg.lstsq(M, np.concatenate([x[:, 0], x[:, 1]]), rcond=None)[0]
    A = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
    if reflect:
        A = A @ np.diag([1.0, -1.0, 1.0])
    return A


def _fit_matrix(moving: np.ndarray, fixed: np.ndarray, kind: str, name: str) -> np.ndarray:
    """Least-squares matrix A with fixed ~ A * moving."""
    need = {"nonreflective": 2, "similarity": 3, "affine": 3, "projective": 4}[kind]
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 2:
        raise MatlabError(f"{name}: the two point sets must be n-by-2 matrices of the same size.")
    n = moving.shape[0]
    if n < need:
        raise MatlabError(f"{name}: this transformation needs at least {need} point pairs.")
    if kind == "nonreflective":
        return _fit_similarity(moving, fixed, False)
    if kind == "similarity":
        cands = [_fit_similarity(moving, fixed, r) for r in (False, True)]
        errs = [np.sum((GeoTform("affine2d", c).forward(moving) - fixed) ** 2) for c in cands]
        return cands[int(np.argmin(errs))]
    if kind == "affine":
        M = np.column_stack([moving, np.ones(n)])
        sol = np.linalg.lstsq(M, fixed, rcond=None)[0]
        return np.vstack([sol.T, [0, 0, 1.0]])

    def normalise(p):
        c = p.mean(axis=0)
        s = np.sqrt(2) / max(float(np.mean(np.hypot(*(p - c).T))), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

    Nu, Nx = normalise(moving), normalise(fixed)
    un = GeoTform("projective2d", Nu).forward(moving)
    xn = GeoTform("projective2d", Nx).forward(fixed)
    rows = []
    for (a, b), (x, y) in zip(un, xn):
        rows.append([a, b, 1, 0, 0, 0, -x * a, -x * b, -x])
        rows.append([0, 0, 0, a, b, 1, -y * a, -y * b, -y])
    _u, _s, vt = np.linalg.svd(np.array(rows))
    H = vt[-1].reshape(3, 3)
    A = np.linalg.inv(Nx) @ H @ Nu
    return A / A[2, 2]


def _fit_points(args: list, name: str) -> tuple[np.ndarray, np.ndarray, str]:
    nargs(args, 3, 4, name)
    moving = _img(args[0], name, "moving points").astype(float)
    fixed = _img(args[1], name, "fixed points").astype(float)
    return moving, fixed, text(args[2]).lower().replace(" ", "").replace("-", "")


@builtin("fitgeotrans")
def _fitgeotrans(interp, args, nargout):
    moving, fixed, kind = _fit_points(list(args), "fitgeotrans")
    table = {"nonreflectivesimilarity": "nonreflective", "similarity": "similarity", "affine": "affine",
             "projective": "projective"}
    if kind not in table:
        raise MatlabError(f"fitgeotrans: the transformation type '{text(args[2])}' is not supported. Use "
                          "'nonreflectivesimilarity', 'similarity', 'affine' or 'projective'.")
    A = _fit_matrix(moving, fixed, table[kind], "fitgeotrans")
    return make_tform("projective2d" if kind == "projective" else "affine2d", A)


@builtin("fitgeotform2d")
def _fitgeotform2d(interp, args, nargout):
    moving, fixed, kind = _fit_points(list(args), "fitgeotform2d")
    table = {"similarity": ("nonreflective", "simtform2d"), "reflectivesimilarity": ("similarity", "affinetform2d"),
             "affine": ("affine", "affinetform2d"), "projective": ("projective", "projtform2d")}
    if kind not in table:
        raise MatlabError(f"fitgeotform2d: the transformation type '{text(args[2])}' is not supported. Use "
                          "'similarity', 'reflectivesimilarity', 'affine' or 'projective'.")
    fit, cls = table[kind]
    return make_tform(cls, _fit_matrix(moving, fixed, fit, "fitgeotform2d"))


def _tform_struct(T: np.ndarray) -> Struct:
    Tinv = np.linalg.inv(T)
    td = Struct.scalar({"T": T, "Tinv": Tinv})
    return Struct.scalar({"ndims_in": _f(2), "ndims_out": _f(2),
                          "forward_fcn": FunctionHandle(name="tformfwd", text="@tformfwd"),
                          "inverse_fcn": FunctionHandle(name="tforminv", text="@tforminv"), "tdata": td})


@builtin("maketform")
def _maketform(interp, args, nargout):
    nargs(args, 2, 3, "maketform")
    kind = text(args[0]).lower()
    if kind not in ("affine", "projective"):
        raise MatlabError("maketform: the transformation type must be 'affine' or 'projective'.")
    if len(args) == 3:
        u = _img(args[1], "maketform", "input points").astype(float)
        x = _img(args[2], "maketform", "output points").astype(float)
        need = 3 if kind == "affine" else 4
        if u.shape != (need, 2) or x.shape != (need, 2):
            raise MatlabError(f"maketform: give {need} input points and {need} output points as {need}-by-2 "
                              "matrices.")
        A = _fit_matrix(u, x, kind, "maketform")
        return _tform_struct(A.T)
    T = _img(args[1], "maketform", "transformation matrix").astype(float)
    if T.shape == (3, 2):
        T = np.column_stack([T, [0, 0, 1.0]])
    if T.shape != (3, 3):
        raise MatlabError("maketform: the transformation matrix must be 3-by-3 (or 3-by-2 for 'affine').")
    if kind == "affine" and not np.allclose(T[:, 2], [0, 0, 1]):
        raise MatlabError("maketform: the last column of an affine matrix must be [0 0 1].")
    return _tform_struct(T)


@builtin("cp2tform")
def _cp2tform(interp, args, nargout):
    nargs(args, 3, 4, "cp2tform")
    moving, fixed, kind = _fit_points(list(args), "cp2tform")
    table = {"nonreflectivesimilarity": "nonreflective", "similarity": "similarity", "affine": "affine",
             "projective": "projective"}
    if kind not in table:
        raise MatlabError(f"cp2tform: the transformation type '{text(args[2])}' is not supported.")
    return _tform_struct(_fit_matrix(moving, fixed, table[kind], "cp2tform").T)


# -- spatial referencing ------------------------------------------------------------------------------------

class ImRef2d(MObject):
    """Where an image sits in world coordinates (the object imref2d returns)."""

    _METHODS = {"worldToIntrinsic", "intrinsicToWorld", "worldToSubscript", "contains", "sizesMatch"}

    def __init__(self, size: tuple[int, int], xlim: tuple[float, float], ylim: tuple[float, float]):
        self.size = (int(size[0]), int(size[1]))
        self.xlim = (float(xlim[0]), float(xlim[1]))
        self.ylim = (float(ylim[0]), float(ylim[1]))

    def class_name(self) -> str:
        return "imref2d"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    @property
    def dx(self) -> float:
        return (self.xlim[1] - self.xlim[0]) / max(self.size[1], 1)

    @property
    def dy(self) -> float:
        return (self.ylim[1] - self.ylim[0]) / max(self.size[0], 1)

    def _props(self) -> dict:
        return {"XWorldLimits": np.array([self.xlim]), "YWorldLimits": np.array([self.ylim]),
                "ImageSize": np.array([[float(self.size[0]), float(self.size[1])]]),
                "PixelExtentInWorldX": _f(self.dx), "PixelExtentInWorldY": _f(self.dy),
                "ImageExtentInWorldX": _f(self.xlim[1] - self.xlim[0]),
                "ImageExtentInWorldY": _f(self.ylim[1] - self.ylim[0]),
                "XIntrinsicLimits": np.array([[0.5, self.size[1] + 0.5]]),
                "YIntrinsicLimits": np.array([[0.5, self.size[0] + 0.5]])}

    def get_field(self, interp, name: str):
        props = self._props()
        if name in props:
            return props[name]
        raise MatlabError(f"Unrecognized property '{name}' for class 'imref2d'.")

    def set_field(self, interp, name: str, value):
        v = np.asarray(value, dtype=float).ravel()
        if name == "XWorldLimits" and v.size == 2:
            return ImRef2d(self.size, (v[0], v[1]), self.ylim)
        if name == "YWorldLimits" and v.size == 2:
            return ImRef2d(self.size, self.xlim, (v[0], v[1]))
        if name == "ImageSize" and v.size >= 2:
            return ImRef2d((int(v[0]), int(v[1])), self.xlim, self.ylim)
        raise MatlabError(f"Cannot set property '{name}' of class 'imref2d'.")

    def to_intrinsic(self, x, y):
        return 0.5 + (x - self.xlim[0]) / self.dx, 0.5 + (y - self.ylim[0]) / self.dy

    def to_world(self, xi, yi):
        return self.xlim[0] + (xi - 0.5) * self.dx, self.ylim[0] + (yi - 0.5) * self.dy

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "sizesMatch":
            a = rest[0]
            return [np.array([[isinstance(a, np.ndarray) and a.shape[:2] == self.size]])]
        x, y = rest[0].astype(float), rest[1].astype(float)
        if name == "worldToIntrinsic":
            return list(self.to_intrinsic(x, y))
        if name == "intrinsicToWorld":
            return list(self.to_world(x, y))
        inside = (x >= self.xlim[0]) & (x <= self.xlim[1]) & (y >= self.ylim[0]) & (y <= self.ylim[1])
        if name == "contains":
            return [inside]
        xi, yi = self.to_intrinsic(x, y)
        r = np.where(inside, np.clip(np.floor(yi + 0.5), 1, self.size[0]), np.nan)
        c = np.where(inside, np.clip(np.floor(xi + 0.5), 1, self.size[1]), np.nan)
        return [r, c]

    def display_lines(self) -> list[str]:
        out = ["  imref2d with properties:", ""]
        for k, v in self._props().items():
            vals = " ".join(f"{x:g}" for x in np.ravel(v))
            out.append(f"    {k.rjust(20)}: {'[' + vals + ']' if np.size(v) > 1 else vals}")
        return out


@builtin("imref2d")
def _imref2d(interp, args, nargout):
    nargs(args, 0, 3, "imref2d")
    if not args:
        return ImRef2d((2, 2), (0.5, 2.5), (0.5, 2.5))
    size = _vec(args[0], "image size").astype(int)
    if size.size < 2 or np.any(size[:2] < 0):
        raise MatlabError("imref2d: the image size must be [rows cols].")
    rows, cols = int(size[0]), int(size[1])
    if len(args) == 1:
        return ImRef2d((rows, cols), (0.5, cols + 0.5), (0.5, rows + 0.5))
    if len(args) != 3:
        raise MatlabError("imref2d: expected imref2d(size), imref2d(size, pixelExtentX, pixelExtentY) or "
                          "imref2d(size, xWorldLimits, yWorldLimits).")
    a, b = _vec(args[1], "x extent"), _vec(args[2], "y extent")
    if a.size == 1 and b.size == 1:
        return ImRef2d((rows, cols), (a[0] / 2, a[0] / 2 + cols * a[0]), (b[0] / 2, b[0] / 2 + rows * b[0]))
    if a.size == 2 and b.size == 2 and a[1] > a[0] and b[1] > b[0]:
        return ImRef2d((rows, cols), (a[0], a[1]), (b[0], b[1]))
    raise MatlabError("imref2d: world limits must be [low high] with low < high.")


# -- resampling -------------------------------------------------------------------------------------------

def _keys(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    inner = 1.5 * a ** 3 - 2.5 * a ** 2 + 1
    outer = -0.5 * a ** 3 + 2.5 * a ** 2 - 4 * a + 2
    return np.where(a <= 1, inner, np.where(a < 2, outer, 0))


def _sample(plane: np.ndarray, xi: np.ndarray, yi: np.ndarray, method: str, fill: float) -> np.ndarray:
    """Values of a 2-D array at intrinsic coordinates (xi, yi), one-based; outside the image gives fill."""
    H, W = plane.shape
    inside = (xi >= 0.5) & (xi <= W + 0.5) & (yi >= 0.5) & (yi <= H + 0.5)
    x = np.clip(xi, 1, W) - 1
    y = np.clip(yi, 1, H) - 1
    x = np.where(np.isfinite(x), x, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    if method == "nearest":
        out = plane[np.floor(y + 0.5).astype(int).clip(0, H - 1), np.floor(x + 0.5).astype(int).clip(0, W - 1)]
    elif method == "linear":
        x0 = np.floor(x).astype(int).clip(0, max(W - 2, 0))
        y0 = np.floor(y).astype(int).clip(0, max(H - 2, 0))
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        fx, fy = x - x0, y - y0
        out = ((1 - fy) * ((1 - fx) * plane[y0, x0] + fx * plane[y0, x1])
               + fy * ((1 - fx) * plane[y1, x0] + fx * plane[y1, x1]))
    else:
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        out = np.zeros(x.shape)
        for dy in range(-1, 3):
            wy = _keys(y - (y0 + dy))
            yy = (y0 + dy).clip(0, H - 1)
            for dx in range(-1, 3):
                out += wy * _keys(x - (x0 + dx)) * plane[yy, (x0 + dx).clip(0, W - 1)]
    return np.where(inside, out, fill)


def _out_limits(tf: GeoTform, xl: np.ndarray, yl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tf.is_affine():
        pts = np.array([[xl[0], yl[0]], [xl[1], yl[0]], [xl[0], yl[1]], [xl[1], yl[1]]])
    else:
        s = np.linspace(0, 1, 101)
        xs, ys = xl[0] + s * (xl[1] - xl[0]), yl[0] + s * (yl[1] - yl[0])
        lo_y, hi_y = np.full(s.size, yl[0]), np.full(s.size, yl[1])
        lo_x, hi_x = np.full(s.size, xl[0]), np.full(s.size, xl[1])
        pts = np.concatenate([np.column_stack([xs, lo_y]), np.column_stack([xs, hi_y]),
                              np.column_stack([lo_x, ys]), np.column_stack([hi_x, ys])])
    out = tf.forward(pts)
    return np.array([out[:, 0].min(), out[:, 0].max()]), np.array([out[:, 1].min(), out[:, 1].max()])


_INTERP = {"nearest": "nearest", "linear": "linear", "bilinear": "linear", "cubic": "cubic", "bicubic": "cubic"}


def _fill_values(v, channels: int, name: str) -> np.ndarray:
    if v is None:
        return np.zeros(channels)
    f = _vec(v, "fill values")
    if f.size == 1:
        return np.repeat(f, channels)
    if f.size != channels:
        raise MatlabError(f"{name}: give one fill value, or one per colour channel.")
    return f


def _warp(a: np.ndarray, to_input, out_shape: tuple[int, int], method: str, fills: np.ndarray) -> np.ndarray:
    """Resample image a: to_input(rows, cols) gives the intrinsic input coordinates of every output pixel."""
    xi, yi = to_input()
    flat = a.reshape(a.shape[0], a.shape[1], -1)
    planes = []
    for k in range(flat.shape[2]):
        planes.append(_sample(flat[..., k].astype(float), xi, yi, method, float(fills[min(k, fills.size - 1)])))
    out = np.stack(planes, axis=-1).reshape(out_shape + a.shape[2:])
    return _like_class(out, a)


@builtin("imwarp")
def _imwarp(interp, args, nargout):
    rest, opts = _options(list(args), ("OutputView", "FillValues", "SmoothEdges", "Interp"), "imwarp")
    nargs(rest, 2, 4, "imwarp")
    a = _img(rest[0], "imwarp")
    if a.ndim < 2 or a.size == 0:
        raise MatlabError("imwarp: the image must not be empty.")
    pos = 1
    ra = None
    if isinstance(rest[pos], ImRef2d):
        ra = rest[pos]
        pos += 1
    tf = _tform_arg(rest[pos], "imwarp")
    method = "linear"
    if len(rest) > pos + 1:
        method = text(rest[pos + 1]).lower()
    if "interp" in opts:
        method = text(opts["interp"]).lower()
    if method not in _INTERP:
        raise MatlabError("imwarp: the interpolation must be 'nearest', 'linear' or 'cubic'.")
    method = _INTERP[method]
    if ra is None:
        ra = ImRef2d(a.shape[:2], (0.5, a.shape[1] + 0.5), (0.5, a.shape[0] + 0.5))
    elif ra.size != a.shape[:2]:
        raise MatlabError("imwarp: the spatial reference does not match the size of the image.")
    if "outputview" in opts:
        rb = opts["outputview"]
        if not isinstance(rb, ImRef2d):
            raise MatlabError("imwarp: 'OutputView' must be an imref2d object.")
    else:
        xo, yo = _out_limits(tf, np.array(ra.xlim), np.array(ra.ylim))
        ncols = max(int(np.ceil((xo[1] - xo[0]) / ra.dx - 1e-9)), 1)
        nrows = max(int(np.ceil((yo[1] - yo[0]) / ra.dy - 1e-9)), 1)
        nx = (ncols * ra.dx - (xo[1] - xo[0])) / 2
        ny = (nrows * ra.dy - (yo[1] - yo[0])) / 2
        rb = ImRef2d((nrows, ncols), (xo[0] - nx, xo[1] + nx), (yo[0] - ny, yo[1] + ny))
    channels = int(np.prod(a.shape[2:])) if a.ndim > 2 else 1
    fills = _fill_values(opts.get("fillvalues"), channels, "imwarp")

    def to_input():
        cc, rr = np.meshgrid(np.arange(1, rb.size[1] + 1, dtype=float), np.arange(1, rb.size[0] + 1, dtype=float))
        xw, yw = rb.to_world(cc, rr)
        src = tf.inverse(np.column_stack([xw.ravel(), yw.ravel()]))
        xi, yi = ra.to_intrinsic(src[:, 0].reshape(cc.shape), src[:, 1].reshape(cc.shape))
        return xi, yi

    out = _warp(a, to_input, rb.size, method, fills)
    return (out, rb) if nargout > 1 else out


@builtin("imtransform")
def _imtransform(interp, args, nargout):
    rest, opts = _options(list(args), ("XData", "YData", "UData", "VData", "XYScale", "Size", "FillValues"),
                          "imtransform")
    nargs(rest, 2, 3, "imtransform")
    a = _img(rest[0], "imtransform")
    tf = _tform_arg(rest[1], "imtransform")
    method = text(rest[2]).lower() if len(rest) > 2 else "bilinear"
    if method not in _INTERP:
        raise MatlabError("imtransform: the interpolation must be 'nearest', 'bilinear' or 'bicubic'.")
    H, W = a.shape[:2]
    ud = _vec(opts["udata"], "UData") if "udata" in opts else np.array([1.0, W])
    vd = _vec(opts["vdata"], "VData") if "vdata" in opts else np.array([1.0, H])
    if "xdata" in opts and "ydata" in opts:
        xd, yd = _vec(opts["xdata"], "XData"), _vec(opts["ydata"], "YData")
    else:
        xd, yd = _out_limits(tf, ud, vd)
    if "xyscale" in opts:
        sc = _vec(opts["xyscale"], "XYScale")
        sx, sy = (sc[0], sc[0]) if sc.size == 1 else (sc[0], sc[1])
    else:
        sx = (ud[1] - ud[0]) / max(W - 1, 1) if W > 1 else 1.0
        sy = (vd[1] - vd[0]) / max(H - 1, 1) if H > 1 else 1.0
    if "size" in opts:
        sz = _vec(opts["size"], "Size").astype(int)
        nrows, ncols = int(sz[0]), int(sz[1])
        if "xyscale" not in opts:
            sx = (xd[1] - xd[0]) / max(ncols - 1, 1)
            sy = (yd[1] - yd[0]) / max(nrows - 1, 1)
    else:
        ncols = int(np.ceil((xd[1] - xd[0]) / sx - 1e-9)) + 1
        nrows = int(np.ceil((yd[1] - yd[0]) / sy - 1e-9)) + 1
    channels = int(np.prod(a.shape[2:])) if a.ndim > 2 else 1
    fills = _fill_values(opts.get("fillvalues"), channels, "imtransform")
    du = (ud[1] - ud[0]) / max(W - 1, 1) if W > 1 else 1.0
    dv = (vd[1] - vd[0]) / max(H - 1, 1) if H > 1 else 1.0

    def to_input():
        xs = xd[0] + sx * np.arange(ncols)
        ys = yd[0] + sy * np.arange(nrows)
        xx, yy = np.meshgrid(xs, ys)
        src = tf.inverse(np.column_stack([xx.ravel(), yy.ravel()]))
        return ((src[:, 0] - ud[0]) / du + 1).reshape(xx.shape), ((src[:, 1] - vd[0]) / dv + 1).reshape(xx.shape)

    out = _warp(a, to_input, (nrows, ncols), _INTERP[method], fills)
    if nargout > 1:
        return out, np.array([[xd[0], xd[0] + sx * (ncols - 1)]]), np.array([[yd[0], yd[0] + sy * (nrows - 1)]])
    return out


_BURT = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])


@builtin("impyramid")
def _impyramid(interp, args, nargout):
    nargs(args, 2, 2, "impyramid")
    a = _img(args[0], "impyramid")
    kind = text(args[1]).lower()
    if kind not in ("reduce", "expand"):
        raise MatlabError("impyramid: the direction must be 'reduce' or 'expand'.")
    nd = _nd()
    x = a.astype(float)

    def one_axis(z, axis):
        n = z.shape[axis]
        if kind == "reduce":
            sm = nd.correlate1d(z, _BURT, axis=axis, mode="nearest")
            sl = [slice(None)] * z.ndim
            sl[axis] = slice(0, None, 2)
            return sm[tuple(sl)]
        up_shape = list(z.shape)
        up_shape[axis] = 2 * n - 1
        up = np.zeros(up_shape)
        sl = [slice(None)] * z.ndim
        sl[axis] = slice(0, None, 2)
        up[tuple(sl)] = nd.correlate1d(z, np.array([0.125, 0.75, 0.125]), axis=axis, mode="nearest")
        if n > 1:
            lo = [slice(None)] * z.ndim
            hi = [slice(None)] * z.ndim
            lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
            sl[axis] = slice(1, None, 2)
            up[tuple(sl)] = 0.5 * (z[tuple(lo)] + z[tuple(hi)])
        return up

    out = one_axis(one_axis(x, 0), 1)
    return _like_class(out, a)


@builtin("imresize3")
def _imresize3(interp, args, nargout):
    rest, opts = _options(list(args), ("Antialiasing", "Method", "OutputSize", "Scale"), "imresize3")
    nargs(rest, 1, 3, "imresize3")
    v = _img(rest[0], "imresize3")
    if v.ndim != 3:
        raise MatlabError("imresize3: the volume must be three-dimensional.")
    method = "cubic"
    spec = None
    for r in rest[1:]:
        if is_text(r):
            method = text(r).lower()
        else:
            spec = _vec(r, "scale or size")
    if "method" in opts:
        method = text(opts["method"]).lower()
    names = {"nearest": "nearest", "linear": "bilinear", "cubic": "bicubic", "box": "box", "triangle": "triangle",
             "lanczos2": "lanczos2", "lanczos3": "lanczos3"}
    if method not in names:
        raise MatlabError("imresize3: the method must be 'nearest', 'linear', 'cubic', 'box', 'triangle', "
                          "'lanczos2' or 'lanczos3'.")
    if "outputsize" in opts:
        spec = _vec(opts["outputsize"], "OutputSize")
    if "scale" in opts:
        spec = _vec(opts["scale"], "Scale")
        scales = np.repeat(spec, 3) if spec.size == 1 else spec
        out_shape = [int(np.ceil(s * n)) for s, n in zip(scales, v.shape)]
    elif spec is None:
        raise MatlabError("imresize3: give a scale factor or an output size [rows cols planes].")
    elif spec.size == 1:
        scales = np.repeat(spec, 3)
        out_shape = [int(np.ceil(s * n)) for s, n in zip(scales, v.shape)]
    elif spec.size == 3:
        out_shape = [int(s) for s in spec]
        scales = np.array([o / n for o, n in zip(out_shape, v.shape)])
    else:
        raise MatlabError("imresize3: the output size must have three elements.")
    antialias = method != "nearest"
    if "antialiasing" in opts:
        antialias = _flag(opts["antialiasing"])
    work = v.astype(float)
    for ax in range(3):
        if out_shape[ax] != v.shape[ax] or scales[ax] != 1:
            work = _resize_axis(work, ax, out_shape[ax], float(scales[ax]), names[method], antialias)
    return _like_class(work, v)


# -- arithmetic, profiles and comparison --------------------------------------------------------------------

@builtin("imlincomb")
def _imlincomb(interp, args, nargout):
    nargs(args, 2, None, "imlincomb")
    rest = list(args)
    cls = None
    if is_text(rest[-1]):
        cls = text(rest.pop()).lower()
    offset = 0.0
    if len(rest) % 2 == 1:
        offset = _scalar(rest.pop(), "offset")
    if not rest:
        raise MatlabError("imlincomb: at least one coefficient and image are required.")
    first = _img(rest[1], "imlincomb")
    total = np.zeros(first.shape)
    for k in range(0, len(rest), 2):
        coef = _scalar(rest[k], "coefficient")
        img = _img(rest[k + 1], "imlincomb")
        if img.shape != first.shape:
            raise MatlabError("imlincomb: all images must have the same size.")
        total = total + coef * img.astype(float)
    total = total + offset
    if cls is None:
        return total if first.dtype == bool else _like_class(total, first)
    types = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "int8": np.int8, "int16": np.int16,
             "int32": np.int32, "single": np.float32, "double": np.float64}
    if cls not in types:
        raise MatlabError(f"imlincomb: unsupported output class '{cls}'.")
    return _cast(total, types[cls])


@builtin("improfile")
def _improfile(interp, args, nargout):
    rest = list(args)
    method = "nearest"
    if rest and is_text(rest[-1]):
        method = text(rest.pop()).lower()
        if method not in ("nearest", "bilinear", "bicubic"):
            raise MatlabError("improfile: the method must be 'nearest', 'bilinear' or 'bicubic'.")
    if len(rest) < 3:
        raise MatlabError("improfile: choosing the line with the mouse is not supported; give "
                          "improfile(I, xi, yi).")
    xdata = ydata = None
    if len(rest) >= 5:
        xdata, ydata = _vec(rest[0], "x data"), _vec(rest[1], "y data")
        rest = rest[2:]
    a = _img(rest[0], "improfile")
    xi, yi = _vec(rest[1], "xi"), _vec(rest[2], "yi")
    if xi.size != yi.size or xi.size < 2:
        raise MatlabError("improfile: xi and yi must have the same number of points (at least two).")
    H, W = a.shape[:2]
    if xdata is not None:
        px = 1 + (xi - xdata[0]) / (xdata[-1] - xdata[0]) * (W - 1) if W > 1 else np.ones(xi.size)
        py = 1 + (yi - ydata[0]) / (ydata[-1] - ydata[0]) * (H - 1) if H > 1 else np.ones(yi.size)
    else:
        px, py = xi, yi
    seg = np.hypot(np.diff(px), np.diff(py))
    if len(rest) > 3:
        n = to_int(rest[3], "number of points")
    else:
        n = int(np.round(np.sum(np.maximum(np.abs(np.diff(px)), np.abs(np.diff(py)))))) + 1
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n) if cum[-1] > 0 else np.zeros(n)
    sx = np.interp(t, cum, px) if cum[-1] > 0 else np.full(n, px[0])
    sy = np.interp(t, cum, py) if cum[-1] > 0 else np.full(n, py[0])
    flat = a.reshape(H, W, -1).astype(float)
    kind = {"nearest": "nearest", "bilinear": "linear", "bicubic": "cubic"}[method]
    vals = np.stack([_sample(flat[..., k], sx, sy, kind, np.nan) for k in range(flat.shape[2])], axis=-1)
    c = vals.reshape(n, 1, flat.shape[2]) if flat.shape[2] > 1 else vals.reshape(n, 1)
    if nargout <= 1:
        return c
    if xdata is not None:
        wx = xdata[0] + (sx - 1) * (xdata[-1] - xdata[0]) / max(W - 1, 1)
        wy = ydata[0] + (sy - 1) * (ydata[-1] - ydata[0]) / max(H - 1, 1)
    else:
        wx, wy = sx, sy
    out = (wx.reshape(-1, 1), wy.reshape(-1, 1), c, xi.reshape(-1, 1), yi.reshape(-1, 1))
    return out[:nargout]


# -- combining images for display ---------------------------------------------------------------------------

def _frames(v, name: str) -> list[np.ndarray]:
    if isinstance(v, Cell):
        items = [_img(x, name) for x in v.values_f()]
    else:
        a = _img(v, name)
        if a.ndim == 4:
            items = [a[..., k] for k in range(a.shape[3])]
        elif a.ndim == 3 and a.shape[2] != 3:
            items = [a[..., k] for k in range(a.shape[2])]
        else:
            items = [a]
    if not items:
        raise MatlabError(f"{name}: there are no images to show.")
    return items


def _tile(interp, items: list[np.ndarray], opts: dict, name: str) -> np.ndarray:
    first = items[0]
    th, tw = first.shape[:2]
    if "thumbnailsize" in opts and opts["thumbnailsize"].size:
        ts = _vec(opts["thumbnailsize"], "ThumbnailSize").astype(int)
        th, tw = int(ts[0]), int(ts[1])
    rgb = any(x.ndim == 3 for x in items)
    n = len(items)
    if "gridsize" in opts:
        g = _vec(opts["gridsize"], "GridSize")
        rows = int(g[0]) if np.isfinite(g[0]) else None
        cols = int(g[1]) if np.isfinite(g[1]) else None
        if rows is None and cols is None:
            raise MatlabError(f"{name}: at least one element of 'GridSize' must be a number.")
        if rows is None:
            rows = int(np.ceil(n / cols))
        if cols is None:
            cols = int(np.ceil(n / rows))
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    border = _vec(opts["bordersize"], "BorderSize").astype(int) if "bordersize" in opts else np.array([0, 0])
    bh, bw = (int(border[0]), int(border[0])) if border.size == 1 else (int(border[0]), int(border[1]))
    back = _vec(opts["backgroundcolor"], "BackgroundColor") if "backgroundcolor" in opts and \
        not is_text(opts["backgroundcolor"]) else np.zeros(3)
    dtype = first.dtype if first.dtype != bool else np.uint8
    hi = 1.0 if np.dtype(dtype).kind == "f" else float(np.iinfo(dtype).max)
    ch = 3 if rgb else 1
    cell_h, cell_w = th + 2 * bh, tw + 2 * bw
    canvas = np.empty((rows * cell_h, cols * cell_w, ch), dtype=float)
    canvas[...] = (back[:ch] if rgb else back[:1].mean()) * hi
    for k, img in enumerate(items[:rows * cols]):
        if img.shape[:2] != (th, tw):
            img = BUILTINS["imresize"](interp, [img, np.array([[float(th), float(tw)]])], 1)
        unit = _to_unit(img) * hi
        if rgb and unit.ndim == 2:
            unit = np.repeat(unit[..., None], 3, axis=2)
        r, c = divmod(k, cols)
        canvas[r * cell_h + bh:r * cell_h + bh + th, c * cell_w + bw:c * cell_w + bw + tw] = \
            unit.reshape(th, tw, ch)
    out = canvas if rgb else canvas[..., 0]
    return _cast(out, dtype) if np.dtype(dtype).kind in "iu" else out.astype(dtype)


_TILE_OPTS = ("GridSize", "BorderSize", "BackgroundColor", "ThumbnailSize", "Frames", "Size", "Indices",
              "DisplayRange", "Parent", "Interpolation")


@builtin("imtile")
def _imtile(interp, args, nargout):
    rest, opts = _options(list(args), _TILE_OPTS, "imtile")
    nargs(rest, 1, 2, "imtile")
    items = _frames(rest[0], "imtile")
    if "frames" in opts:
        idx = _vec(opts["frames"], "Frames").astype(int)
        items = [items[i - 1] for i in idx]
    return _tile(interp, items, opts, "imtile")


@builtin("montage")
def _montage(interp, args, nargout):
    rest, opts = _options(list(args), _TILE_OPTS, "montage")
    nargs(rest, 1, 2, "montage")
    items = _frames(rest[0], "montage")
    if "size" in opts and "gridsize" not in opts:
        opts["gridsize"] = opts["size"]
    for key in ("indices", "frames"):
        if key in opts:
            idx = _vec(opts[key], key).astype(int)
            items = [items[i - 1] for i in idx]
            break
    tiled = _tile(interp, items, opts, "montage")
    show = [tiled]
    if "displayrange" in opts:
        show.append(opts["displayrange"])
    h = interp.call_named("imshow", show, 1 if nargout else 0, interp.frame)
    return h[0] if nargout and h else None


def _fuse_gray(a: np.ndarray) -> np.ndarray:
    return _rgb2gray(a) if _is_rgb(a) else a


def _stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.clip(x, 0, 1)


def _fuse(interp, args: list, name: str) -> np.ndarray:
    rest, opts = _options(list(args), ("Scaling", "ColorChannels", "Parent", "Interpolation"), name)
    nargs(rest, 2, 3, name)
    a, b = _img(rest[0], name), _img(rest[1], name)
    method = text(rest[2]).lower() if len(rest) > 2 else "falsecolor"
    if method not in ("falsecolor", "blend", "diff", "checkerboard", "montage"):
        raise MatlabError(f"{name}: the method must be 'falsecolor', 'blend', 'diff', 'checkerboard' or 'montage'.")
    scaling = text(opts["scaling"]).lower() if "scaling" in opts else "independent"
    if scaling not in ("independent", "joint", "none"):
        raise MatlabError(f"{name}: 'Scaling' must be 'independent', 'joint' or 'none'.")
    keep_colour = method in ("blend", "montage", "checkerboard") and (_is_rgb(a) or _is_rgb(b))

    def prep(x):
        if keep_colour:
            u = _to_unit(x)
            return u if u.ndim == 3 else np.repeat(u[..., None], 3, axis=2)
        g = _fuse_gray(x)
        return g.astype(float) if scaling != "none" else _to_unit(g)

    A, B = prep(a), prep(b)
    if not keep_colour:
        if scaling == "independent":
            A, B = _stretch(A, A.min(), A.max()), _stretch(B, B.min(), B.max())
        elif scaling == "joint":
            lo, hi = min(A.min(), B.min()), max(A.max(), B.max())
            A, B = _stretch(A, lo, hi), _stretch(B, lo, hi)
    rows, cols = max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1])

    def grow(x):
        pad = [(0, rows - x.shape[0]), (0, cols - x.shape[1])] + [(0, 0)] * (x.ndim - 2)
        return np.pad(x, pad, mode="constant")

    if method == "montage":
        def tall(x):
            return np.pad(x, [(0, rows - x.shape[0]), (0, 0)] + [(0, 0)] * (x.ndim - 2), mode="constant")

        out = np.concatenate([tall(A), tall(B)], axis=1)
    else:
        A, B = grow(A), grow(B)
        if method == "falsecolor":
            cc = opts.get("colorchannels")
            if cc is None or (is_text(cc) and text(cc).lower() == "green-magenta"):
                order = [2, 1, 2]
            elif is_text(cc) and text(cc).lower() == "red-cyan":
                order = [1, 2, 2]
            elif is_text(cc):
                raise MatlabError(f"{name}: 'ColorChannels' must be 'green-magenta', 'red-cyan' or [R G B].")
            else:
                order = [int(v) for v in _vec(cc, "ColorChannels")]
                if len(order) != 3 or any(v not in (0, 1, 2) for v in order):
                    raise MatlabError(f"{name}: a 'ColorChannels' vector needs three entries from 0, 1 and 2.")
            src = {0: np.zeros(A.shape), 1: A, 2: B}
            out = np.stack([src[k] for k in order], axis=-1)
        elif method == "blend":
            out = (A + B) / 2
        elif method == "diff":
            d = np.abs(A - B)
            out = _stretch(d, d.min(), d.max()) if scaling != "none" else d
        else:
            tile = max(int(np.ceil(max(rows, cols) / 16)), 1)
            rr, cc_ = np.indices((rows, cols))
            pick = ((rr // tile) + (cc_ // tile)) % 2 == 0
            out = np.where(pick[..., None] if A.ndim == 3 else pick, A, B)
    return _from_unit(np.clip(out, 0, 1), np.uint8)


@builtin("imfuse")
def _imfuse(interp, args, nargout):
    out = _fuse(interp, list(args), "imfuse")
    if nargout > 1:
        return out, ImRef2d(out.shape[:2], (0.5, out.shape[1] + 0.5), (0.5, out.shape[0] + 0.5))
    return out


@builtin("imshowpair")
def _imshowpair(interp, args, nargout):
    out = _fuse(interp, list(args), "imshowpair")
    h = interp.call_named("imshow", [out], 1 if nargout else 0, interp.frame)
    return h[0] if nargout and h else None


# -- DICOM files (through the optional pydicom package) ---------------------------------------------------

def _pydicom():
    try:
        import pydicom
    except ImportError:
        raise MatlabError("Reading DICOM files needs the pydicom package (pip install pydicom).")
    return pydicom


def _dicom_file(v, name: str) -> str:
    if isinstance(v, Struct) and "Filename" in v.fields:
        return text(v.data.flat[0]["Filename"])
    if not is_text(v):
        raise MatlabError(f"{name}: the file name must be text.")
    return text(v)


@builtin("dicomread")
def _dicomread(interp, args, nargout):
    nargs(args, 1, None, "dicomread")
    pydicom = _pydicom()
    ds = pydicom.dcmread(_dicom_file(args[0], "dicomread"))
    px = np.asarray(ds.pixel_array)
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frames > 1:
        px = np.moveaxis(px, 0, -1)
        if px.ndim == 3:
            px = px[:, :, None, :]
    cmap = np.zeros((0, 0))
    return (px, cmap)[:nargout] if nargout > 1 else px


@builtin("dicominfo")
def _dicominfo(interp, args, nargout):
    nargs(args, 1, None, "dicominfo")
    pydicom = _pydicom()
    fname = _dicom_file(args[0], "dicominfo")
    ds = pydicom.dcmread(fname, stop_before_pixels=True)
    info = Struct.scalar({"Filename": char(fname)})

    def convert(val):
        if isinstance(val, (int, float)):
            return _f(val)
        if isinstance(val, (list, tuple)) or type(val).__name__ == "MultiValue":
            try:
                return np.array([[float(x) for x in val]]).reshape(-1, 1)
            except (TypeError, ValueError):
                return char("\\".join(str(x) for x in val))
        try:
            return _f(float(val)) if type(val).__name__ in ("DSfloat", "IS", "DSdecimal") else char(str(val))
        except (TypeError, ValueError):
            return char(str(val))

    for elem in ds:
        key = elem.keyword
        if not key or elem.VR == "SQ":
            continue
        info.set_field(0, key, convert(elem.value))
    if "Columns" in info.fields:
        info.set_field(0, "Width", info.get_field(0, "Columns"))
    if "Rows" in info.fields:
        info.set_field(0, "Height", info.get_field(0, "Rows"))
    return info
