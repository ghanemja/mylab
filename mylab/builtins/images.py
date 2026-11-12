"""Image files and image processing.

Images are H-by-W (grayscale, binary or indexed) or H-by-W-by-3 (RGB) arrays. As in MATLAB the
value range follows the class: 0..255 for uint8, 0..65535 for uint16, 0..1 for double and single.
"""

from __future__ import annotations

import os

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, as_2d, char, is_char, to_int
from ._registry import BUILTINS, builtin
from ._util import is_text, nargs, text

_RANGE = {"uint8": (0.0, 255.0), "uint16": (0.0, 65535.0), "int16": (-32768.0, 32767.0),
          "int8": (-128.0, 127.0), "uint32": (0.0, 4294967295.0), "int32": (-2147483648.0, 2147483647.0)}


# -- helpers ------------------------------------------------------------------------------------

def _img(v, name: str, what: str = "image") -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf":
        raise MatlabError(f"{name}: the {what} must be a numeric or logical array.")
    return v


def _class_range(a: np.ndarray) -> tuple[float, float]:
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _to_unit(a: np.ndarray) -> np.ndarray:
    """Double image scaled to 0..1 (the im2double rule)."""
    if a.dtype == bool:
        return a.astype(float)
    if a.dtype.kind == "f":
        return a.astype(float)
    lo, hi = _class_range(a)
    return (a.astype(float) - lo) / (hi - lo)


def _round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _cast_like(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Put a double result back into the class of ``like`` (rounding and saturating integers)."""
    return _cast(x, like.dtype)


def _cast(x: np.ndarray, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return x > 0.5 if x.dtype != bool else x
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        x = np.where(np.isnan(x), 0.0, x)
        return np.clip(_round_half_away(x), info.min, info.max).astype(dtype)
    return x.astype(dtype)


def _from_unit(x: np.ndarray, dtype) -> np.ndarray:
    """Scale a 0..1 double image into the range of ``dtype``."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return x > 0.5
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return _cast(x * (float(info.max) - float(info.min)) + float(info.min), dtype)
    return x.astype(dtype)


def _is_rgb(a: np.ndarray) -> bool:
    return a.ndim == 3 and a.shape[2] == 3


def _gray2d(a: np.ndarray, name: str) -> np.ndarray:
    """A 2-D image; RGB input is reduced to luminance."""
    if _is_rgb(a):
        return _rgb2gray(a)
    if a.ndim != 2:
        raise MatlabError(f"{name}: the image must be a 2-D grayscale image or an M-by-N-by-3 RGB image.")
    return a


def _scalar(v, what: str) -> float:
    if not isinstance(v, np.ndarray) or v.size != 1 or v.dtype.kind not in "biuf":
        raise MatlabError(f"Expected {what} to be a numeric scalar.")
    return float(v.flat[0])


def _vec(v, what: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf":
        raise MatlabError(f"Expected {what} to be numeric.")
    return v.astype(float).flatten(order="F")


def _options(args: list, names: tuple[str, ...], name: str) -> tuple[list, dict]:
    """Split trailing name-value pairs (names are matched without regard to case)."""
    lower = {n.lower(): n for n in names}
    rest, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if is_text(a) and text(a).lower() in lower:
            if i + 1 >= len(args):
                raise MatlabError(f"{name}: the option '{text(a)}' needs a value.")
            opts[text(a).lower()] = args[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return rest, opts


def _nd():
    from scipy import ndimage
    return ndimage


def _pil():
    try:
        from PIL import Image
    except ImportError:
        raise MatlabError("Reading and writing image files needs the Pillow package (pip install pillow).")
    return Image


def _col(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, 1)


# -- class conversion ---------------------------------------------------------------------------

def _im2(name: str, dtype):
    def fn(interp, args, nargout):
        nargs(args, 1, 2, name)
        a = _img(args[0], name)
        if len(args) > 1:
            if text(args[1]).lower() != "indexed":
                raise MatlabError(f"{name}: the only option is 'indexed'.")
            # indexed images keep their index values; only the zero/one base moves
            if np.dtype(dtype).kind == "f":
                return a.astype(dtype) + (1 if a.dtype.kind in "iu" else 0)
            return _cast(a.astype(float) - (0 if a.dtype.kind in "iu" else 1), dtype)
        if a.dtype == np.dtype(dtype):
            return a
        if a.dtype.kind == "f" and np.dtype(dtype).kind == "f":
            return a.astype(dtype)
        return _from_unit(_to_unit(a), dtype)
    return fn


builtin("im2double")(_im2("im2double", np.float64))
builtin("im2single")(_im2("im2single", np.float32))
builtin("im2uint8")(_im2("im2uint8", np.uint8))
builtin("im2uint16")(_im2("im2uint16", np.uint16))
builtin("im2int16")(_im2("im2int16", np.int16))


@builtin("mat2gray")
def _mat2gray(interp, args, nargout):
    nargs(args, 1, 2, "mat2gray")
    a = _img(args[0], "mat2gray").astype(float)
    if len(args) > 1 and args[1].size:
        lim = _vec(args[1], "limits")
        if lim.size != 2:
            raise MatlabError("mat2gray: the limits must be [low high].")
        lo, hi = lim
    elif a.size:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    else:
        return a
    if hi == lo:
        return np.clip(a, 0.0, 1.0)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


# luminance weights of the NTSC (Rec. 601) colour standard
_LUMA = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])


def _rgb2gray(a: np.ndarray) -> np.ndarray:
    g = a[..., :3].astype(float) @ _LUMA
    if a.dtype.kind == "f":
        return g.astype(a.dtype)
    return _cast(g, a.dtype)


@builtin("rgb2gray", "im2gray")
def _rgb2gray_b(interp, args, nargout):
    nargs(args, 1, 1, "rgb2gray")
    a = _img(args[0], "rgb2gray")
    if a.ndim == 2 and a.shape[1] == 3 and a.dtype.kind == "f" and a.size and a.min() >= 0 and a.max() <= 1:
        g = a @ _LUMA  # a colormap: keep it a colormap
        return np.repeat(g.reshape(-1, 1), 3, axis=1)
    if a.ndim == 2:
        return a
    if not _is_rgb(a):
        raise MatlabError("rgb2gray: the input must be an M-by-N-by-3 RGB image or a colormap.")
    return _rgb2gray(a)


def _rgb_parts(args, name: str):
    """(float array of shape (..., 3), restore function, was colormap) for colour conversions."""
    if len(args) == 3:
        parts = [np.asarray(_img(a, name), dtype=float) for a in args]
        a = np.stack(parts, axis=-1)
        return a, "planes"
    a = _img(args[0], name)
    if a.ndim == 2 and a.shape[1] == 3:
        return _to_unit(a) if a.dtype.kind != "f" else a.astype(float), "map"
    if _is_rgb(a):
        return _to_unit(a) if a.dtype.kind != "f" else a.astype(float), "image"
    raise MatlabError(f"{name}: the input must be an M-by-N-by-3 image or an N-by-3 colormap.")


def _color_out(out: np.ndarray, kind: str, nargout: int):
    if nargout > 1:
        return tuple(as_2d(out[..., k]) for k in range(min(nargout, 3)))
    return out


@builtin("rgb2hsv")
def _rgb2hsv(interp, args, nargout):
    nargs(args, 1, 3, "rgb2hsv")
    a, kind = _rgb_parts(args, "rgb2hsv")
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    v = a.max(axis=-1)
    span = v - a.min(axis=-1)
    safe = np.where(span == 0, 1.0, span)
    h = np.where(v == r, (g - b) / safe, np.where(v == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe))
    h = np.where(span == 0, 0.0, (h / 6.0) % 1.0)
    s = np.where(v == 0, 0.0, span / np.where(v == 0, 1.0, v))
    return _color_out(np.stack([h, s, v], axis=-1), kind, nargout)


@builtin("hsv2rgb")
def _hsv2rgb(interp, args, nargout):
    nargs(args, 1, 3, "hsv2rgb")
    a, kind = _rgb_parts(args, "hsv2rgb")
    h, s, v = a[..., 0], a[..., 1], a[..., 2]
    h6 = (h % 1.0) * 6.0
    h6 = np.where(h == 1.0, 0.0, h6)
    k = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(k, [v, q, p, p, t, v])
    g = np.choose(k, [t, v, v, q, p, p])
    b = np.choose(k, [p, p, t, v, v, q])
    return _color_out(np.stack([r, g, b], axis=-1), kind, nargout)


def _index_base(x: np.ndarray) -> np.ndarray:
    """Zero-based palette positions of an indexed image (integer classes count from 0)."""
    if x.dtype.kind in "iu" or x.dtype == bool:
        return x.astype(np.int64)
    return np.round(x).astype(np.int64) - 1


@builtin("ind2rgb")
def _ind2rgb(interp, args, nargout):
    nargs(args, 2, 2, "ind2rgb")
    x = _img(args[0], "ind2rgb")
    cmap = _img(args[1], "ind2rgb", "colormap").astype(float)
    if cmap.ndim != 2 or cmap.shape[1] != 3:
        raise MatlabError("ind2rgb: the colormap must have three columns.")
    idx = np.clip(_index_base(x), 0, cmap.shape[0] - 1)
    return cmap[idx]


@builtin("ind2gray")
def _ind2gray(interp, args, nargout):
    nargs(args, 2, 2, "ind2gray")
    x = _img(args[0], "ind2gray")
    cmap = _img(args[1], "ind2gray", "colormap").astype(float)
    idx = np.clip(_index_base(x), 0, cmap.shape[0] - 1)
    g = (cmap @ _LUMA)[idx]
    return _from_unit(g, x.dtype) if x.dtype.kind in "iu" else g


def _index_out(idx: np.ndarray, n: int) -> np.ndarray:
    """Indexed image in the class MATLAB uses: zero-based uint8/uint16, else one-based double."""
    if n <= 256:
        return idx.astype(np.uint8)
    if n <= 65536:
        return idx.astype(np.uint16)
    return idx.astype(float) + 1


@builtin("gray2ind")
def _gray2ind(interp, args, nargout):
    nargs(args, 1, 2, "gray2ind")
    a = _img(args[0], "gray2ind")
    n = to_int(args[1], "number of colours") if len(args) > 1 else (2 if a.dtype == bool else 64)
    if n < 1 or n > 65536:
        raise MatlabError("gray2ind: the number of colours must be between 1 and 65536.")
    idx = np.clip(_round_half_away(_to_unit(a) * (n - 1)), 0, n - 1)
    ramp = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    cmap = np.repeat(ramp.reshape(-1, 1), 3, axis=1)
    x = _index_out(idx, n)
    return (x, cmap) if nargout > 1 else x


@builtin("rgb2ind")
def _rgb2ind(interp, args, nargout):
    nargs(args, 2, 3, "rgb2ind")
    a = _img(args[0], "rgb2ind")
    if not _is_rgb(a):
        raise MatlabError("rgb2ind: the first input must be an M-by-N-by-3 RGB image.")
    rgb8 = _from_unit(_to_unit(a), np.uint8)
    spec = _img(args[1], "rgb2ind", "second input")
    if spec.size > 1:
        cmap = spec.astype(float)
        if cmap.ndim != 2 or cmap.shape[1] != 3:
            raise MatlabError("rgb2ind: the colormap must have three columns.")
        flat = _to_unit(a).reshape(-1, 3)
        idx = np.empty(flat.shape[0], dtype=np.int64)
        for start in range(0, flat.shape[0], 65536):  # nearest colour, in chunks to bound memory
            block = flat[start:start + 65536]
            d = ((block[:, None, :] - cmap[None, :, :]) ** 2).sum(axis=2)
            idx[start:start + 65536] = d.argmin(axis=1)
        idx = idx.reshape(a.shape[:2])
    else:
        n = int(spec.flat[0])
        if n < 1 or n > 65536:
            raise MatlabError("rgb2ind: the number of colours must be between 1 and 65536.")
        flat = rgb8.reshape(-1, 3)
        colours, inverse = np.unique(flat, axis=0, return_inverse=True)
        if colours.shape[0] <= n:
            cmap = colours.astype(float) / 255.0
            idx = np.asarray(inverse).reshape(a.shape[:2])
        else:
            Image = _pil()
            q = Image.fromarray(np.ascontiguousarray(rgb8)).quantize(
                colors=min(n, 256), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            idx = np.asarray(q).astype(np.int64)
            used = int(idx.max()) + 1
            cmap = np.asarray(q.getpalette()[:used * 3], dtype=float).reshape(-1, 3) / 255.0
    x = _index_out(idx, cmap.shape[0])
    return (x, cmap) if nargout > 1 else x


def _colormap(interp, spec, n: int, name: str) -> np.ndarray:
    if isinstance(spec, np.ndarray) and not is_char(spec):
        cmap = spec.astype(float)
        if cmap.ndim != 2 or cmap.shape[1] != 3:
            raise MatlabError(f"{name}: a colormap must have three columns.")
        return cmap
    cname = text(spec, "colormap name")
    fn = BUILTINS.get(cname)
    if fn is None:
        raise MatlabError(f"{name}: unknown colormap '{cname}'.")
    return np.asarray(fn(interp, [np.array([[float(n)]])], 1), dtype=float)


_COLOR_LETTERS = {"r": (1, 0, 0), "g": (0, 1, 0), "b": (0, 0, 1), "c": (0, 1, 1), "m": (1, 0, 1),
                  "y": (1, 1, 0), "k": (0, 0, 0), "w": (1, 1, 1)}


@builtin("label2rgb")
def _label2rgb(interp, args, nargout):
    nargs(args, 1, 4, "label2rgb")
    lab = np.round(_img(args[0], "label2rgb", "label matrix").astype(float)).astype(np.int64)
    n = max(int(lab.max()) if lab.size else 0, 1)
    cmap = _colormap(interp, args[1] if len(args) > 1 else char("jet"), n, "label2rgb")
    if cmap.shape[0] < n:
        cmap = cmap[np.arange(n) % cmap.shape[0]]
    zero = np.array([1.0, 1.0, 1.0])
    if len(args) > 2:
        z = args[2]
        if is_text(z):
            key = text(z).lower()[:1]
            if key not in _COLOR_LETTERS:
                raise MatlabError("label2rgb: unknown colour for the background.")
            zero = np.array(_COLOR_LETTERS[key], dtype=float)
        else:
            zero = _vec(z, "background colour")
    if len(args) > 3 and text(args[3]).lower() == "shuffle":
        cmap = cmap[np.random.default_rng(0).permutation(cmap.shape[0])]
    table = np.vstack([zero.reshape(1, 3), cmap[:n]])
    out = table[np.clip(lab, 0, n)]
    return _from_unit(out, np.uint8)


# -- thresholds, histograms and contrast --------------------------------------------------------

def _hist_counts(a: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts and bin centres of an intensity histogram over the class range."""
    if a.dtype == bool:
        n = 2
    lo, hi = _class_range(a)
    x = a.astype(float).ravel()
    x = x[~np.isnan(x)]
    pos = np.clip(_round_half_away((x - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(np.int64)
    counts = np.bincount(pos, minlength=n).astype(float)
    centers = lo + np.arange(n) * (hi - lo) / max(n - 1, 1)
    return counts, centers


@builtin("imhist")
def _imhist(interp, args, nargout):
    nargs(args, 1, 2, "imhist")
    a = _img(args[0], "imhist")
    if a.ndim > 2:
        raise MatlabError("imhist: the image must be 2-D; convert an RGB image with rgb2gray first.")
    n = 256
    if len(args) > 1:
        second = _img(args[1], "imhist", "second input")
        if second.size > 1:  # imhist(X, map): how often each palette colour is used
            m = second.shape[0]
            idx = np.clip(_index_base(a), 0, m - 1)
            counts = np.bincount(idx.ravel(), minlength=m).astype(float)
            centers = np.arange(1, m + 1, dtype=float)
            return (_col(counts), _col(centers)) if nargout > 1 else _col(counts)
        n = int(second.flat[0])
        if n < 1:
            raise MatlabError("imhist: the number of bins must be positive.")
    counts, centers = _hist_counts(a, n)
    if nargout == 0:
        interp.call_named("bar", [centers.reshape(1, -1), counts.reshape(1, -1)], 0, interp.frame)
        return None
    return (_col(counts), _col(centers)) if nargout > 1 else _col(counts)


def _otsu(counts: np.ndarray) -> tuple[float, float]:
    """Otsu's threshold as a 0..1 level, and the separability measure."""
    total = counts.sum()
    n = counts.size
    if total == 0 or n < 2:
        return 0.0, 0.0
    p = counts / total
    levels = np.arange(n, dtype=float)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    between[~np.isfinite(between)] = -1.0
    best = between.max()
    if best <= 0:
        return 0.0, 0.0
    where = np.flatnonzero(between >= best * (1 - 1e-12))
    level = where.mean() / (n - 1)
    total_var = float(np.sum(p * (levels - mu_t) ** 2))
    return float(level), float(best / total_var) if total_var > 0 else 0.0


@builtin("graythresh")
def _graythresh(interp, args, nargout):
    nargs(args, 1, 1, "graythresh")
    a = _img(args[0], "graythresh")
    counts, _ = _hist_counts(a if a.dtype != bool else a.astype(np.uint8) * 255, 256)
    level, em = _otsu(counts)
    return (level, em) if nargout > 1 else level


@builtin("otsuthresh")
def _otsuthresh(interp, args, nargout):
    nargs(args, 1, 1, "otsuthresh")
    level, em = _otsu(_vec(args[0], "histogram counts"))
    return (level, em) if nargout > 1 else level


@builtin("im2bw")
def _im2bw(interp, args, nargout):
    nargs(args, 1, 3, "im2bw")
    a = _img(args[0], "im2bw")
    rest = list(args[1:])
    if len(rest) == 2 or (rest and isinstance(rest[0], np.ndarray) and rest[0].size > 1):
        a = _ind2rgb(interp, [a, rest.pop(0)], 1)  # indexed image with its colormap
    level = _scalar(rest[0], "level") if rest else 0.5
    if not 0 <= level <= 1:
        raise MatlabError("im2bw: the level must be between 0 and 1.")
    g = _gray2d(a, "im2bw")
    if g.dtype == bool:
        return g
    return _to_unit(g) > level


def _local_mean(g: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _nd().uniform_filter(g, size=size, mode="nearest")


@builtin("imbinarize")
def _imbinarize(interp, args, nargout):
    nargs(args, 1, None, "imbinarize")
    a = _img(args[0], "imbinarize")
    if a.dtype == bool:
        return a
    rest, opts = _options(list(args[1:]), ("Sensitivity", "ForegroundPolarity"), "imbinarize")
    unit = _to_unit(a)
    method = "global"
    if rest and is_text(rest[0]):
        method = text(rest[0]).lower()
        if method not in ("global", "adaptive"):
            raise MatlabError("imbinarize: the method must be 'global' or 'adaptive'.")
    elif rest:
        t = _img(rest[0], "imbinarize", "threshold").astype(float)
        if t.size != 1 and t.shape != a.shape:
            raise MatlabError("imbinarize: the threshold must be a scalar or have the size of the image.")
        return unit > t
    if method == "adaptive":
        # Bradley's method: compare each pixel with a fraction of the mean around it. A pixel must
        # be clearly darker than its surroundings to become false, so flat areas come out true.
        sens = _scalar(opts["sensitivity"], "Sensitivity") if "sensitivity" in opts else 0.5
        if not 0 <= sens <= 1:
            raise MatlabError("imbinarize: 'Sensitivity' must be between 0 and 1.")
        dark = "foregroundpolarity" in opts and text(opts["foregroundpolarity"]).lower() == "dark"
        size = tuple(2 * (s // 16) + 1 for s in unit.shape[:2])
        mean = _per_plane(unit, lambda p: _local_mean(p, size))
        # more sensitivity means more foreground: a lower threshold for bright foregrounds,
        # a higher one for dark foregrounds
        factor = (0.7 + 0.3 * sens) if dark else (1.0 - 0.3 * sens)
        return unit > mean * factor
    counts, _ = _hist_counts(a, 256)
    return unit > _otsu(counts)[0]


@builtin("imcomplement")
def _imcomplement(interp, args, nargout):
    nargs(args, 1, 1, "imcomplement")
    a = _img(args[0], "imcomplement")
    if a.dtype == bool:
        return ~a
    if a.dtype.kind == "u":
        return (np.iinfo(a.dtype).max - a).astype(a.dtype)
    if a.dtype.kind == "i":
        return (-1 - a.astype(np.int64)).astype(a.dtype)
    return 1 - a


def _stretchlim(a: np.ndarray, tol: tuple[float, float]) -> np.ndarray:
    """2-by-C limits (0..1) that leave the given fractions of pixels below and above."""
    planes = a.reshape(a.shape[0], a.shape[1], -1) if a.ndim > 2 else a[..., None]
    out = np.zeros((2, planes.shape[2]))
    for k in range(planes.shape[2]):
        counts, _ = _hist_counts(planes[..., k], 256)
        total = counts.sum()
        if total == 0:
            out[:, k] = (0.0, 1.0)
            continue
        cdf = np.cumsum(counts) / total
        lo = int(np.argmax(cdf > tol[0])) if np.any(cdf > tol[0]) else 0
        hi = int(np.argmax(cdf >= tol[1])) if np.any(cdf >= tol[1]) else 255
        if lo >= hi:
            lo, hi = 0, 255
        out[:, k] = (lo / 255.0, hi / 255.0)
    return out


@builtin("stretchlim")
def _stretchlim_b(interp, args, nargout):
    nargs(args, 1, 2, "stretchlim")
    a = _img(args[0], "stretchlim")
    tol = (0.01, 0.99)
    if len(args) > 1 and args[1].size:
        t = _vec(args[1], "tolerance")
        tol = (float(t[0]), 1.0 - float(t[0])) if t.size == 1 else (float(t[0]), float(t[1]))
    return _stretchlim(a, tol)


def _limits(v, channels: int, default: np.ndarray, name: str) -> np.ndarray:
    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
        return default
    lim = _img(v, name, "limits").astype(float)
    if lim.size == 2:
        lim = np.repeat(lim.reshape(2, 1, order="F"), channels, axis=1)
    elif lim.shape != (2, channels):
        raise MatlabError(f"{name}: limits must be [low high] or a 2-by-3 matrix for an RGB image.")
    if np.any(lim < 0) or np.any(lim > 1):
        raise MatlabError(f"{name}: limits must lie between 0 and 1.")
    return lim


@builtin("imadjust")
def _imadjust(interp, args, nargout):
    nargs(args, 1, 4, "imadjust")
    a = _img(args[0], "imadjust")
    channels = a.shape[2] if a.ndim == 3 else 1
    unit = _to_unit(a)
    full = np.repeat(np.array([[0.0], [1.0]]), channels, axis=1)
    if len(args) == 1:
        if a.ndim == 3:
            raise MatlabError("imadjust: give the limits for an RGB image, e.g. imadjust(RGB, stretchlim(RGB)).")
        in_lim = _stretchlim(a, (0.01, 0.99))
    else:
        in_lim = _limits(args[1], channels, full, "imadjust")
    out_lim = _limits(args[2] if len(args) > 2 else None, channels, full, "imadjust")
    gamma = np.ones(channels)
    if len(args) > 3 and args[3].size:
        g = _vec(args[3], "gamma")
        gamma = np.repeat(g, channels) if g.size == 1 else g
        if gamma.size != channels or np.any(gamma < 0):
            raise MatlabError("imadjust: gamma must be a non-negative scalar, or one value per channel.")
    if np.any(in_lim[0] >= in_lim[1]):
        raise MatlabError("imadjust: each low input limit must be below its high limit.")
    planes = unit.reshape(unit.shape[0], unit.shape[1], channels) if unit.ndim >= 2 else unit
    out = np.empty_like(planes)
    for k in range(channels):
        lo, hi = in_lim[:, k]
        t = np.clip((planes[..., k] - lo) / (hi - lo), 0.0, 1.0) ** gamma[k]
        out[..., k] = out_lim[0, k] + t * (out_lim[1, k] - out_lim[0, k])
    return _from_unit(out.reshape(a.shape), a.dtype)


@builtin("histeq")
def _histeq(interp, args, nargout):
    nargs(args, 1, 2, "histeq")
    a = _img(args[0], "histeq")
    if a.ndim > 2:
        raise MatlabError("histeq: the image must be 2-D; equalise each channel of an RGB image separately.")
    target = None
    n = 64
    if len(args) > 1:
        second = _vec(args[1], "second input")
        if second.size > 1:
            target = second
            n = second.size
        else:
            n = int(second[0])
            if n < 1:
                raise MatlabError("histeq: the number of levels must be positive.")
    bins = 256
    counts, _ = _hist_counts(a, bins)
    total = counts.sum()
    if total == 0:
        return a
    have = np.cumsum(counts) / total
    want = np.cumsum(target / target.sum()) if target is not None else np.arange(1, n + 1) / n
    # for every input level pick the output level whose cumulative target share is closest
    pos = np.searchsorted(want, have, side="left")
    pos = np.clip(pos, 0, n - 1)
    left = np.clip(pos - 1, 0, n - 1)
    use_left = (pos > 0) & (np.abs(want[left] - have) < np.abs(want[pos] - have) - 1e-12)
    pos = np.where(use_left, left, pos)
    transform = pos / max(n - 1, 1)
    lo, hi = _class_range(a)
    idx = np.clip(_round_half_away((a.astype(float) - lo) / (hi - lo) * (bins - 1)), 0, bins - 1).astype(np.int64)
    out = _from_unit(transform[idx], a.dtype)
    return (out, transform.reshape(1, -1)) if nargout > 1 else out


@builtin("entropy")
def _entropy(interp, args, nargout):
    nargs(args, 1, 1, "entropy")
    a = _img(args[0], "entropy")
    counts, _ = _hist_counts(a, 256)
    p = counts[counts > 0] / counts.sum() if counts.sum() else np.zeros(0)
    return float(-(p * np.log2(p)).sum())


# -- arithmetic and comparison ------------------------------------------------------------------

def _pair(args, name: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = _img(args[0], name), _img(args[1], name)
    if a.shape != b.shape and a.size != 1 and b.size != 1:
        raise MatlabError(f"{name}: the two images must have the same size.")
    return a, b


def _arith(name: str, op):
    def fn(interp, args, nargout):
        nargs(args, 2, 3, name)
        a, b = _pair(args, name)
        like = a if a.size >= b.size or a.dtype.kind != "f" else b
        out_dtype = like.dtype
        if len(args) > 2:
            from ._util import cast_class
            return cast_class(op(a.astype(float), b.astype(float)), text(args[2]))
        if out_dtype.kind == "b":
            out_dtype = np.dtype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _cast(op(a.astype(float), b.astype(float)), out_dtype)
    return fn


builtin("imadd")(_arith("imadd", lambda x, y: x + y))
builtin("imsubtract")(_arith("imsubtract", lambda x, y: x - y))
builtin("immultiply")(_arith("immultiply", lambda x, y: x * y))
builtin("imdivide")(_arith("imdivide", lambda x, y: x / y))
builtin("imabsdiff")(_arith("imabsdiff", lambda x, y: np.abs(x - y)))


@builtin("immse")
def _immse(interp, args, nargout):
    nargs(args, 2, 2, "immse")
    a, b = _pair(args, "immse")
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2)) if a.size else float("nan")


@builtin("psnr")
def _psnr(interp, args, nargout):
    nargs(args, 2, 3, "psnr")
    a, b = _pair(args, "psnr")
    lo, hi = _class_range(a)
    peak = _scalar(args[2], "peak value") if len(args) > 2 else hi - lo
    x, y = a.astype(float), b.astype(float)
    mse = np.mean((x - y) ** 2)
    with np.errstate(divide="ignore"):
        value = float(10 * np.log10(peak ** 2 / mse))
        snr = float(10 * np.log10(np.mean(y ** 2) / mse))
    return (value, snr) if nargout > 1 else value


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


@builtin("ssim")
def _ssim(interp, args, nargout):
    nargs(args, 2, None, "ssim")
    rest, opts = _options(list(args), ("DynamicRange", "Radius", "RegularizationConstants", "Exponents"), "ssim")
    a, b = _pair(rest, "ssim")
    lo, hi = _class_range(a)
    rng = _scalar(opts["dynamicrange"], "DynamicRange") if "dynamicrange" in opts else hi - lo
    sigma = _scalar(opts["radius"], "Radius") if "radius" in opts else 1.5
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    if "regularizationconstants" in opts:
        c = _vec(opts["regularizationconstants"], "RegularizationConstants")
        c1, c2 = float(c[0]), float(c[1])
    x, y = a.astype(float), b.astype(float)
    g = _gauss1d(sigma, int(np.ceil(3 * sigma)))
    nd = _nd()

    def blur(z):
        for axis in range(min(z.ndim, 2)):
            z = nd.correlate1d(z, g, axis=axis, mode="nearest")
        return z

    mx, my = blur(x), blur(y)
    vx = np.maximum(blur(x * x) - mx * mx, 0)
    vy = np.maximum(blur(y * y) - my * my, 0)
    cov = blur(x * y) - mx * my
    smap = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    value = float(smap.mean())
    return (value, smap) if nargout > 1 else value


@builtin("mean2")
def _mean2(interp, args, nargout):
    nargs(args, 1, 1, "mean2")
    a = _img(args[0], "mean2")
    return float(a.astype(float).mean()) if a.size else float("nan")


@builtin("std2")
def _std2(interp, args, nargout):
    nargs(args, 1, 1, "std2")
    a = _img(args[0], "std2")
    return float(a.astype(float).std(ddof=1)) if a.size > 1 else 0.0


@builtin("corr2")
def _corr2(interp, args, nargout):
    nargs(args, 2, 2, "corr2")
    a, b = _pair(args, "corr2")
    x = a.astype(float) - a.astype(float).mean()
    y = b.astype(float) - b.astype(float).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


@builtin("integralImage")
def _integral_image(interp, args, nargout):
    nargs(args, 1, 2, "integralImage")
    a = _img(args[0], "integralImage").astype(float)
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:])
    out[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return out


# -- padding and filtering ----------------------------------------------------------------------

_PAD_MODES = {"replicate": "edge", "symmetric": "symmetric", "circular": "wrap"}


def _pad(a: np.ndarray, widths: list[tuple[int, int]], how, name: str) -> np.ndarray:
    widths = widths + [(0, 0)] * (a.ndim - len(widths))
    if isinstance(how, str):
        if how not in _PAD_MODES:
            raise MatlabError(f"{name}: unknown padding '{how}'. Use a number, 'replicate', 'symmetric' "
                              "or 'circular'.")
        if a.size == 0:
            raise MatlabError(f"{name}: cannot pad an empty array by repeating its values.")
        return np.pad(a, widths, mode=_PAD_MODES[how])
    if a.dtype == bool:
        value = bool(how)
    elif a.dtype.kind in "iu":
        value = _cast(np.array(float(how)), a.dtype).item()
    else:
        value = how
    return np.pad(a, widths, mode="constant", constant_values=value)


@builtin("padarray")
def _padarray(interp, args, nargout):
    nargs(args, 2, 4, "padarray")
    a = args[0]
    if not isinstance(a, np.ndarray):
        raise MatlabError("padarray: the first input must be a numeric, logical or char array.")
    size = _vec(args[1], "pad size")
    if np.any(size < 0) or np.any(size != np.round(size)):
        raise MatlabError("padarray: pad sizes must be non-negative integers.")
    how: object = 0.0
    direction = "both"
    for extra in args[2:]:
        if is_text(extra):
            key = text(extra).lower()
            if key in ("pre", "post", "both"):
                direction = key
            else:
                how = key
        else:
            how = _scalar(extra, "pad value")
    if a.ndim < size.size:
        a = a.reshape(a.shape + (1,) * (size.size - a.ndim))
    widths = []
    for s in size.astype(int):
        widths.append((s if direction in ("pre", "both") else 0, s if direction in ("post", "both") else 0))
    if is_char(a) and not isinstance(how, str):
        out = np.pad(a, widths + [(0, 0)] * (a.ndim - len(widths)), mode="constant",
                     constant_values=chr(int(how)) if how else "\0")
        return as_2d(out)
    return as_2d(_pad(a, widths, how, "padarray"))


def _correlate(a: np.ndarray, h: np.ndarray, how, shape: str, conv: bool, name: str) -> np.ndarray:
    """N-D correlation (or convolution) of a float array with explicit boundary handling."""
    if h.ndim > a.ndim:
        if any(s != 1 for s in h.shape[a.ndim:]):
            raise MatlabError(f"{name}: the filter has more dimensions than the image.")
        h = h.reshape(h.shape[:a.ndim])
    h = h.reshape(h.shape + (1,) * (a.ndim - h.ndim))
    if conv:
        h = h[tuple(slice(None, None, -1) for _ in range(h.ndim))]
    padded = _pad(a, [(k - 1, k - 1) for k in h.shape], how, name)
    full = _nd().correlate(padded, h, mode="constant", cval=0.0)
    # entry i of the full result sits at i + k//2 of the padded correlation
    slices = []
    for n, k in zip(a.shape, h.shape):
        if shape == "full":
            slices.append(slice(k // 2, k // 2 + n + k - 1))
        else:
            centre = (k - 1) // 2  # of the kernel as applied (already flipped for convolution)
            start = k // 2 + (k - 1 - centre)
            slices.append(slice(start, start + n))
    return full[tuple(slices)]


def _filter_image(a: np.ndarray, h: np.ndarray, how, shape: str, conv: bool, name: str) -> np.ndarray:
    work = a.astype(float)
    out = _correlate(work, h.astype(float), how, shape, conv, name)
    if a.dtype == bool:
        return out
    return _cast(out, a.dtype) if a.dtype.kind in "iu" else out.astype(a.dtype)


@builtin("imfilter")
def _imfilter(interp, args, nargout):
    nargs(args, 2, 5, "imfilter")
    a = _img(args[0], "imfilter")
    h = _img(args[1], "imfilter", "filter")
    how: object = 0.0
    shape, conv = "same", False
    for extra in args[2:]:
        if is_text(extra):
            key = text(extra).lower()
            if key in ("same", "full"):
                shape = key
            elif key in ("corr", "conv"):
                conv = key == "conv"
            elif key in _PAD_MODES:
                how = key
            else:
                raise MatlabError(f"imfilter: unknown option '{text(extra)}'.")
        else:
            how = _scalar(extra, "padding value")
    if a.size == 0:
        return a
    out = _filter_image(a, h, how, shape, conv, "imfilter")
    return out > 0.5 if a.dtype == bool else out


@builtin("filter2")
def _filter2(interp, args, nargout):
    nargs(args, 2, 3, "filter2")
    h = _img(args[0], "filter2", "filter").astype(float)
    a = _img(args[1], "filter2", "data").astype(float)
    shape = text(args[2]).lower() if len(args) > 2 else "same"
    if shape not in ("same", "full", "valid"):
        raise MatlabError("filter2: the shape must be 'same', 'full' or 'valid'.")
    if shape == "valid":
        full = _correlate(a, h, 0.0, "same", False, "filter2")
        slices = []
        for n, k in zip(a.shape, h.shape + (1,) * (a.ndim - h.ndim)):
            c = (k - 1) // 2
            slices.append(slice(c, c + max(n - k + 1, 0)))
        return full[tuple(slices)]
    return _correlate(a, h, 0.0, shape, False, "filter2")


def _hsize(v, default: int) -> tuple[int, int]:
    if v is None:
        return default, default
    s = _vec(v, "filter size")
    if s.size == 1:
        return int(s[0]), int(s[0])
    return int(s[0]), int(s[1])


def _grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(rows, dtype=float) - (rows - 1) / 2
    x = np.arange(cols, dtype=float) - (cols - 1) / 2
    return np.meshgrid(x, y)


def _gaussian_kernel(rows: int, cols: int, sigma: float) -> np.ndarray:
    x, y = _grid(rows, cols)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0
    return h / h.sum()


def _laplacian_kernel(alpha: float) -> np.ndarray:
    a, b = alpha / (alpha + 1), (1 - alpha) / (alpha + 1)
    return np.array([[a, b, a], [b, -4 / (alpha + 1), b], [a, b, a]])


def _disk_kernel(radius: float) -> np.ndarray:
    """Circular averaging filter: each weight is the area of the pixel that lies inside the circle."""
    half = int(np.ceil(radius - 0.5))
    size = 2 * half + 1
    # integrate the chord length across each pixel column strip with a fine midpoint rule
    sub = 400
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    h = np.zeros((size, size))
    centres = np.arange(-half, half + 1, dtype=float)
    for j, cx in enumerate(centres):
        xs = cx + offs
        chord = np.sqrt(np.maximum(radius * radius - xs * xs, 0.0))
        for i, cy in enumerate(centres):
            top = np.clip(chord, cy - 0.5, cy + 0.5)
            bottom = np.clip(-chord, cy - 0.5, cy + 0.5)
            h[i, j] = np.mean(top - bottom)
    h = (h + h.T) / 2
    return h / h.sum()


def _motion_kernel(length: float, theta: float) -> np.ndarray:
    """Line of the given length and angle (degrees, counter-clockwise), with soft one-pixel edges."""
    half = (length - 1) / 2
    t = np.deg2rad(theta)
    c, s = np.cos(t), np.sin(t)
    cols = int(np.ceil(abs(half * c) - 1e-9))
    rows = int(np.ceil(abs(half * s) - 1e-9))
    x, y = np.meshgrid(np.arange(-cols, cols + 1, dtype=float), np.arange(-rows, rows + 1, dtype=float))
    y = -y  # image rows grow downwards
    along = x * c + y * s
    across = np.abs(-x * s + y * c)
    beyond = np.maximum(np.abs(along) - half, 0.0)
    h = np.maximum(1.0 - np.hypot(across, beyond), 0.0)
    total = h.sum()
    return h / total if total else h


@builtin("fspecial")
def _fspecial(interp, args, nargout):
    nargs(args, 1, 3, "fspecial")
    kind = text(args[0], "filter type").lower()
    p = list(args[1:])

    def arg(k, default):
        return _scalar(p[k], "filter parameter") if len(p) > k and p[k].size else default

    if kind == "average":
        r, c = _hsize(p[0] if p else None, 3)
        return np.full((r, c), 1.0 / (r * c))
    if kind == "gaussian":
        r, c = _hsize(p[0] if p else None, 3)
        sigma = arg(1, 0.5)
        if sigma <= 0:
            raise MatlabError("fspecial: sigma must be positive.")
        return _gaussian_kernel(r, c, sigma)
    if kind == "sobel":
        return np.array([[1.0, 2, 1], [0, 0, 0], [-1, -2, -1]])
    if kind == "prewitt":
        return np.array([[1.0, 1, 1], [0, 0, 0], [-1, -1, -1]])
    if kind == "laplacian":
        alpha = arg(0, 0.2)
        if not 0 <= alpha <= 1:
            raise MatlabError("fspecial: alpha must be between 0 and 1.")
        return _laplacian_kernel(alpha)
    if kind == "unsharp":
        alpha = arg(0, 0.2)
        h = -_laplacian_kernel(alpha)
        h[1, 1] += 1
        return h
    if kind == "log":
        r, c = _hsize(p[0] if p else None, 5)
        sigma = arg(1, 0.5)
        x, y = _grid(r, c)
        g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        g[g < np.finfo(float).eps * g.max()] = 0
        g = g / g.sum() if g.sum() else g
        h = g * (x * x + y * y - 2 * sigma * sigma) / sigma ** 4
        return h - h.sum() / h.size
    if kind == "disk":
        radius = arg(0, 5.0)
        if radius <= 0:
            raise MatlabError("fspecial: the radius must be positive.")
        return _disk_kernel(radius)
    if kind == "motion":
        return _motion_kernel(arg(0, 9.0), arg(1, 0.0))
    raise MatlabError(f"fspecial: unknown filter type '{kind}'. Use 'average', 'disk', 'gaussian', "
                      "'laplacian', 'log', 'motion', 'prewitt', 'sobel' or 'unsharp'.")


@builtin("imgaussfilt")
def _imgaussfilt(interp, args, nargout):
    nargs(args, 1, None, "imgaussfilt")
    rest, opts = _options(list(args), ("FilterSize", "Padding", "FilterDomain"), "imgaussfilt")
    a = _img(rest[0], "imgaussfilt")
    sig = _vec(rest[1], "sigma") if len(rest) > 1 else np.array([0.5])
    if np.any(sig <= 0):
        raise MatlabError("imgaussfilt: sigma must be positive.")
    sy, sx = (sig[0], sig[0]) if sig.size == 1 else (sig[0], sig[1])
    if "filtersize" in opts:
        fr, fc = _hsize(opts["filtersize"], 3)
        if fr % 2 == 0 or fc % 2 == 0:
            raise MatlabError("imgaussfilt: the filter size must be odd.")
    else:
        fr, fc = 2 * int(np.ceil(2 * sy)) + 1, 2 * int(np.ceil(2 * sx)) + 1
    how: object = "replicate"
    if "padding" in opts:
        how = text(opts["padding"]).lower() if is_text(opts["padding"]) else _scalar(opts["padding"], "Padding")
    out = a.astype(float)
    gy = _gauss1d(sy, fr // 2).reshape(-1, 1)
    gx = _gauss1d(sx, fc // 2).reshape(1, -1)
    out = _correlate(out, gy, how, "same", False, "imgaussfilt")
    out = _correlate(out, gx, how, "same", False, "imgaussfilt")
    return _cast(out, a.dtype) if a.dtype.kind in "iu" else (out if a.dtype == bool else out.astype(a.dtype))


def _window(v, default: int, name: str) -> tuple[int, int]:
    if v is None or v.size == 0:
        return default, default
    s = _vec(v, "neighbourhood size")
    r, c = (int(s[0]), int(s[0])) if s.size == 1 else (int(s[0]), int(s[1]))
    if r < 1 or c < 1:
        raise MatlabError(f"{name}: the neighbourhood size must be positive.")
    return r, c


def _origin(size: tuple[int, ...]) -> tuple[int, ...]:
    """scipy filter origin that puts the window centre at floor((n+1)/2), counting from 1."""
    return tuple(-1 if n % 2 == 0 else 0 for n in size)


def _per_plane(a: np.ndarray, fn) -> np.ndarray:
    if a.ndim == 2:
        return fn(a)
    flat = a.reshape(a.shape[0], a.shape[1], -1)
    return np.stack([fn(flat[..., k]) for k in range(flat.shape[2])], axis=-1).reshape(a.shape)


@builtin("medfilt2")
def _medfilt2(interp, args, nargout):
    nargs(args, 1, 3, "medfilt2")
    a = _img(args[0], "medfilt2")
    rest = list(args[1:])
    padding = "zeros"
    if rest and is_text(rest[-1]):
        padding = text(rest.pop()).lower()
        if padding not in ("zeros", "symmetric", "indexed"):
            raise MatlabError("medfilt2: padding must be 'zeros', 'symmetric' or 'indexed'.")
    r, c = _window(rest[0] if rest else None, 3, "medfilt2")
    nd = _nd()
    foot = np.ones((r, c), dtype=bool)

    def run(p):
        work = p.astype(np.uint8) if p.dtype == bool else p
        if padding == "symmetric":
            out = nd.median_filter(work, footprint=foot, mode="reflect", origin=_origin((r, c)))
        else:
            out = nd.median_filter(work, footprint=foot, mode="constant", cval=0, origin=_origin((r, c)))
        return out.astype(bool) if p.dtype == bool else out

    return _per_plane(a, run)


@builtin("ordfilt2")
def _ordfilt2(interp, args, nargout):
    nargs(args, 3, 4, "ordfilt2")
    a = _img(args[0], "ordfilt2")
    order = to_int(args[1], "order")
    dom = _img(args[2], "ordfilt2", "domain") != 0
    padding = text(args[3]).lower() if len(args) > 3 else "zeros"
    count = int(dom.sum())
    if not 1 <= order <= count:
        raise MatlabError("ordfilt2: the order must be between 1 and the number of nonzero domain elements.")
    nd = _nd()
    mode = "reflect" if padding == "symmetric" else "constant"

    def run(p):
        work = p.astype(np.uint8) if p.dtype == bool else p
        out = nd.rank_filter(work, rank=order - 1, footprint=dom, mode=mode, cval=0, origin=_origin(dom.shape))
        return out.astype(bool) if p.dtype == bool else out

    return _per_plane(a, run)


@builtin("wiener2")
def _wiener2(interp, args, nargout):
    nargs(args, 1, 3, "wiener2")
    a = _img(args[0], "wiener2")
    if a.ndim != 2:
        raise MatlabError("wiener2: the image must be 2-D.")
    r, c = _window(args[1] if len(args) > 1 else None, 3, "wiener2")
    g = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    box = np.ones((r, c))
    mean = _correlate(g, box, 0.0, "same", False, "wiener2") / (r * c)
    var = _correlate(g * g, box, 0.0, "same", False, "wiener2") / (r * c) - mean * mean
    noise = _scalar(args[2], "noise power") if len(args) > 2 and args[2].size else float(var.mean())
    gain = np.maximum(var - noise, 0.0) / np.maximum(var, noise) if noise > 0 else np.ones_like(var)
    out = mean + gain * (g - mean)
    res = _from_unit(out, a.dtype) if a.dtype.kind != "f" else out.astype(a.dtype)
    return (res, noise) if nargout > 1 else res


@builtin("imnoise")
def _imnoise(interp, args, nargout):
    nargs(args, 1, 4, "imnoise")
    from .core import _rng
    a = _img(args[0], "imnoise")
    kind = text(args[1]).lower() if len(args) > 1 else "gaussian"
    p = [_scalar(x, "noise parameter") for x in args[2:]]
    g = _to_unit(a)
    rng = _rng(interp)
    if kind == "gaussian":
        mean = p[0] if p else 0.0
        var = p[1] if len(p) > 1 else 0.01
        out = g + mean + np.sqrt(var) * rng.standard_normal(g.shape)
    elif kind in ("salt & pepper", "salt and pepper", "salt&pepper"):
        d = p[0] if p else 0.05
        u = rng.random(g.shape)
        out = g.copy()
        out[u < d / 2] = 0.0
        out[(u >= d / 2) & (u < d)] = 1.0
    elif kind == "speckle":
        var = p[0] if p else 0.05
        out = g + np.sqrt(var) * g * rng.standard_normal(g.shape)
    elif kind == "poisson":
        scale = 1e12 if a.dtype == np.float64 else (1e6 if a.dtype == np.float32 else _class_range(a)[1])
        out = rng.poisson(np.maximum(g, 0) * scale) / scale
    else:
        raise MatlabError("imnoise: the noise type must be 'gaussian', 'salt & pepper', 'speckle' or 'poisson'.")
    return _from_unit(np.clip(out, 0.0, 1.0), a.dtype)


@builtin("im2col")
def _im2col(interp, args, nargout):
    nargs(args, 2, 3, "im2col")
    a = _img(args[0], "im2col")
    if a.ndim != 2:
        raise MatlabError("im2col: the input must be a matrix.")
    r, c = _window(args[1], 1, "im2col")
    kind = text(args[2]).lower() if len(args) > 2 else "sliding"
    if kind == "sliding":
        if a.shape[0] < r or a.shape[1] < c:
            return np.zeros((r * c, 0), dtype=a.dtype)
        win = np.lib.stride_tricks.sliding_window_view(a, (r, c))  # (rows, cols, r, c)
        return win.transpose(2, 3, 0, 1).reshape(r * c, -1, order="F")
    if kind == "distinct":
        rows = -(-a.shape[0] // r) * r
        cols = -(-a.shape[1] // c) * c
        padded = np.zeros((rows, cols), dtype=a.dtype)
        padded[:a.shape[0], :a.shape[1]] = a
        blocks = padded.reshape(rows // r, r, cols // c, c)  # (block row, r, block col, c)
        return blocks.transpose(1, 3, 0, 2).reshape(r * c, -1, order="F")
    raise MatlabError("im2col: the block type must be 'sliding' or 'distinct'.")


# -- edges ----------------------------------------------------------------------------------------

def _thin(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray, keep_h: bool = True, keep_v: bool = True) -> np.ndarray:
    """Keep pixels that are local maxima of the gradient along its dominant axis."""
    padded = np.pad(mag, 1, mode="constant")
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    horiz = np.abs(gx) >= np.abs(gy)  # gradient mostly along x: a vertical edge
    out = np.zeros(mag.shape, dtype=bool)
    if keep_v:
        out |= horiz & (mag > left) & (mag >= right)
    if keep_h:
        out |= ~horiz & (mag > up) & (mag >= down)
    return out


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    lab, n = _nd().label(weak, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(weak.shape, dtype=bool)
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def _canny(g: np.ndarray, thresh, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    nd = _nd()
    radius = max(int(np.ceil(4 * sigma)), 1)
    k = _gauss1d(sigma, radius)
    x = np.arange(-radius, radius + 1, dtype=float)
    dk = -x * k / (sigma * sigma)
    dk /= np.abs(dk * x).sum()
    smooth_r = nd.correlate1d(g, k, axis=0, mode="nearest")
    smooth_c = nd.correlate1d(g, k, axis=1, mode="nearest")
    gx = nd.correlate1d(smooth_r, dk[::-1], axis=1, mode="nearest")
    gy = nd.correlate1d(smooth_c, dk[::-1], axis=0, mode="nearest")
    mag = np.hypot(gx, gy)
    top = mag.max()
    if top > 0:
        mag = mag / top
    if thresh is None:
        counts, edges = np.histogram(mag, bins=64, range=(0.0, 1.0))
        cdf = np.cumsum(counts) / max(mag.size, 1)
        high = (int(np.argmax(cdf > 0.7)) + 1) / 64
        low = 0.4 * high
    elif np.size(thresh) == 1:
        high = float(np.ravel(thresh)[0])
        low = 0.4 * high
    else:
        low, high = float(np.ravel(thresh)[0]), float(np.ravel(thresh)[1])
    if not 0 <= low < high < 1:
        raise MatlabError("edge: canny thresholds must satisfy 0 <= low < high < 1.")
    # non-maximum suppression along the gradient, interpolating between neighbours
    rows, cols = np.indices(mag.shape)
    norm = np.where(mag > 0, np.hypot(gx, gy), 1.0)
    ux, uy = gx / norm, gy / norm
    ahead = nd.map_coordinates(mag, [rows + uy, cols + ux], order=1, mode="constant")
    behind = nd.map_coordinates(mag, [rows - uy, cols - ux], order=1, mode="constant")
    ridge = (mag > ahead) & (mag >= behind)
    weak = ridge & (mag > low)
    strong = ridge & (mag > high)
    return _hysteresis(strong, weak), np.array([[low, high]])


@builtin("edge")
def _edge(interp, args, nargout):
    nargs(args, 1, 5, "edge")
    a = _img(args[0], "edge")
    if a.ndim != 2:
        raise MatlabError("edge: the image must be 2-D; convert an RGB image with rgb2gray first.")
    g = _to_unit(a) if a.dtype.kind != "f" else a.astype(float)
    rest = list(args[1:])
    method = text(rest.pop(0)).lower() if rest and is_text(rest[0]) else "sobel"
    thresh = None
    if rest and not is_text(rest[0]):
        t = rest.pop(0)
        thresh = _vec(t, "threshold") if t.size else None
    if method == "canny":
        sigma = _scalar(rest[0], "sigma") if rest and not is_text(rest[0]) else np.sqrt(2.0)
        bw, used = _canny(g, thresh, sigma)
        return (bw, used) if nargout > 1 else bw
    if method in ("log", "zerocross"):
        sigma = _scalar(rest[0], "sigma") if rest and not is_text(rest[0]) else 2.0
        size = int(np.ceil(sigma * 3)) * 2 + 1
        h = _fspecial(interp, [char("log"), np.array([[float(size)]]), np.array([[sigma]])], 1)
        resp = _correlate(g, h, "replicate", "same", False, "edge")
        cut = float(thresh[0]) if thresh is not None else 0.75 * float(np.mean(np.abs(resp)))
        bw = np.zeros(g.shape, dtype=bool)
        # a sign change between neighbours that is large enough marks an edge
        for axis in (0, 1):
            first = [slice(None)] * 2
            second = [slice(None)] * 2
            first[axis], second[axis] = slice(0, -1), slice(1, None)
            p, q = resp[tuple(first)], resp[tuple(second)]
            cross = (p * q < 0) & (np.abs(p - q) > cut)
            closer = np.abs(p) <= np.abs(q)
            bw[tuple(first)] |= cross & closer
            bw[tuple(second)] |= cross & ~closer
        return (bw, cut) if nargout > 1 else bw
    direction = "both"
    thinning = True
    for extra in rest:
        if is_text(extra):
            key = text(extra).lower()
            if key in ("both", "horizontal", "vertical"):
                direction = key
            elif key in ("thinning", "nothinning"):
                thinning = key == "thinning"
            else:
                raise MatlabError(f"edge: unknown option '{text(extra)}'.")
    if method == "sobel":
        ky = np.array([[1.0, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
        kx, factor = ky.T, 4.0
    elif method == "prewitt":
        ky = np.array([[1.0, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 8
        kx, factor = ky.T, 4.0
    elif method == "roberts":
        kx = np.array([[1.0, 0], [0, -1]]) / 2
        ky = np.array([[0, 1.0], [-1, 0]]) / 2
        factor = 6.0
    else:
        raise MatlabError(f"edge: unknown method '{method}'. Use 'sobel', 'prewitt', 'roberts', 'log' "
                          "or 'canny'.")
    gx = _correlate(g, kx, "replicate", "same", False, "edge")
    gy = _correlate(g, ky, "replicate", "same", False, "edge")
    if direction == "horizontal":
        gx = np.zeros_like(gx)
    elif direction == "vertical":
        gy = np.zeros_like(gy)
    power = gx * gx + gy * gy
    cut = float(thresh[0]) if thresh is not None else float(np.sqrt(factor * power.mean()))
    mag = np.sqrt(power)
    bw = mag > cut
    if thinning and method != "roberts":
        bw &= _thin(mag, gx, gy)
    elif thinning:
        bw &= _thin(mag, gx - gy, gx + gy)
    return (bw, cut) if nargout > 1 else bw


# -- morphology -----------------------------------------------------------------------------------

def _line_strel(length: float, degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    half = (length - 1) / 2
    dx, dy = half * np.cos(t), half * np.sin(t)
    cols, rows = int(_round_half_away(np.array(abs(dx)))), int(_round_half_away(np.array(abs(dy))))
    out = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=bool)
    steps = max(rows, cols)
    if steps == 0:
        out[rows, cols] = True
        return out
    for k in range(-steps, steps + 1):
        frac = k / steps
        x = int(_round_half_away(np.array(frac * cols * np.sign(dx if dx else 1))))
        y = int(_round_half_away(np.array(frac * rows * np.sign(dy if dy else 1))))
        out[rows - y, cols + x] = True
    return out


@builtin("strel")
def _strel(interp, args, nargout):
    nargs(args, 1, 4, "strel")
    if not is_text(args[0]):
        return _img(args[0], "strel", "neighbourhood") != 0
    kind = text(args[0]).lower()
    p = list(args[1:])
    if kind == "arbitrary":
        nargs(p, 1, 2, "strel")
        return _img(p[0], "strel", "neighbourhood") != 0
    if not p:
        raise MatlabError(f"strel: the shape '{kind}' needs a size.")
    if kind in ("disk", "diamond", "octagon", "sphere"):
        r = _scalar(p[0], "radius")
        if r < 0 or r != int(r):
            raise MatlabError("strel: the radius must be a non-negative integer.")
        r = int(r)
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        if kind == "disk":
            return x * x + y * y <= r * r
        if kind == "diamond":
            return np.abs(x) + np.abs(y) <= r
        if kind == "sphere":
            z = np.arange(-r, r + 1).reshape(1, 1, -1)
            return x[..., None] ** 2 + y[..., None] ** 2 + z ** 2 <= r * r
        if r % 3:
            raise MatlabError("strel: the radius of an octagon must be a multiple of 3.")
        return np.abs(x) + np.abs(y) <= r + r // 3
    if kind == "square":
        w = int(_scalar(p[0], "width"))
        return np.ones((w, w), dtype=bool)
    if kind == "rectangle":
        s = _vec(p[0], "size")
        if s.size != 2:
            raise MatlabError("strel: a rectangle needs a size [rows cols].")
        return np.ones((int(s[0]), int(s[1])), dtype=bool)
    if kind == "cube":
        w = int(_scalar(p[0], "width"))
        return np.ones((w, w, w), dtype=bool)
    if kind == "line":
        if len(p) < 2:
            raise MatlabError("strel: a line needs a length and an angle in degrees.")
        return _line_strel(_scalar(p[0], "length"), _scalar(p[1], "angle"))
    raise MatlabError(f"strel: unknown shape '{kind}'. Use 'disk', 'square', 'rectangle', 'line', "
                      "'diamond', 'octagon' or a matrix of zeros and ones.")


@builtin("getnhood")
def _getnhood(interp, args, nargout):
    nargs(args, 1, 1, "getnhood")
    return _img(args[0], "getnhood", "structuring element") != 0


def _nhood(v, name: str, ndim: int) -> np.ndarray:
    se = _img(v, name, "structuring element") != 0
    if se.ndim > ndim:
        raise MatlabError(f"{name}: the structuring element has more dimensions than the image.")
    if not se.any():
        raise MatlabError(f"{name}: the structuring element must have at least one nonzero element.")
    return se.reshape(se.shape + (1,) * (ndim - se.ndim))


def _morph(a: np.ndarray, se: np.ndarray, dilate: bool) -> np.ndarray:
    nd = _nd()
    work = a.astype(np.uint8) if a.dtype == bool else a
    if work.dtype.kind == "f":
        fill = -np.inf if dilate else np.inf
    elif a.dtype == bool:
        fill = 0 if dilate else 1
    else:
        info = np.iinfo(work.dtype)
        fill = info.min if dilate else info.max
    if dilate:
        # dilation takes the maximum over the reflected neighbourhood
        flipped = se[tuple(slice(None, None, -1) for _ in range(se.ndim))]
        origin = tuple(0 for _ in se.shape)
        out = nd.maximum_filter(work, footprint=flipped, mode="constant", cval=fill, origin=origin)
    else:
        out = nd.minimum_filter(work, footprint=se, mode="constant", cval=fill, origin=_origin(se.shape))
    return out.astype(bool) if a.dtype == bool else out


def _morph_fn(name: str, steps: tuple[bool, ...]):
    def fn(interp, args, nargout):
        nargs(args, 2, 4, name)
        a = _img(args[0], name)
        se = _nhood(args[1], name, max(a.ndim, 2))
        out = a
        for dilate in steps:
            out = _morph(out, se, dilate)
        return out
    return fn


builtin("imdilate")(_morph_fn("imdilate", (True,)))
builtin("imerode")(_morph_fn("imerode", (False,)))
builtin("imopen")(_morph_fn("imopen", (False, True)))
builtin("imclose")(_morph_fn("imclose", (True, False)))


@builtin("imtophat")
def _imtophat(interp, args, nargout):
    nargs(args, 2, 2, "imtophat")
    a = _img(args[0], "imtophat")
    se = _nhood(args[1], "imtophat", max(a.ndim, 2))
    opened = _morph(_morph(a, se, False), se, True)
    if a.dtype == bool:
        return a & ~opened
    return _cast(a.astype(float) - opened.astype(float), a.dtype) if a.dtype.kind in "iu" else a - opened


@builtin("imbothat")
def _imbothat(interp, args, nargout):
    nargs(args, 2, 2, "imbothat")
    a = _img(args[0], "imbothat")
    se = _nhood(args[1], "imbothat", max(a.ndim, 2))
    closed = _morph(_morph(a, se, True), se, False)
    if a.dtype == bool:
        return closed & ~a
    return _cast(closed.astype(float) - a.astype(float), a.dtype) if a.dtype.kind in "iu" else closed - a


def _conn_structure(v, default: int, name: str) -> tuple[np.ndarray, int]:
    conn = default if v is None else int(_scalar(v, "connectivity"))
    if conn == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), 4
    if conn == 8:
        return np.ones((3, 3), dtype=bool), 8
    raise MatlabError(f"{name}: connectivity must be 4 or 8.")


def _label(bw: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components numbered in column-major order of their first pixel."""
    lab, n = _nd().label(bw.T, structure=structure)
    return lab.T, int(n)


def _binary2d(v, name: str) -> np.ndarray:
    a = _img(v, name)
    if a.ndim != 2:
        raise MatlabError(f"{name}: the image must be 2-D.")
    return a != 0


@builtin("bwlabel")
def _bwlabel(interp, args, nargout):
    nargs(args, 1, 2, "bwlabel")
    bw = _binary2d(args[0], "bwlabel")
    structure, _ = _conn_structure(args[1] if len(args) > 1 else None, 8, "bwlabel")
    lab, n = _label(bw, structure)
    lab = lab.astype(float)
    return (lab, float(n)) if nargout > 1 else lab


def _pixel_lists(lab: np.ndarray, n: int) -> list[np.ndarray]:
    """Zero-based column-major linear indices of every label 1..n."""
    flat = lab.ravel(order="F")
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    bounds = np.searchsorted(sorted_labels, np.arange(1, n + 2))
    return [order[bounds[k]:bounds[k + 1]] for k in range(n)]


def _idx_cell(lists: list[np.ndarray]) -> Cell:
    data = np.empty((1, len(lists)), dtype=object)
    for k, idx in enumerate(lists):
        data[0, k] = (idx + 1).astype(float).reshape(-1, 1)
    return Cell(data)


@builtin("bwconncomp")
def _bwconncomp(interp, args, nargout):
    nargs(args, 1, 2, "bwconncomp")
    bw = _binary2d(args[0], "bwconncomp")
    structure, conn = _conn_structure(args[1] if len(args) > 1 else None, 8, "bwconncomp")
    lab, n = _label(bw, structure)
    return Struct.scalar({"Connectivity": float(conn), "ImageSize": np.array([bw.shape], dtype=float),
                          "NumObjects": float(n), "PixelIdxList": _idx_cell(_pixel_lists(lab, n))})


def _cc_to_labels(cc: Struct, name: str) -> tuple[np.ndarray, int]:
    s = cc.data.flat[0]
    if "PixelIdxList" not in s or "ImageSize" not in s:
        raise MatlabError(f"{name}: the struct must come from bwconncomp.")
    shape = tuple(int(v) for v in np.ravel(s["ImageSize"]))
    flat = np.zeros(int(np.prod(shape)), dtype=np.int64)
    lists = s["PixelIdxList"].values_f()
    for k, idx in enumerate(lists):
        flat[np.ravel(idx).astype(np.int64) - 1] = k + 1
    return flat.reshape(shape, order="F"), len(lists)


@builtin("labelmatrix")
def _labelmatrix(interp, args, nargout):
    nargs(args, 1, 1, "labelmatrix")
    if not isinstance(args[0], Struct):
        raise MatlabError("labelmatrix: the input must be the struct returned by bwconncomp.")
    lab, n = _cc_to_labels(args[0], "labelmatrix")
    return lab.astype(np.uint8 if n <= 255 else (np.uint16 if n <= 65535 else np.float64))


@builtin("bwareaopen")
def _bwareaopen(interp, args, nargout):
    nargs(args, 2, 3, "bwareaopen")
    bw = _binary2d(args[0], "bwareaopen")
    least = _scalar(args[1], "minimum size")
    structure, _ = _conn_structure(args[2] if len(args) > 2 else None, 8, "bwareaopen")
    lab, n = _label(bw, structure)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= least
    keep[0] = False
    return keep[lab]


@builtin("bwperim")
def _bwperim(interp, args, nargout):
    nargs(args, 1, 2, "bwperim")
    bw = _binary2d(args[0], "bwperim")
    structure, _ = _conn_structure(args[1] if len(args) > 1 else None, 4, "bwperim")
    inner = _nd().binary_erosion(bw, structure=structure, border_value=0)
    return bw & ~inner


@builtin("imfill")
def _imfill(interp, args, nargout):
    nargs(args, 1, 3, "imfill")
    a = _img(args[0], "imfill")
    opts = [text(x).lower() for x in args[1:] if is_text(x)]
    nums = [x for x in args[1:] if not is_text(x)]
    if a.dtype != bool and not np.all((a == 0) | (a == 1)):
        raise MatlabError("imfill: only binary images are supported (use imfill(BW, 'holes')).")
    bw = a != 0
    if bw.ndim != 2:
        raise MatlabError("imfill: the image must be 2-D.")
    conn = 4
    if nums and nums[-1].size == 1 and ("holes" in opts or len(nums) > 1) and int(nums[-1].flat[0]) in (4, 8):
        conn = int(nums.pop().flat[0])
    # the background connectivity is the complement of the fill connectivity's reach
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool) if conn == 4 else np.ones((3, 3), bool)
    if "holes" in opts or not nums:
        if "holes" not in opts:
            raise MatlabError("imfill: interactive filling is not supported; use imfill(BW, 'holes') or "
                              "give start locations.")
        return _nd().binary_fill_holes(bw, structure=structure)
    loc = nums[0].astype(float)
    if loc.ndim == 2 and loc.shape[1] == 2 and loc.shape[0] >= 1:
        rows, cols = loc[:, 0].astype(int) - 1, loc[:, 1].astype(int) - 1
    else:
        lin = loc.ravel().astype(int) - 1
        rows, cols = lin % bw.shape[0], lin // bw.shape[0]
    seed = np.zeros(bw.shape, dtype=bool)
    seed[rows, cols] = True
    grown = _nd().binary_propagation(seed & ~bw, structure=structure, mask=~bw)
    return bw | grown


@builtin("bwdist")
def _bwdist(interp, args, nargout):
    nargs(args, 1, 2, "bwdist")
    bw = _img(args[0], "bwdist") != 0
    method = text(args[1]).lower() if len(args) > 1 else "euclidean"
    nd = _nd()
    if not bw.any():
        dist = np.full(bw.shape, np.inf)
        idx = np.zeros(bw.shape)
        return (dist.astype(np.float32), idx.astype(np.uint32)) if nargout > 1 else dist.astype(np.float32)
    if method == "euclidean":
        dist, where = nd.distance_transform_edt(~bw, return_indices=True)
    elif method in ("cityblock", "chessboard"):
        dist, where = nd.distance_transform_cdt(~bw, metric="taxicab" if method == "cityblock" else "chessboard",
                                                return_indices=True)
    elif method == "quasi-euclidean":
        dist, where = nd.distance_transform_edt(~bw, return_indices=True)
        dr = np.abs(where[0] - np.indices(bw.shape)[0])
        dc = np.abs(where[1] - np.indices(bw.shape)[1])
        dist = np.where(dr > dc, dr + (np.sqrt(2) - 1) * dc, (np.sqrt(2) - 1) * dr + dc)
    else:
        raise MatlabError("bwdist: the method must be 'euclidean', 'cityblock', 'chessboard' or "
                          "'quasi-euclidean'.")
    dist = dist.astype(np.float32)
    if nargout > 1:
        lin = np.ravel_multi_index(tuple(where), bw.shape, order="F") + 1
        return dist, lin.astype(np.uint32)
    return dist


_STEPS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # clockwise from east


def _chain_code(mask: np.ndarray) -> list[int]:
    """Directions (indices into _STEPS) walked along the outer boundary of an 8-connected region."""
    padded = np.pad(mask, 1, mode="constant")
    start = tuple(int(v) for v in np.argwhere(padded)[0])  # top-most pixel, left-most in its row
    codes: list[int] = []
    pos, scan = start, 4  # nothing lies to the west or above the start pixel
    limit = 4 * padded.size + 8
    while len(codes) < limit:
        for k in range(8):
            d = (scan + k) % 8
            if padded[pos[0] + _STEPS[d][0], pos[1] + _STEPS[d][1]]:
                break
        else:
            return codes  # a single pixel
        if pos == start and codes and d == codes[0]:
            return codes  # back at the start, about to repeat the first move
        codes.append(d)
        pos = (pos[0] + _STEPS[d][0], pos[1] + _STEPS[d][1])
        # resume the clockwise search at the background pixel examined just before this move
        scan = (d + 6) % 8 if d % 2 == 0 else (d + 5) % 8
    return codes


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Zero-based [row col] points along the outer boundary, clockwise, ending where it starts."""
    start = np.argwhere(mask)[0]
    steps = np.array([_STEPS[c] for c in _chain_code(mask)], dtype=np.int64).reshape(-1, 2)
    pts = np.vstack([start, start + np.cumsum(steps, axis=0)]) if steps.size else np.vstack([start, start])
    return pts


@builtin("bwboundaries")
def _bwboundaries(interp, args, nargout):
    nargs(args, 1, 3, "bwboundaries")
    bw = _binary2d(args[0], "bwboundaries")
    conn = 8
    for extra in args[1:]:
        if is_text(extra):
            if text(extra).lower() not in ("holes", "noholes"):
                raise MatlabError("bwboundaries: the option must be 'holes' or 'noholes'.")
        else:
            conn = int(_scalar(extra, "connectivity"))
    if conn != 8:
        raise MatlabError("bwboundaries: only 8-connected tracing is supported.")
    lab, n = _label(bw, np.ones((3, 3), dtype=bool))
    data = np.empty((n, 1), dtype=object)
    objects = _nd().find_objects(lab)
    for k in range(n):
        sl = objects[k]
        pts = _boundary(lab[sl] == k + 1)
        data[k, 0] = (pts + [sl[0].start + 1, sl[1].start + 1]).astype(float)
    outs = (Cell(data), lab.astype(float), float(n), np.zeros((n, n)))
    return outs[:nargout] if nargout > 1 else outs[0]


@builtin("poly2mask")
def _poly2mask(interp, args, nargout):
    nargs(args, 4, 4, "poly2mask")
    from matplotlib.path import Path
    x, y = _vec(args[0], "x"), _vec(args[1], "y")
    rows, cols = to_int(args[2], "rows"), to_int(args[3], "columns")
    if x.size != y.size:
        raise MatlabError("poly2mask: x and y must have the same number of elements.")
    if x.size < 3 or rows < 1 or cols < 1:
        return np.zeros((max(rows, 0), max(cols, 0)), dtype=bool)
    cc, rr = np.meshgrid(np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float))
    inside = Path(np.column_stack([x, y])).contains_points(np.column_stack([cc.ravel(), rr.ravel()]), radius=1e-9)
    return inside.reshape(rows, cols)


def _neighbours(bw: np.ndarray) -> list[np.ndarray]:
    """The eight neighbours of every pixel, clockwise from north (outside counts as false)."""
    p = np.pad(bw, 1, mode="constant")
    r, c = bw.shape
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return [p[1 + dr:1 + dr + r, 1 + dc:1 + dc + c] for dr, dc in offsets]


def _thin_once(bw: np.ndarray) -> np.ndarray:
    """One pass (two sub-iterations) of Zhang and Suen's thinning."""
    for first in (True, False):
        n = _neighbours(bw)
        count = sum(x.astype(np.int64) for x in n)
        ring = n + [n[0]]
        transitions = sum((~ring[k] & ring[k + 1]).astype(np.int64) for k in range(8))
        north, east, south, west = n[0], n[2], n[4], n[6]
        if first:
            cond = ~(north & east & south) & ~(east & south & west)
        else:
            cond = ~(north & east & west) & ~(north & south & west)
        bw = bw & ~((count >= 2) & (count <= 6) & (transitions == 1) & cond)
    return bw


def _bwmorph_step(bw: np.ndarray, op: str) -> np.ndarray:
    nd = _nd()
    box = np.ones((3, 3), dtype=bool)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if op == "dilate":
        return nd.binary_dilation(bw, box)
    if op == "erode":
        return nd.binary_erosion(bw, box, border_value=1)
    if op == "open":
        return nd.binary_dilation(nd.binary_erosion(bw, box, border_value=1), box)
    if op == "close":
        return nd.binary_erosion(nd.binary_dilation(bw, box), box, border_value=1)
    n = _neighbours(bw)
    count = sum(x.astype(np.int64) for x in n)
    four = n[0].astype(np.int64) + n[2] + n[4] + n[6]
    if op == "remove":
        return bw & ~(four == 4)
    if op == "clean":
        return bw & (count > 0)
    if op == "fill":
        return bw | (four == 4)
    if op == "majority":
        return (count + bw) >= 5
    if op == "endpoints":
        return bw & (count == 1)
    if op == "branchpoints":
        ring = n + [n[0]]
        arms = sum((~ring[k] & ring[k + 1]).astype(np.int64) for k in range(8))
        return bw & (arms >= 3)
    if op == "spur":
        return bw & ~(count == 1)
    if op == "diag":
        out = bw.copy()
        # join pixels that touch only at a corner
        out |= (n[0] & n[2] & ~n[1]) | (n[2] & n[4] & ~n[3]) | (n[4] & n[6] & ~n[5]) | (n[6] & n[0] & ~n[7])
        return out
    if op == "bridge":
        lab, _ = nd.label(bw, structure=box)
        p = np.pad(lab, 1, mode="constant")
        r, c = bw.shape
        around = [p[1 + dr:1 + dr + r, 1 + dc:1 + dc + c] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        stack = np.stack(around)
        top = stack.max(axis=0)
        low = np.where(stack > 0, stack, np.iinfo(stack.dtype).max).min(axis=0)
        return bw | (~bw & (top > 0) & (low != top))
    if op in ("thin", "skel"):
        return _thin_once(bw)
    if op == "perim4":
        return bw & ~nd.binary_erosion(bw, box, border_value=0)
    if op == "perim8":
        return bw & ~nd.binary_erosion(bw, cross, border_value=0)
    raise MatlabError(f"bwmorph: the operation '{op}' is not supported. Supported: bridge, branchpoints, clean, "
                      "close, diag, dilate, endpoints, erode, fill, majority, open, perim4, perim8, remove, skel, "
                      "spur, thin.")


@builtin("bwmorph")
def _bwmorph(interp, args, nargout):
    nargs(args, 2, 3, "bwmorph")
    bw = _binary2d(args[0], "bwmorph")
    op = text(args[1], "operation").lower()
    times = _scalar(args[2], "number of repetitions") if len(args) > 2 else 1.0
    if times < 0:
        raise MatlabError("bwmorph: the number of repetitions must not be negative.")
    done = 0
    while done < times:
        nxt = _bwmorph_step(bw, op)
        done += 1
        if np.array_equal(nxt, bw):
            break
        bw = nxt
        if op in ("endpoints", "branchpoints", "perim4", "perim8"):
            break
    return bw


def _perimeter(mask: np.ndarray) -> float:
    """Length of a region's outer boundary from its chain code.

    Straight and diagonal steps are weighted, with a correction per change of direction, following
    the estimator of Vossepoel and Smeulders (1982), which is nearly unbiased for smooth outlines.
    """
    lab, n = _nd().label(mask, structure=np.ones((3, 3)))
    total = 0.0
    for k in range(1, n + 1):
        codes = np.array(_chain_code(lab == k) if n > 1 else _chain_code(mask), dtype=np.int64)
        if codes.size == 0:
            continue
        even = int((codes % 2 == 0).sum())
        corners = int((codes != np.roll(codes, 1)).sum())
        total += 0.980 * even + 1.406 * (codes.size - even) - 0.091 * corners
    return float(total)


_SHAPE_PROPS = ["Area", "Centroid", "BoundingBox", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
                "Orientation", "EquivDiameter", "Extent", "Perimeter", "Circularity", "Image", "FilledImage",
                "FilledArea", "EulerNumber", "PixelIdxList", "PixelList", "SubarrayIdx"]
_VALUE_PROPS = ["MeanIntensity", "MinIntensity", "MaxIntensity", "PixelValues", "WeightedCentroid"]


def _region(prop_names: list[str], idx: np.ndarray, shape: tuple[int, int], values) -> dict:
    rows, cols = idx % shape[0], idx // shape[0]
    out: dict = {}
    area = float(idx.size)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    mask = None

    def image():
        nonlocal mask
        if mask is None:
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rows - r0, cols - c0] = True
        return mask

    def ellipse():
        x = cols - cols.mean()
        y = -(rows - rows.mean())
        uxx = np.mean(x * x) + 1 / 12  # each pixel is a unit square, which adds 1/12
        uyy = np.mean(y * y) + 1 / 12
        uxy = np.mean(x * y)
        common = np.sqrt((uxx - uyy) ** 2 + 4 * uxy * uxy)
        major = 2 * np.sqrt(2) * np.sqrt(uxx + uyy + common)
        minor = 2 * np.sqrt(2) * np.sqrt(max(uxx + uyy - common, 0.0))
        ecc = 2 * np.sqrt(max((major / 2) ** 2 - (minor / 2) ** 2, 0.0)) / major if major else 0.0
        if uyy > uxx:
            num_, den = uyy - uxx + common, 2 * uxy
        else:
            num_, den = 2 * uxy, uxx - uyy + common
        angle = 0.0 if (num_ == 0 and den == 0) else float(np.degrees(np.arctan2(num_, den)))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        return float(major), float(minor), float(ecc), angle

    shape_cache = None
    for p in prop_names:
        if p == "Area":
            out[p] = area
        elif p == "Centroid":
            out[p] = np.array([[cols.mean() + 1, rows.mean() + 1]])
        elif p == "BoundingBox":
            out[p] = np.array([[c0 + 0.5, r0 + 0.5, c1 - c0 + 1.0, r1 - r0 + 1.0]])
        elif p in ("MajorAxisLength", "MinorAxisLength", "Eccentricity", "Orientation"):
            shape_cache = shape_cache or ellipse()
            out[p] = shape_cache[["MajorAxisLength", "MinorAxisLength", "Eccentricity", "Orientation"].index(p)]
        elif p == "EquivDiameter":
            out[p] = float(np.sqrt(4 * area / np.pi))
        elif p == "Extent":
            out[p] = area / float((r1 - r0 + 1) * (c1 - c0 + 1))
        elif p == "Perimeter":
            out[p] = _perimeter(image())
        elif p == "Circularity":
            per = _perimeter(image())
            out[p] = float(4 * np.pi * area / per ** 2) if per else float("inf")
        elif p == "Image":
            out[p] = image().copy()
        elif p in ("FilledImage", "FilledArea"):
            filled = _nd().binary_fill_holes(image())
            out[p] = filled if p == "FilledImage" else float(filled.sum())
        elif p == "EulerNumber":
            filled = _nd().binary_fill_holes(image())
            _, holes = _nd().label(filled & ~image(), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            out[p] = float(1 - holes)
        elif p == "PixelIdxList":
            out[p] = (idx + 1).astype(float).reshape(-1, 1)
        elif p == "PixelList":
            out[p] = np.column_stack([cols + 1, rows + 1]).astype(float)
        elif p == "SubarrayIdx":
            out[p] = Cell.from_list([np.arange(r0 + 1, r1 + 2, dtype=float).reshape(1, -1),
                                     np.arange(c0 + 1, c1 + 2, dtype=float).reshape(1, -1)])
        elif p in _VALUE_PROPS:
            if values is None:
                raise MatlabError(f"regionprops: '{p}' needs the grayscale image as the second input.")
            v = values.ravel(order="F")[idx]
            vf = v.astype(float)
            if p == "MeanIntensity":
                out[p] = float(vf.mean())
            elif p == "MinIntensity":
                out[p] = v.min().reshape(1, 1)
            elif p == "MaxIntensity":
                out[p] = v.max().reshape(1, 1)
            elif p == "PixelValues":
                out[p] = v.reshape(-1, 1)
            else:
                total = vf.sum()
                out[p] = np.array([[(vf * (cols + 1)).sum() / total, (vf * (rows + 1)).sum() / total]]) \
                    if total else np.array([[np.nan, np.nan]])
    return out


@builtin("regionprops")
def _regionprops(interp, args, nargout):
    nargs(args, 1, None, "regionprops")
    rest = list(args)
    if rest and is_text(rest[0]) and text(rest[0]).lower() in ("struct", "table"):
        if text(rest.pop(0)).lower() == "table":
            raise MatlabError("regionprops: table output is not supported; use the default struct array.")
    if not rest:
        raise MatlabError("regionprops: an image is required.")
    src = rest.pop(0)
    if isinstance(src, Struct):
        lab, n = _cc_to_labels(src, "regionprops")
    else:
        a = _img(src, "regionprops")
        if a.ndim != 2:
            raise MatlabError("regionprops: the image must be 2-D.")
        if a.dtype == bool:
            lab, n = _label(a, np.ones((3, 3), dtype=bool))
        else:
            if a.size and (a.min() < 0 or np.any(a != np.round(a))):
                raise MatlabError("regionprops: a label matrix must hold non-negative integers.")
            lab = a.astype(np.int64)
            n = int(lab.max()) if lab.size else 0
    values = None
    if rest and not is_text(rest[0]) and not isinstance(rest[0], Cell):
        values = _img(rest.pop(0), "regionprops", "grayscale image")
        if values.shape != lab.shape:
            raise MatlabError("regionprops: the grayscale image must have the size of the label image.")
    wanted: list[str] = []
    for r in rest:
        wanted.extend([text(x) for x in r.values_f()] if isinstance(r, Cell) else [text(r)])
    known = {p.lower(): p for p in _SHAPE_PROPS + _VALUE_PROPS}
    known.update({"majoraxislength": "MajorAxisLength", "minoraxislength": "MinorAxisLength"})
    names: list[str] = []
    for w in wanted or ["basic"]:
        key = w.lower()
        if key == "basic":
            group = ["Area", "Centroid", "BoundingBox"]
        elif key == "all":
            group = _SHAPE_PROPS + (_VALUE_PROPS if values is not None else [])
        elif key in known:
            group = [known[key]]
        else:
            raise MatlabError(f"regionprops: the property '{w}' is not supported. Supported properties: "
                              + ", ".join(_SHAPE_PROPS + _VALUE_PROPS) + ".")
        names.extend(g for g in group if g not in names)
    lists = _pixel_lists(lab, n)
    data = np.empty((n, 1), dtype=object)
    for k, idx in enumerate(lists):
        if idx.size == 0:
            empty: dict = {}
            for p in names:
                empty[p] = 0.0 if p in ("Area", "FilledArea") else np.zeros((0, 0))
            data[k, 0] = empty
        else:
            data[k, 0] = _region(names, idx, lab.shape, values)
    return Struct(data, names)


# -- geometry -------------------------------------------------------------------------------------

def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= 1, 1.5 * ax ** 3 - 2.5 * ax ** 2 + 1,
                    np.where(ax < 2, -0.5 * ax ** 3 + 2.5 * ax ** 2 - 4 * ax + 2, 0.0))


def _lanczos(a: int):
    def k(x):
        return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)
    return k


_KERNELS = {
    "nearest": (lambda x: ((x >= -0.5) & (x < 0.5)).astype(float), 1.0),
    "box": (lambda x: ((x >= -0.5) & (x < 0.5)).astype(float), 1.0),
    "bilinear": (lambda x: np.maximum(1 - np.abs(x), 0.0), 2.0),
    "triangle": (lambda x: np.maximum(1 - np.abs(x), 0.0), 2.0),
    "bicubic": (_cubic, 4.0),
    "cubic": (_cubic, 4.0),
    "lanczos2": (_lanczos(2), 4.0),
    "lanczos3": (_lanczos(3), 6.0),
}


def _resize_axis(a: np.ndarray, axis: int, out_len: int, scale: float, method: str, antialias: bool) -> np.ndarray:
    """Resample one dimension: output sample x sits at input coordinate x/scale + (1 - 1/scale)/2."""
    kernel, width = _KERNELS[method]
    n = a.shape[axis]
    if method == "nearest":
        u = (np.arange(1, out_len + 1) - 0.5) / scale + 0.5
        idx = np.clip(np.floor(u + 0.5).astype(np.int64), 1, n) - 1
        return np.take(a, idx, axis=axis)
    stretch = scale if (antialias and scale < 1) else 1.0
    support = width / stretch
    u = (np.arange(1, out_len + 1) - 0.5) / scale + 0.5
    left = np.floor(u - support / 2).astype(np.int64)
    taps = int(np.ceil(support)) + 2
    idx = left[:, None] + np.arange(taps)[None, :]
    w = stretch * kernel(stretch * (u[:, None] - idx))
    total = w.sum(axis=1, keepdims=True)
    w = w / np.where(total == 0, 1.0, total)
    idx = np.clip(idx, 1, n) - 1  # samples past the edge repeat the border pixel
    moved = np.moveaxis(a, axis, -1)
    out = np.einsum("...ok,ok->...o", moved[..., idx], w)
    return np.moveaxis(out, -1, axis)


def _resize(a: np.ndarray, out_shape: tuple[int, int], scales: tuple[float, float], method: str,
            antialias: bool) -> np.ndarray:
    work = a.astype(float)
    order = sorted((0, 1), key=lambda d: scales[d])
    for d in order:
        if out_shape[d] == a.shape[d] and scales[d] == 1:
            continue
        work = _resize_axis(work, d, out_shape[d], scales[d], method, antialias)
    return work


@builtin("imresize")
def _imresize(interp, args, nargout):
    nargs(args, 2, None, "imresize")
    a = _img(args[0], "imresize")
    rest, opts = _options(list(args[1:]), ("Antialiasing", "Method", "OutputSize", "Scale", "Colormap",
                                           "Dither"), "imresize")
    method = "bicubic"
    spec = None
    for r in rest:
        if is_text(r):
            method = text(r).lower()
        elif spec is None:
            spec = _vec(r, "scale or size")
        else:
            raise MatlabError("imresize: resizing an indexed image with its colormap is not supported; "
                              "convert it with ind2rgb first.")
    if "method" in opts:
        method = text(opts["method"]).lower()
    if method not in _KERNELS:
        raise MatlabError(f"imresize: unknown method '{method}'. Use 'nearest', 'bilinear', 'bicubic', "
                          "'box', 'lanczos2' or 'lanczos3'.")
    if "outputsize" in opts:
        spec = _vec(opts["outputsize"], "OutputSize")
    scale_given = None
    if "scale" in opts:
        scale_given = _vec(opts["scale"], "Scale")
    elif spec is not None and spec.size == 1:
        scale_given = spec
    if a.ndim < 2 or a.size == 0:
        raise MatlabError("imresize: the image must not be empty.")
    rows, cols = a.shape[:2]
    if scale_given is not None:
        sr, sc = (scale_given[0], scale_given[0]) if scale_given.size == 1 else scale_given[:2]
        if sr <= 0 or sc <= 0:
            raise MatlabError("imresize: the scale must be positive.")
        out_shape = (int(np.ceil(sr * rows)), int(np.ceil(sc * cols)))
    elif spec is not None and spec.size == 2:
        r_, c_ = spec
        if np.isnan(r_) and np.isnan(c_):
            raise MatlabError("imresize: at most one of the output sizes may be NaN.")
        if np.isnan(r_):
            sc = c_ / cols
            sr = sc
            out_shape = (int(np.ceil(rows * sr)), int(c_))
        elif np.isnan(c_):
            sr = r_ / rows
            sc = sr
            out_shape = (int(r_), int(np.ceil(cols * sc)))
        else:
            sr, sc = r_ / rows, c_ / cols
            out_shape = (int(r_), int(c_))
    else:
        raise MatlabError("imresize: give a scale factor or an output size [rows cols].")
    if out_shape[0] < 1 or out_shape[1] < 1:
        raise MatlabError("imresize: the output size must be at least 1-by-1.")
    antialias = method != "nearest"
    if "antialiasing" in opts:
        v = opts["antialiasing"]
        antialias = text(v).lower() in ("true", "on") if is_text(v) else bool(np.ravel(v)[0])
    if method == "box" and not antialias:
        method = "nearest"
    out = _resize(a, out_shape, (float(sr), float(sc)), method, antialias)
    if a.dtype == bool:
        return out > 0.5
    return _cast(out, a.dtype) if a.dtype.kind in "iu" else out.astype(a.dtype)


@builtin("imrotate")
def _imrotate(interp, args, nargout):
    nargs(args, 2, 4, "imrotate")
    a = _img(args[0], "imrotate")
    angle = _scalar(args[1], "angle")
    method, bbox = "nearest", "loose"
    for extra in args[2:]:
        key = text(extra).lower()
        if key in ("nearest", "bilinear", "bicubic"):
            method = key
        elif key in ("loose", "crop"):
            bbox = key
        else:
            raise MatlabError(f"imrotate: unknown option '{text(extra)}'.")
    if a.size == 0:
        return a
    quarter = angle / 90.0
    if quarter == int(quarter) and (bbox == "loose" or a.shape[0] == a.shape[1] or int(quarter) % 2 == 0):
        return np.ascontiguousarray(np.rot90(a, int(quarter) % 4, axes=(0, 1)))
    rows, cols = a.shape[:2]
    t = np.deg2rad(angle)
    c, s = np.cos(t), np.sin(t)
    if bbox == "crop":
        out_rows, out_cols = rows, cols
    else:
        out_cols = int(np.ceil(abs(cols * c) + abs(rows * s) - 1e-9))
        out_rows = int(np.ceil(abs(cols * s) + abs(rows * c) - 1e-9))
    yy, xx = np.indices((out_rows, out_cols), dtype=float)
    x = xx - (out_cols - 1) / 2
    y = yy - (out_rows - 1) / 2
    # inverse map: rotate output coordinates back (image y axis points down)
    src_x = c * x - s * y + (cols - 1) / 2
    src_y = s * x + c * y + (rows - 1) / 2
    order = {"nearest": 0, "bilinear": 1, "bicubic": 3}[method]
    nd = _nd()

    def run(p):
        return nd.map_coordinates(p.astype(float), [src_y, src_x], order=order, mode="constant", cval=0.0,
                                  prefilter=order > 1)

    if order == 0:
        ri = np.floor(src_y + 0.5).astype(np.int64)
        ci = np.floor(src_x + 0.5).astype(np.int64)
        ok = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
        out = np.zeros((out_rows, out_cols) + a.shape[2:], dtype=a.dtype)
        out[ok] = a[ri[ok], ci[ok]]
        return out
    if a.ndim > 2:
        flat = a.reshape(rows, cols, -1)
        out = np.stack([run(flat[..., k]) for k in range(flat.shape[2])], axis=-1)
        out = out.reshape((out_rows, out_cols) + a.shape[2:])
    else:
        out = run(a)
    if a.dtype == bool:
        return out > 0.5
    return _cast(out, a.dtype) if a.dtype.kind in "iu" else out.astype(a.dtype)


@builtin("imcrop")
def _imcrop(interp, args, nargout):
    nargs(args, 1, 3, "imcrop")
    a = _img(args[0], "imcrop")
    if len(args) < 2:
        raise MatlabError("imcrop: choosing the rectangle with the mouse is not supported; give "
                          "imcrop(I, [xmin ymin width height]).")
    rect = _vec(args[-1], "rectangle")  # with three inputs the middle one is a colormap, which is not needed
    if rect.size != 4:
        raise MatlabError("imcrop: the rectangle must be [xmin ymin width height].")
    x0, y0, w, h = rect
    c_first = max(int(np.floor(x0 + 0.5)), 1)
    c_last = min(int(np.ceil(x0 + w - 0.5)), a.shape[1])
    r_first = max(int(np.floor(y0 + 0.5)), 1)
    r_last = min(int(np.ceil(y0 + h - 0.5)), a.shape[0])
    if w < 0 or h < 0 or c_last < c_first or r_last < r_first:
        out = np.zeros((0, 0), dtype=a.dtype)
    else:
        out = a[r_first - 1:r_last, c_first - 1:c_last].copy()
    return (out, rect.reshape(1, 4)) if nargout > 1 else out


@builtin("imtranslate")
def _imtranslate(interp, args, nargout):
    nargs(args, 2, None, "imtranslate")
    rest, opts = _options(list(args), ("FillValues", "OutputView"), "imtranslate")
    a = _img(rest[0], "imtranslate")
    shift = _vec(rest[1], "translation")
    method = text(rest[2]).lower() if len(rest) > 2 else "linear"
    order = {"nearest": 0, "linear": 1, "bilinear": 1, "cubic": 3, "bicubic": 3}.get(method)
    if order is None:
        raise MatlabError("imtranslate: the method must be 'nearest', 'linear' or 'cubic'.")
    fill = _scalar(opts["fillvalues"], "FillValues") if "fillvalues" in opts else 0.0
    nd = _nd()

    def run(p):
        return nd.shift(p.astype(float), (shift[1], shift[0]), order=order, mode="constant", cval=fill,
                        prefilter=order > 1)

    out = _per_plane(a, run)
    if a.dtype == bool:
        return out > 0.5
    return _cast(out, a.dtype) if a.dtype.kind in "iu" else out.astype(a.dtype)


# -- image files ------------------------------------------------------------------------------------

_SAMPLE_NOTE = (" MATLAB's sample pictures (cameraman.tif, peppers.png, ...) do not ship with mylab; "
                "use an image file of your own.")


def _find_image(interp, name: str, fmt: str | None, fn: str) -> str:
    candidates = [name]
    if fmt and not os.path.splitext(name)[1]:
        candidates.append(f"{name}.{fmt}")
    for cand in candidates:
        cand = os.path.expanduser(cand)
        if os.path.isfile(cand):
            return cand
        if not os.path.isabs(cand):
            for d in getattr(interp, "path", []):
                p = os.path.join(d, cand)
                if os.path.isfile(p):
                    return p
    raise MatlabError(f"{fn}: file \"{name}\" does not exist.{_SAMPLE_NOTE}", "MATLAB:imagesci:imread:fileDoesNotExist")


def _open_image(interp, source: str, fmt: str | None, fn: str):
    Image = _pil()
    if source.lower().startswith(("http://", "https://")):
        import io
        from .fileops import _fetch
        data = io.BytesIO(_fetch(source, fn))
        try:
            return Image.open(data), source
        except Exception:
            raise MatlabError(f"{fn}: the data at '{source}' is not an image in a supported format.")
    path = _find_image(interp, source, fmt, fn)
    try:
        return Image.open(path), path
    except Exception:
        raise MatlabError(f"{fn}: '{source}' is not an image file in a supported format (PNG, JPEG, BMP, "
                          "GIF, TIFF, ...).", "MATLAB:imagesci:imread:fileFormat")


def _palette(im) -> np.ndarray:
    raw = im.getpalette("RGB") if im.palette is not None else None
    if not raw:
        return np.zeros((0, 3))
    return np.asarray(raw, dtype=float).reshape(-1, 3) / 255.0


def _frame_arrays(im):
    """(pixels, colormap or None, alpha or None) of the current frame of a Pillow image."""
    mode = im.mode
    cmap = alpha = None
    if mode == "P":
        cmap = _palette(im)
        pixels = np.asarray(im).copy()
        if cmap.shape[0]:
            used = int(pixels.max()) + 1 if pixels.size else 1
            keep = max(used, 1)
            # GIF and PNG palettes are padded to 256 entries by the decoder; keep the real ones
            real = len(im.palette.colors) if getattr(im.palette, "colors", None) else cmap.shape[0]
            cmap = cmap[:max(keep, min(real, cmap.shape[0]))]
    elif mode == "1":
        pixels = np.asarray(im).astype(bool)
    elif mode == "L":
        pixels = np.asarray(im).copy()
    elif mode == "LA":
        both = np.asarray(im)
        pixels, alpha = both[..., 0].copy(), both[..., 1].copy()
    elif mode == "RGB":
        pixels = np.asarray(im).copy()
    elif mode == "RGBA":
        both = np.asarray(im)
        pixels, alpha = both[..., :3].copy(), both[..., 3].copy()
    elif mode in ("I;16", "I;16L", "I;16B", "I;16N"):
        pixels = np.asarray(im).astype(np.uint16)
    elif mode == "I":
        raw = np.asarray(im)
        pixels = raw.astype(np.uint16) if raw.size and raw.min() >= 0 and raw.max() <= 65535 else raw.astype(np.int32)
    elif mode == "F":
        pixels = np.asarray(im).astype(np.float32)
    elif mode == "PA":
        conv = im.convert("RGBA")
        both = np.asarray(conv)
        pixels, alpha = both[..., :3].copy(), both[..., 3].copy()
    else:
        pixels = np.asarray(im.convert("RGB")).copy()
    return pixels, cmap, alpha


def _reindex(rgb: np.ndarray, first_map) -> tuple[np.ndarray, np.ndarray]:
    """Indexed form of an RGB frame, using ``first_map`` when it holds every colour of the frame."""
    flat = rgb.reshape(-1, 3)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    if first_map is not None and first_map.shape[0]:
        table = np.clip(_round_half_away(first_map * 255), 0, 255).astype(np.int64)
        keys = table[:, 0] * 65536 + table[:, 1] * 256 + table[:, 2]
        lookup = {}
        for pos, key in enumerate(keys):
            lookup.setdefault(int(key), pos)
        want = colours[:, 0].astype(np.int64) * 65536 + colours[:, 1].astype(np.int64) * 256 + colours[:, 2]
        if all(int(w) in lookup for w in want):
            remap = np.array([lookup[int(w)] for w in want], dtype=np.int64)
            return remap[inverse].reshape(rgb.shape[:2]).astype(np.uint8), first_map
    if colours.shape[0] > 256:
        Image = _pil()
        q = Image.fromarray(np.ascontiguousarray(rgb)).quantize(256, dither=Image.Dither.NONE)
        return np.asarray(q).copy(), _palette(q)
    return inverse.reshape(rgb.shape[:2]).astype(np.uint8), colours.astype(float) / 255.0


@builtin("imread")
def _imread(interp, args, nargout):
    nargs(args, 1, None, "imread")
    source = text(args[0], "file name")
    rest = list(args[1:])
    fmt = None
    frames = None
    rest, opts = _options(rest, ("Frames", "Index", "BackgroundColor", "PixelRegion", "Info"), "imread")
    for r in rest:
        if is_text(r):
            fmt = text(r).lower().lstrip(".")
        else:
            frames = r
    for key in ("frames", "index"):
        if key in opts:
            frames = opts[key]
    im, _path = _open_image(interp, source, fmt, "imread")
    try:
        total = getattr(im, "n_frames", 1)
        if frames is None:
            wanted = [0]
        elif is_text(frames):
            if text(frames).lower() != "all":
                raise MatlabError("imread: 'Frames' must be a frame number, a vector of them, or 'all'.")
            wanted = list(range(total))
        else:
            wanted = [int(v) - 1 for v in _vec(frames, "frame numbers")]
            if any(k < 0 or k >= total for k in wanted):
                raise MatlabError(f"imread: frame numbers must be between 1 and {total}.")
        results = []
        cmap = alpha = None
        is_gif = (im.format or "").upper() == "GIF"
        first_map = None
        if is_gif and total > 1:
            first_map = _frame_arrays(im)[1]
        for k in wanted:
            if total > 1:
                im.seek(k)
            pixels, cm, al = _frame_arrays(im)
            if is_gif and pixels.ndim == 3:
                # the decoder hands later animation frames back as RGB; GIF frames are indexed
                pixels, cm = _reindex(pixels, first_map)
                al = None
            if cmap is None:
                cmap, alpha = cm, al
            results.append(pixels)
    finally:
        im.close()
    if len(results) == 1:
        out = results[0]
    else:
        if any(r.shape != results[0].shape for r in results):
            raise MatlabError("imread: the frames of this file differ in size or type.")
        out = np.stack(results, axis=-1)
        if out.ndim == 3:  # frames of a 2-D image go along the fourth dimension
            out = out.reshape(out.shape[0], out.shape[1], 1, out.shape[2])
    if "pixelregion" in opts:
        reg = opts["pixelregion"]
        if not isinstance(reg, Cell) or len(reg.values_f()) != 2:
            raise MatlabError("imread: 'PixelRegion' must be {rows, cols}.")
        sl = []
        for v in reg.values_f():
            p = _vec(v, "PixelRegion")
            start, step, stop = (p[0], 1, p[1]) if p.size == 2 else (p[0], p[1], p[2])
            sl.append(slice(int(start) - 1, int(stop), int(step)))
        out = out[sl[0], sl[1]]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[..., 0]
    if nargout <= 1:
        return out
    cmap_out = cmap if cmap is not None else np.zeros((0, 0))
    alpha_out = alpha if alpha is not None else np.zeros((0, 0))
    return (out, cmap_out, alpha_out)[:nargout]


_FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "bmp": "BMP", "gif": "GIF", "tif": "TIFF",
            "tiff": "TIFF", "pbm": "PPM", "pgm": "PPM", "ppm": "PPM", "pnm": "PPM", "webp": "WEBP",
            "ico": "ICO", "pcx": "PCX", "tga": "TGA", "jp2": "JPEG2000", "j2k": "JPEG2000"}


def _pil_from_array(a: np.ndarray, cmap, fmt: str, name: str):
    """A Pillow image for a MATLAB image array (and optional colormap)."""
    Image = _pil()
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if cmap is not None:
        if a.ndim != 2:
            raise MatlabError(f"{name}: an indexed image must be 2-D.")
        pal = np.clip(_round_half_away(np.asarray(cmap, dtype=float) * 255), 0, 255).astype(np.uint8)
        if pal.ndim != 2 or pal.shape[1] != 3 or pal.shape[0] > 256:
            raise MatlabError(f"{name}: the colormap must be N-by-3 with at most 256 colours.")
        idx = np.clip(_index_base(a), 0, max(pal.shape[0] - 1, 0)).astype(np.uint8)
        im = Image.fromarray(idx)
        im.putpalette(pal.tobytes(), "RGB")
        return im
    if a.ndim == 2:
        if a.dtype == bool:
            return Image.fromarray(a.astype(np.uint8) * 255).convert("1")
        if a.dtype == np.uint16 and fmt in ("PNG", "TIFF"):
            return Image.fromarray(np.ascontiguousarray(a))
        if a.dtype == np.uint8:
            return Image.fromarray(np.ascontiguousarray(a))
        return Image.fromarray(_from_unit(_to_unit(a), np.uint8))
    if _is_rgb(a):
        return Image.fromarray(np.ascontiguousarray(_from_unit(_to_unit(a), np.uint8)))
    raise MatlabError(f"{name}: the image must be M-by-N (grayscale or indexed) or M-by-N-by-3 (RGB).")


def _gif_frames(interp) -> dict:
    return interp.state.setdefault("gif_frames", {})


@builtin("imwrite")
def _imwrite(interp, args, nargout):
    nargs(args, 2, None, "imwrite")
    a = _img(args[0], "imwrite")
    rest = list(args[1:])
    cmap = None
    if rest and not is_text(rest[0]):
        cmap = _img(rest.pop(0), "imwrite", "colormap")
        if cmap.size == 0:
            cmap = None
    if not rest or not is_text(rest[0]):
        raise MatlabError("imwrite: a file name is required.")
    fname = os.path.expanduser(text(rest.pop(0)))
    names = ("Quality", "WriteMode", "DelayTime", "LoopCount", "Alpha", "Transparency", "Compression",
             "BitDepth", "DisposalMethod", "Comment", "Mode", "TransparentColor", "BackgroundColor",
             "Resolution", "Author", "Description", "Title", "Software", "Lossless", "RowsPerStrip",
             "ColorSpace", "Interlace", "InterlaceType", "ScreenSize", "Location")
    fmt_name = None
    if rest and is_text(rest[0]) and text(rest[0]).lower() not in {n.lower() for n in names}:
        fmt_name = text(rest.pop(0)).lower().lstrip(".")
    rest, opts = _options(rest, names, "imwrite")
    if rest:
        raise MatlabError("imwrite: options must be given as name, value pairs.")
    ext = fmt_name or os.path.splitext(fname)[1].lower().lstrip(".")
    if ext not in _FORMATS:
        raise MatlabError(f"imwrite: cannot tell the image format from '{fname}'. Use an extension such as "
                          ".png, .jpg, .bmp, .gif or .tif, or name the format.")
    fmt = _FORMATS[ext]
    Image = _pil()
    append = "writemode" in opts and text(opts["writemode"]).lower() == "append"
    if "writemode" in opts and text(opts["writemode"]).lower() not in ("append", "overwrite"):
        raise MatlabError("imwrite: 'WriteMode' must be 'overwrite' or 'append'.")
    save: dict = {}
    try:
        if fmt == "GIF":
            _write_gif(interp, a, cmap, fname, opts, append)
            return None
        if a.ndim == 4:
            raise MatlabError("imwrite: only GIF files can hold several frames given as a 4-D array.")
        if a.dtype == np.uint16 and _is_rgb(a):
            interp.warn("imwrite: 16-bit RGB data is written with 8 bits per channel.")
        im = _pil_from_array(a, cmap, fmt, "imwrite")
        if fmt == "JPEG":
            q = _scalar(opts["quality"], "Quality") if "quality" in opts else 75
            if not 0 <= q <= 100:
                raise MatlabError("imwrite: 'Quality' must be between 0 and 100.")
            save["quality"] = int(q)
            if q == 100:
                save["subsampling"] = 0
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB" if im.mode == "P" else "L")
        if fmt == "PNG" and "alpha" in opts:
            al = _from_unit(_to_unit(_img(opts["alpha"], "imwrite", "alpha")), np.uint8)
            if al.shape != a.shape[:2]:
                raise MatlabError("imwrite: 'Alpha' must have the same number of rows and columns as the image.")
            im = im.convert("RGBA" if im.mode in ("RGB", "P") else "LA")
            im.putalpha(Image.fromarray(al))
        if fmt == "TIFF":
            comp = text(opts["compression"]).lower() if "compression" in opts else "none"
            mapping = {"none": None, "lzw": "tiff_lzw", "deflate": "tiff_adobe_deflate", "packbits": "packbits",
                       "jpeg": "jpeg"}
            if mapping.get(comp):
                save["compression"] = mapping[comp]
            if append and os.path.isfile(fname):
                old = Image.open(fname)
                pages = []
                for k in range(getattr(old, "n_frames", 1)):
                    old.seek(k)
                    pages.append(old.copy())
                old.close()
                pages[0].save(fname, format="TIFF", save_all=True, append_images=pages[1:] + [im], **save)
                return None
        if fmt in ("BMP", "PPM") and im.mode == "I;16":
            im = Image.fromarray(_from_unit(_to_unit(a), np.uint8))
        im.save(fname, format=fmt, **save)
    except MatlabError:
        raise
    except (OSError, ValueError, KeyError) as e:
        raise MatlabError(f"imwrite: cannot write '{fname}': {e}.")
    return None


def _write_gif(interp, a: np.ndarray, cmap, fname: str, opts: dict, append: bool) -> None:
    Image = _pil()
    delay = _scalar(opts["delaytime"], "DelayTime") if "delaytime" in opts else 0.5
    duration = max(int(round(delay * 1000)), 0)
    if a.ndim == 4:
        if a.shape[2] not in (1, 3):
            raise MatlabError("imwrite: a 4-D array for a GIF must be M-by-N-by-1-by-frames.")
        new = [_pil_from_array(a[..., k], cmap, "GIF", "imwrite") for k in range(a.shape[3])]
    else:
        new = [_pil_from_array(a, cmap, "GIF", "imwrite")]
    new = [im.convert("L") if im.mode == "1" else im for im in new]
    new = [im.quantize(256, dither=Image.Dither.NONE) if im.mode == "RGB" else im for im in new]
    key = os.path.abspath(fname)
    store = _gif_frames(interp)
    record = None
    if append:
        record = store.get(key)
        stamp = (os.path.getmtime(key), os.path.getsize(key)) if os.path.isfile(key) else None
        if record is None or record["stamp"] != stamp:
            if stamp is None:
                raise MatlabError(f"imwrite: cannot append to '{fname}' because the file does not exist. "
                                  "Write the first frame without 'WriteMode', 'append'.")
            old = Image.open(key)  # a file from elsewhere: load its frames
            frames, durations = [], []
            for k in range(getattr(old, "n_frames", 1)):
                old.seek(k)
                frames.append(old.copy())
                durations.append(int(old.info.get("duration", 100)))
            record = {"frames": frames, "durations": durations, "loop": old.info.get("loop")}
            old.close()
    if record is None:
        record = {"frames": [], "durations": [], "loop": None}
    if "loopcount" in opts:
        count = _scalar(opts["loopcount"], "LoopCount")
        # 0 in the file means repeat forever
        record["loop"] = 0 if np.isinf(count) else int(min(max(count, 0), 65535))
    record["frames"].extend(new)
    record["durations"].extend([duration] * len(new))
    frames = record["frames"]
    kw: dict = {"format": "GIF"}
    if len(frames) > 1:
        kw.update(save_all=True, append_images=frames[1:], duration=record["durations"], disposal=1,
                  optimize=False)
    elif "delaytime" in opts:
        kw["duration"] = record["durations"][0]
    if record["loop"] is not None:
        kw["loop"] = record["loop"]
    frames[0].save(key, **kw)
    record["stamp"] = (os.path.getmtime(key), os.path.getsize(key))
    store[key] = record


@builtin("imfinfo")
def _imfinfo(interp, args, nargout):
    nargs(args, 1, 2, "imfinfo")
    source = text(args[0], "file name")
    fmt = text(args[1]).lower() if len(args) > 1 else None
    im, path = _open_image(interp, source, fmt, "imfinfo")
    try:
        import datetime
        total = getattr(im, "n_frames", 1)
        is_file = os.path.isfile(path)
        size = float(os.path.getsize(path)) if is_file else 0.0
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%b-%Y %H:%M:%S") if is_file else ""
        kind = (im.format or "").lower()
        kind = {"jpeg": "jpg", "tiff": "tif", "ppm": "ppm"}.get(kind, kind)
        fields = ["Filename", "FileModDate", "FileSize", "Format", "Width", "Height", "BitDepth", "ColorType",
                  "NumberOfFrames", "DelayTime", "Colormap"]
        data = np.empty((total, 1), dtype=object)
        for k in range(total):
            if total > 1:
                im.seek(k)
            mode = im.mode
            depth = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "LA": 16, "CMYK": 32, "F": 32, "I": 32,
                     "PA": 16}.get(mode, 16 if mode.startswith("I;16") else 8)
            colour = {"P": "indexed", "PA": "indexed", "RGB": "truecolor", "RGBA": "truecolor",
                      "CMYK": "CMYK"}.get(mode, "grayscale")
            data[k, 0] = {
                "Filename": char(os.path.abspath(path) if is_file else path), "FileModDate": char(stamp),
                "FileSize": size, "Format": char(kind), "Width": float(im.width), "Height": float(im.height),
                "BitDepth": float(depth), "ColorType": char(colour), "NumberOfFrames": float(total),
                "DelayTime": float(im.info.get("duration", 0)) / 10.0,
                "Colormap": _palette(im) if mode in ("P", "PA") else np.zeros((0, 0)),
            }
            for name in fields:
                v = data[k, 0][name]
                if isinstance(v, float):
                    data[k, 0][name] = np.array([[v]])
    finally:
        im.close()
    return Struct(data, fields)


# -- movie frames (so that animation loops can save what is on screen) -------------------------------

def _figure_pixels(interp, target=None) -> np.ndarray:
    from . import plotting
    fig = None
    obj = getattr(target, "obj", None)
    if obj is not None:
        fig = getattr(obj, "figure", None) or obj
        if not hasattr(fig, "canvas"):
            fig = None
    if fig is None:
        fig = plotting._figure(interp)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = fig.canvas if hasattr(fig.canvas, "buffer_rgba") else FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()
    except Exception as e:
        raise MatlabError(f"getframe: could not capture the figure ({e}).")


def _getframe(interp, args, nargout):
    nargs(args, 0, 2, "getframe")
    pixels = _figure_pixels(interp, args[0] if args else None)
    return Struct.scalar({"cdata": pixels, "colormap": np.zeros((0, 0))})


def _frame2im(interp, args, nargout):
    nargs(args, 1, 1, "frame2im")
    f = args[0]
    if not isinstance(f, Struct) or "cdata" not in f.fields:
        raise MatlabError("frame2im: the input must be a movie frame (a struct with a 'cdata' field).")
    el = f.data.flat[0]
    cmap = el.get("colormap", np.zeros((0, 0)))
    return (el["cdata"], cmap) if nargout > 1 else el["cdata"]


def _im2frame(interp, args, nargout):
    nargs(args, 1, 2, "im2frame")
    a = _img(args[0], "im2frame")
    cmap = args[1] if len(args) > 1 else np.zeros((0, 0))
    return Struct.scalar({"cdata": a, "colormap": cmap})


# Registered only when the plotting module does not already provide them.
for _name, _fn in (("getframe", _getframe), ("frame2im", _frame2im), ("im2frame", _im2frame)):
    if _name not in BUILTINS:
        builtin(_name)(_fn)


