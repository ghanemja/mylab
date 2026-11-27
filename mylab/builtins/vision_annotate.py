"""Drawing into images: insertShape, insertMarker, insertText and insertObjectAnnotation.

The result is always an RGB image of the input's class. Positions use MATLAB pixel coordinates
(pixel centres at 1, 2, ...); boxes are [x y w h]. Drawing goes through OpenCV with sub-pixel
coordinates, so a line from (1, 1) to (10, 1) covers the first ten pixels of the top row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, is_char, to_str
from ._cv import cv2mod, flag, fmat, fscalar, like_input, options, rgb_u8, text_list
from ._registry import builtin
from ._util import is_text, text

_COLORS = {
    "blue": (0, 0, 255), "green": (0, 255, 0), "red": (255, 0, 0), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "yellow": (255, 255, 0), "black": (0, 0, 0), "white": (255, 255, 255),
    "b": (0, 0, 255), "g": (0, 255, 0), "r": (255, 0, 0), "c": (0, 255, 255), "m": (255, 0, 255),
    "y": (255, 255, 0), "k": (0, 0, 0), "w": (255, 255, 255),
}
_SHIFT = 4  # OpenCV fixed-point bits for sub-pixel drawing
_ONE = 1 << _SHIFT


def colors(v: Any, n: int, fname: str, img_dtype) -> list[tuple[int, int, int]]:
    """n RGB colours (0..255) from a name, a list of names, an RGB triplet or an n-by-3 matrix."""
    if isinstance(v, (MString, StringArray, Cell)) or is_char(v):
        names = text_list(v)
        out = []
        for s in names:
            c = _COLORS.get(s.lower())
            if c is None:
                raise MatlabError(f"{fname}: unknown colour name '{s}'. Use blue, green, red, cyan, magenta, "
                                  "yellow, black or white, or an RGB triplet.", f"mylab:{fname}:invalidColor")
            out.append(c)
    else:
        a = fmat(v, fname, "colour", 3)
        if img_dtype.kind in "fb":  # colours of double, single and logical images are in 0..1
            a = a * 255.0
        elif img_dtype.kind in "iu" and img_dtype != np.uint8:
            info = np.iinfo(img_dtype)
            a = (a - info.min) / (float(info.max) - info.min) * 255.0
        out = [tuple(int(np.clip(round(x), 0, 255)) for x in r) for r in a]
    if len(out) == 1:
        return out * n
    if len(out) != n:
        raise MatlabError(f"{fname}: give one colour, or one colour per shape ({n}).", f"mylab:{fname}:invalidColor")
    return out


def _fixed(x: float, y: float) -> tuple[int, int]:
    return int(round((x - 1) * _ONE)), int(round((y - 1) * _ONE))


def _blend(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    m = mask.astype(bool)
    out = base.copy()
    out[m] = np.round(base[m] * (1 - opacity) + layer[m] * opacity).astype(np.uint8)
    return out


_SHAPES = ("rectangle", "filled-rectangle", "line", "polygon", "filled-polygon", "circle", "filled-circle",
           "filledrectangle", "filledpolygon", "filledcircle", "projected-cuboid", "projectedcuboid")


@builtin("insertShape")
def _insert_shape(interp, args, nargout):
    fname = "insertShape"
    if len(args) < 3:
        raise MatlabError("insertShape: expected insertShape(I, shape, position, ...).", "MATLAB:minrhs")
    img = args[0]
    if not is_text(args[1]):
        raise MatlabError("insertShape: the shape must be text, e.g. 'rectangle' or 'filled-circle'.")
    shape = text(args[1]).lower().replace("_", "-")
    if shape not in _SHAPES:
        raise MatlabError(f"insertShape: unknown shape '{text(args[1])}'.", "mylab:insertShape:invalidShape")
    shape = {"filledrectangle": "filled-rectangle", "filledpolygon": "filled-polygon",
             "filledcircle": "filled-circle", "projectedcuboid": "projected-cuboid"}.get(shape, shape)
    _, opts = options(list(args[3:]), ("Color", "ShapeColor", "LineWidth", "Opacity", "SmoothEdges"), fname, 0)
    rgb = rgb_u8(img, fname)
    pos = args[2]
    if isinstance(pos, Cell):
        items = [np.asarray(p, dtype=float).ravel() for p in pos.data.flatten(order="F")]
    else:
        P = fmat(pos, fname, "position")
        items = [P[i] for i in range(P.shape[0])]
    cv2 = cv2mod(fname)
    cv = opts.get("Color", opts.get("ShapeColor"))
    cols = colors(cv, len(items), fname, img.dtype) if cv is not None else [(255, 255, 0)] * len(items)
    lw = int(round(fscalar(opts["LineWidth"], fname, "LineWidth"))) if "LineWidth" in opts else 1
    opacity = fscalar(opts["Opacity"], fname, "Opacity") if "Opacity" in opts else 0.6
    smooth = flag(opts["SmoothEdges"]) if "SmoothEdges" in opts else True
    line_type = cv2.LINE_AA if smooth else cv2.LINE_8
    filled = shape.startswith("filled")
    out = rgb.copy()
    layer = rgb.copy()
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for it, c in zip(items, cols):
        target = out
        color = tuple(int(v) for v in c)
        if shape in ("rectangle", "filled-rectangle"):
            if it.size != 4:
                raise MatlabError("insertShape: a rectangle position is [x y width height].")
            x, y, w, h = it
            p1 = _fixed(x, y)
            p2 = _fixed(x + w - 1, y + h - 1)
            if filled:
                cv2.rectangle(layer, p1, p2, color, -1, line_type, _SHIFT)
                cv2.rectangle(mask, p1, p2, 255, -1, line_type, _SHIFT)
            else:
                cv2.rectangle(target, p1, p2, color, lw, line_type, _SHIFT)
        elif shape in ("circle", "filled-circle"):
            if it.size != 3:
                raise MatlabError("insertShape: a circle position is [x y radius].")
            cen = _fixed(it[0], it[1])
            r = int(round(it[2] * _ONE))
            if filled:
                cv2.circle(layer, cen, r, color, -1, line_type, _SHIFT)
                cv2.circle(mask, cen, r, 255, -1, line_type, _SHIFT)
            else:
                cv2.circle(target, cen, r, color, lw, line_type, _SHIFT)
        elif shape == "projected-cuboid":
            raise MatlabError("insertShape: 'projected-cuboid' is not supported.")
        else:
            v = it[~np.isnan(it)]
            if v.size < 4 or v.size % 2:
                raise MatlabError("insertShape: a line or polygon position is [x1 y1 x2 y2 ...].")
            pts = np.array([_fixed(v[k], v[k + 1]) for k in range(0, v.size, 2)], dtype=np.int32)
            if shape == "line":
                cv2.polylines(out, [pts], False, color, lw, line_type, _SHIFT)
            elif shape == "polygon":
                cv2.polylines(out, [pts], True, color, lw, line_type, _SHIFT)
            else:
                cv2.fillPoly(layer, [pts], color, line_type, _SHIFT)
                cv2.fillPoly(mask, [pts], 255, line_type, _SHIFT)
    if filled:
        alpha = mask.astype(float) / 255.0 * opacity
        out = np.round(rgb * (1 - alpha[:, :, None]) + layer * alpha[:, :, None]).astype(np.uint8)
    return like_input(out, img)


_MARKERS = {"plus": "+", "+": "+", "circle": "o", "o": "o", "star": "*", "*": "*", "x-mark": "x", "x": "x",
            "square": "s", "s": "s"}


@builtin("insertMarker")
def _insert_marker(interp, args, nargout):
    fname = "insertMarker"
    if len(args) < 2:
        raise MatlabError("insertMarker: expected insertMarker(I, position, ...).", "MATLAB:minrhs")
    img = args[0]
    rest = list(args[2:])
    marker = "+"
    if rest and is_text(rest[0]) and text(rest[0]).lower() in _MARKERS:
        marker = _MARKERS[text(rest[0]).lower()]
        rest = rest[1:]
    _, opts = options(rest, ("Color", "Size"), fname, 0)
    P = fmat(args[1], fname, "position", 2)
    cv2 = cv2mod(fname)
    out = rgb_u8(img, fname)
    cols = colors(opts["Color"], P.shape[0], fname, img.dtype) if "Color" in opts else [(0, 255, 0)] * P.shape[0]
    s = fscalar(opts["Size"], fname, "Size") if "Size" in opts else 3.0
    for (x, y), c in zip(P, cols):
        c = tuple(int(v) for v in c)
        segs = []
        if marker in ("+", "*"):
            segs += [((x - s, y), (x + s, y)), ((x, y - s), (x, y + s))]
        if marker in ("x", "*"):
            segs += [((x - s, y - s), (x + s, y + s)), ((x - s, y + s), (x + s, y - s))]
        for a, b in segs:
            cv2.line(out, _fixed(*a), _fixed(*b), c, 1, cv2.LINE_8, _SHIFT)
        if marker == "o":
            cv2.circle(out, _fixed(x, y), int(round(s * _ONE)), c, 1, cv2.LINE_8, _SHIFT)
        elif marker == "s":
            cv2.rectangle(out, _fixed(x - s, y - s), _fixed(x + s, y + s), c, 1, cv2.LINE_8, _SHIFT)
    return like_input(out, img)


_ANCHORS = ("LeftTop", "LeftCenter", "LeftBottom", "CenterTop", "Center", "CenterBottom", "RightTop",
            "RightCenter", "RightBottom")


def _labels(v: Any, n: int, fname: str) -> list[str]:
    if isinstance(v, np.ndarray) and not is_char(v):
        vals = np.asarray(v).ravel(order="F")
        out = [f"{x:g}" if np.issubdtype(vals.dtype, np.number) else str(x) for x in vals]
    elif isinstance(v, Cell):
        out = [to_str(x) if (is_char(x) or isinstance(x, MString)) else f"{float(np.ravel(x)[0]):g}"
               for x in v.data.flatten(order="F")]
    else:
        out = text_list(v)
    if len(out) == 1 and n > 1:
        out = out * n
    if len(out) != n:
        raise MatlabError(f"{fname}: give one label, or one label per position ({n}).",
                          f"mylab:{fname}:labelCount")
    return out


def draw_text(cv2, out: np.ndarray, x: float, y: float, label: str, font_size: float, text_color, box_color,
              box_opacity: float, anchor: str) -> None:
    scale = font_size / 22.0
    thick = max(1, int(round(font_size / 12)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = label.split("\n")
    sizes = [cv2.getTextSize(s or " ", font, scale, thick) for s in lines]
    lh = max(sz[0][1] + sz[1] for sz in sizes) + 2
    tw = max(sz[0][0] for sz in sizes) + 4
    th = lh * len(lines) + 2
    x0, y0 = x - 1, y - 1  # 0-based pixel position of the anchor
    if anchor.startswith("Center"):
        x0 -= tw / 2
    elif anchor.startswith("Right"):
        x0 -= tw
    if anchor.endswith("Center") or anchor == "Center":
        y0 -= th / 2
    elif anchor.endswith("Bottom"):
        y0 -= th
    x0, y0 = int(round(x0)), int(round(y0))
    H, W = out.shape[:2]
    if box_opacity > 0 and box_color is not None:
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + tw, W), min(y0 + th, H)
        if xb > xa and yb > ya:
            region = out[ya:yb, xa:xb].astype(float)
            out[ya:yb, xa:xb] = np.round(region * (1 - box_opacity) + np.array(box_color) * box_opacity)
    for k, s in enumerate(lines):
        base = y0 + (k + 1) * lh - sizes[k][1] + 1
        cv2.putText(out, s, (x0 + 2, base), font, scale, tuple(int(c) for c in text_color), thick, cv2.LINE_AA)


@builtin("insertText")
def _insert_text(interp, args, nargout):
    fname = "insertText"
    if len(args) < 3:
        raise MatlabError("insertText: expected insertText(I, position, text, ...).", "MATLAB:minrhs")
    img = args[0]
    _, opts = options(list(args[3:]), ("FontSize", "Font", "TextColor", "TextBoxColor", "BoxColor", "BoxOpacity",
                                       "TextBoxOpacity", "AnchorPoint", "FontColor"), fname, 0)
    P = fmat(args[1], fname, "position", 2)
    labels = _labels(args[2], P.shape[0], fname)
    cv2 = cv2mod(fname)
    out = rgb_u8(img, fname)
    fs = fscalar(opts["FontSize"], fname, "FontSize") if "FontSize" in opts else 12.0
    tc_v = opts.get("TextColor", opts.get("FontColor"))
    tcs = colors(tc_v, P.shape[0], fname, img.dtype) if tc_v is not None else [(0, 0, 0)] * P.shape[0]
    bc_v = opts.get("BoxColor", opts.get("TextBoxColor"))
    bcs = colors(bc_v, P.shape[0], fname, img.dtype) if bc_v is not None else [(255, 255, 0)] * P.shape[0]
    op_v = opts.get("BoxOpacity", opts.get("TextBoxOpacity"))
    op = fscalar(op_v, fname, "BoxOpacity") if op_v is not None else 0.6
    anchor = "LeftTop"
    if "AnchorPoint" in opts:
        from ._util import match_option
        anchor = match_option(text(opts["AnchorPoint"]), _ANCHORS) or "LeftTop"
    for (x, y), s, tc, bc in zip(P, labels, tcs, bcs):
        draw_text(cv2, out, x, y, s, fs, tc, bc, op, anchor)
    return like_input(out, img)


@builtin("insertObjectAnnotation")
def _insert_object_annotation(interp, args, nargout):
    fname = "insertObjectAnnotation"
    if len(args) < 4:
        raise MatlabError("insertObjectAnnotation: expected insertObjectAnnotation(I, shape, position, label).",
                          "MATLAB:minrhs")
    img = args[0]
    shape = text(args[1]).lower()
    if shape not in ("rectangle", "circle"):
        raise MatlabError("insertObjectAnnotation: the shape must be 'rectangle' or 'circle'.")
    _, opts = options(list(args[4:]), ("Color", "AnnotationColor", "LineWidth", "FontSize", "Font", "TextColor",
                                       "TextBoxOpacity", "FontColor", "ShapeOpacity"), fname, 0)
    P = fmat(args[2], fname, "position")
    n = P.shape[0]
    labels = _labels(args[3], n, fname)
    cv2 = cv2mod(fname)
    out = rgb_u8(img, fname)
    cv = opts.get("Color", opts.get("AnnotationColor"))
    cols = colors(cv, n, fname, img.dtype) if cv is not None else [(255, 255, 0)] * n
    lw = int(round(fscalar(opts["LineWidth"], fname, "LineWidth"))) if "LineWidth" in opts else 1
    fs = fscalar(opts["FontSize"], fname, "FontSize") if "FontSize" in opts else 12.0
    tv = opts.get("TextColor", opts.get("FontColor"))
    tcs = colors(tv, n, fname, img.dtype) if tv is not None else [(0, 0, 0)] * n
    op = fscalar(opts["TextBoxOpacity"], fname, "TextBoxOpacity") if "TextBoxOpacity" in opts else 0.6
    for p, s, c, tc in zip(P, labels, cols, tcs):
        c = tuple(int(v) for v in c)
        if shape == "rectangle":
            if p.size != 4:
                raise MatlabError("insertObjectAnnotation: rectangle positions are [x y width height].")
            x, y, w, h = p
            cv2.rectangle(out, _fixed(x, y), _fixed(x + w - 1, y + h - 1), c, lw, cv2.LINE_8, _SHIFT)
            draw_text(cv2, out, x, y, s, fs, tc, c, op, "LeftBottom")
        else:
            if p.size != 3:
                raise MatlabError("insertObjectAnnotation: circle positions are [x y radius].")
            cv2.circle(out, _fixed(p[0], p[1]), int(round(p[2] * _ONE)), c, lw, cv2.LINE_8, _SHIFT)
            draw_text(cv2, out, p[0] - p[2], p[1] - p[2], s, fs, tc, c, op, "LeftBottom")
    return like_input(out, img)
