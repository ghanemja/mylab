"""Shared plumbing for the computer vision functions (vision_*.py and pointcloud.py).

OpenCV is optional: :func:`cv2mod` imports it on first use and raises a MatlabError that names the
install command when it is missing. Coordinates follow the MATLAB convention throughout the public
interface: pixel centres sit at integer positions starting at 1, points are M-by-2 ``[x y]`` rows and
boxes are ``[x y w h]`` with (x, y) the top-left corner of the top-left pixel (so a box that covers
pixel column 1 starts at x = 0.5 in spatial terms, but MATLAB writes integer boxes as starting at 1;
we follow the integer convention: a box [x y w h] covers pixel columns x .. x+w-1). OpenCV uses
0-based pixel centres, so a point moves by exactly 1 in each direction between the two.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, char, class_name, is_char, to_matlab, to_str
from ._util import is_text, match_option, text, unknown_option_error

INSTALL_HINT = 'pip install "mylab-lang[vision]"  (or: pip install "opencv-python-headless<5")'

_cv2 = None


def cv2mod(fname: str):
    """The cv2 module, or a MatlabError that says how to install it."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            raise MatlabError(f"{fname} needs OpenCV, which is not installed. Install it with: {INSTALL_HINT}",
                              "mylab:vision:noOpenCV") from None
        _cv2 = cv2
    return _cv2


def has_cv2() -> bool:
    try:
        cv2mod("vision")
        return True
    except MatlabError:
        return False


def feature2d(fname: str, kind: str, **kw):
    """Create an OpenCV feature detector that moved between modules across OpenCV versions.

    OpenCV 4 has ``cv2.BRISK_create`` etc. in the main module; OpenCV 5 builds with the extra modules
    keep them in ``cv2.xfeatures2d``.
    """
    cv2 = cv2mod(fname)
    maker = getattr(cv2, f"{kind}_create", None)
    if maker is None and hasattr(cv2, "xfeatures2d"):
        maker = getattr(cv2.xfeatures2d, f"{kind}_create", None)
    if maker is None:
        raise MatlabError(f"{fname}: this OpenCV build ({cv2.__version__}) has no {kind} detector. "
                          f"Install a build that includes it, e.g. opencv-python 4.x.", "mylab:vision:noDetector")
    return maker(**kw)


# -- images ---------------------------------------------------------------------------------------------

def image_arg(v: Any, fname: str, what: str = "image") -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf" or v.size == 0:
        raise MatlabError(f"{fname}: the {what} must be a non-empty numeric or logical array.",
                          f"mylab:{fname}:invalidImage")
    return v


def to_unit(a: np.ndarray) -> np.ndarray:
    """Double image in 0..1 (the im2double rule)."""
    if a.dtype == bool:
        return a.astype(float)
    if a.dtype.kind == "f":
        return a.astype(float)
    info = np.iinfo(a.dtype)
    return (a.astype(float) - info.min) / (float(info.max) - info.min)


def gray_float(v: Any, fname: str, what: str = "image") -> np.ndarray:
    """A 2-D grayscale image as float64 in 0..1 (RGB is reduced with the usual luminance weights)."""
    a = image_arg(v, fname, what)
    u = to_unit(a)
    if u.ndim == 3 and u.shape[2] == 3:
        u = u @ np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    elif u.ndim != 2:
        raise MatlabError(f"{fname}: the {what} must be a 2-D grayscale image or an M-by-N-by-3 RGB image.",
                          f"mylab:{fname}:invalidImage")
    return np.nan_to_num(u)


def gray_u8(v: Any, fname: str, what: str = "image") -> np.ndarray:
    """A 2-D grayscale image as uint8 (OpenCV's preferred input for detectors)."""
    a = image_arg(v, fname, what)
    if a.dtype == np.uint8 and a.ndim == 2:
        return np.ascontiguousarray(a)
    g = gray_float(a, fname, what)
    return np.ascontiguousarray(np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8))


def rgb_u8(v: Any, fname: str) -> np.ndarray:
    """An H-by-W-by-3 uint8 RGB copy of an image (gray and logical images are replicated)."""
    a = image_arg(v, fname)
    if a.dtype == np.uint8:
        out = a
    else:
        out = np.clip(np.round(np.nan_to_num(to_unit(a)) * 255.0), 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    if out.ndim != 3 or out.shape[2] != 3:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image or an M-by-N-by-3 RGB image.")
    return np.ascontiguousarray(out)


def like_input(out_u8: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB result into the class of the input image (insertShape and friends)."""
    if a.dtype == np.uint8:
        return out_u8
    u = out_u8.astype(float) / 255.0
    if a.dtype.kind == "f":
        return u.astype(a.dtype)
    if a.dtype == bool:
        return u
    info = np.iinfo(a.dtype)
    return np.round(u * (float(info.max) - info.min) + info.min).astype(a.dtype)


# -- numbers --------------------------------------------------------------------------------------------

def fmat(v: Any, fname: str, what: str, cols: int | None = None, allow_empty: bool = True) -> np.ndarray:
    """A real float matrix, optionally with a fixed number of columns."""
    if hasattr(v, "point_xy"):  # a points object stands for its Location
        v = v.point_xy()
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf":
        raise MatlabError(f"{fname}: the {what} must be numeric.", f"mylab:{fname}:invalidInput")
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        if not allow_empty:
            raise MatlabError(f"{fname}: the {what} must not be empty.", f"mylab:{fname}:invalidInput")
        return np.zeros((0, cols or 0))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if cols is not None:
        if a.ndim != 2 or a.shape[1] != cols:
            if a.size == cols:
                return a.reshape(1, cols)
            raise MatlabError(f"{fname}: the {what} must be an M-by-{cols} matrix.", f"mylab:{fname}:invalidInput")
    return a


def fscalar(v: Any, fname: str, what: str) -> float:
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size == 1:
        return float(v.flat[0])
    raise MatlabError(f"{fname}: {what} must be a numeric scalar.", f"mylab:{fname}:invalidInput")


def fvec(v: Any, fname: str, what: str, n: int | None = None) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf":
        raise MatlabError(f"{fname}: {what} must be numeric.", f"mylab:{fname}:invalidInput")
    a = v.astype(float).flatten(order="F")
    if n is not None and a.size != n:
        raise MatlabError(f"{fname}: {what} must have {n} elements.", f"mylab:{fname}:invalidInput")
    return a


def flag(v: Any) -> bool:
    if is_text(v):
        return text(v).lower() in ("true", "on", "yes")
    return bool(np.ravel(np.asarray(v))[0]) if np.size(v) else False


def col(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1, 1)


def row(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(1, -1)


def scalar(x) -> np.ndarray:
    return np.array([[float(x)]])


def text_list(v: Any) -> list[str]:
    if isinstance(v, StringArray):
        return [str(s) for s in v.values_f()]
    if isinstance(v, Cell):
        return [to_str(x) for x in v.data.flatten(order="F")]
    if isinstance(v, MString) or is_char(v):
        return [to_str(v)]
    raise MatlabError("Expected text, a string array or a cell array of character vectors.")


def cellstr_col(items: list[str]) -> Cell:
    data = np.empty((len(items), 1), dtype=object)
    for i, s in enumerate(items):
        data[i, 0] = char(s)
    return Cell(data)


# -- options --------------------------------------------------------------------------------------------

def options(args: list, known, fname: str, npos_max: int | None = None, lead: int = 0) -> tuple[list, dict]:
    """Positional arguments, then Name,Value pairs. Names match case-insensitively by unique prefix.

    The options start at the first text argument; positional text arguments are therefore not
    supported, which suits the functions here (their text inputs are all option names or values),
    except for the first ``lead`` arguments, which are always positional.
    """
    i = min(lead, len(args))
    while i < len(args) and not is_text(args[i]):
        i += 1
    pos = list(args[:i])
    if npos_max is not None and len(pos) > npos_max:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    opts: dict = {}
    k = i
    while k < len(args):
        a = args[k]
        if not is_text(a):
            raise MatlabError(f"{fname}: expected an option name, but found a value of class '{class_name(a)}'.",
                              f"mylab:{fname}:InvalidOptionName")
        key = match_option(text(a), known)
        if key is None:
            raise unknown_option_error(fname, text(a), known, None)
        if k + 1 >= len(args):
            raise MatlabError(f"{fname}: the option '{text(a)}' needs a value.", f"mylab:{fname}:MissingValue")
        opts[key] = args[k + 1]
        k += 2
    return pos, opts


def opt_text(opts: dict, key: str, default: str, choices: tuple[str, ...], fname: str) -> str:
    if key not in opts:
        return default
    s = text(opts[key])
    hit = match_option(s, choices)
    if hit is None:
        raise MatlabError(f"{fname}: '{s}' is not a valid value for {key}; use "
                          + ", ".join(f"'{c}'" for c in choices) + ".", f"mylab:{fname}:invalidOption")
    return hit


# -- a property-bag object ------------------------------------------------------------------------------

class VisionObject(MObject):
    """Base for the vision objects: named properties, a method table and a property-list display.

    Subclasses set ``CLASS``, ``PROPS`` (display order), ``SETTABLE`` and register methods with
    :func:`method`. Property values live in ``self.p`` as MATLAB values (or plain Python numbers);
    a subclass may compute some properties in :meth:`prop`. Value objects copy themselves on
    assignment; handle objects (``is_handle = True``) change in place.
    """

    CLASS = "object"
    PROPS: tuple = ()
    SETTABLE: frozenset = frozenset()
    METHODS: dict = {}

    def __init__(self, **props):
        self.p: dict = dict(props)

    def class_name(self) -> str:
        return self.CLASS

    def isa_class(self, name: str) -> bool:
        return name == self.CLASS or (self.is_handle and name == "handle")

    def property_names(self) -> list[str]:
        return list(self.PROPS)

    def method_names(self) -> list[str]:
        return sorted(self.METHODS, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in self.METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = self.METHODS.get(name)
        if fn is None:
            return super().call_method(interp, name, args, nargout)
        me = next((i for i, a in enumerate(args) if a is self), 0)
        rest = [a for i, a in enumerate(args) if i != me]
        res = fn(self, interp, rest, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, (tuple, list)) else [to_matlab(res)]

    def prop(self, name: str) -> Any:
        return self.p[name]

    def get_field(self, interp, name: str) -> Any:
        if name in self.PROPS:
            return to_matlab(self.prop(name))
        if name in self.METHODS:
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.CLASS}'.",
                          "MATLAB:noSuchMethodOrField")

    def check_set(self, name: str, value: Any) -> Any:
        return value

    def set_field(self, interp, name: str, value: Any) -> "VisionObject":
        if name not in self.SETTABLE:
            if name in self.PROPS:
                raise MatlabError(f"Unable to set the '{name}' property of class '{self.CLASS}' because it is "
                                  "read-only.", "MATLAB:class:SetProhibited")
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.CLASS}'.",
                              "MATLAB:noPublicFieldForClass")
        obj = self if self.is_handle else self.clone()
        obj.p[name] = obj.check_set(name, value)
        obj.changed(name)
        return obj

    def changed(self, name: str) -> None:
        """Hook run after a property is set."""

    def clone(self) -> "VisionObject":
        obj = copy.copy(self)
        obj.p = dict(self.p)
        return obj

    def apply_options(self, opts: dict) -> None:
        for k, v in opts.items():
            self.p[k] = self.check_set(k, v)

    def display_lines(self) -> list[str]:
        from ..display import summary
        names = self.property_names()
        width = max((len(n) for n in names), default=0)
        lines = [f"  {self.CLASS} with properties:", ""]
        for n in names:
            v = to_matlab(self.prop(n))
            lines.append(f"    {n.rjust(width)}: {summary(v, False)}")
        return lines


def method(cls: type, name: str | None = None) -> Callable:
    """Decorator: register ``fn(self, interp, args, nargout)`` as method ``name`` of ``cls``."""
    def deco(fn):
        if "METHODS" not in cls.__dict__:
            cls.METHODS = dict(cls.METHODS)
        cls.METHODS[name or fn.__name__.lstrip("_")] = fn
        return fn
    return deco


def known_props(cls: type) -> tuple[str, ...]:
    return tuple(cls.SETTABLE)


def textval(v: Any) -> Any:
    """Text as a char row; other values unchanged."""
    if isinstance(v, MString):
        return char(str(v))
    return v
