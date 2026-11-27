"""Object detection: Viola-Jones cascades, HOG people detection, ONNX YOLO-style detectors run through
OpenCV's dnn module, non-maximum suppression, box overlap and detection metrics.

No model weights are bundled with mylab. The cascade models are the BSD-licensed XML files that
opencv-python 4.x ships in ``cv2.data.haarcascades`` (OpenCV 5 wheels no longer include them; set
MYLAB_HAARCASCADES to a folder that has them, or pass the path of an XML file). The people detector
uses the linear HOG SVM compiled into OpenCV. YOLO detectors need an ONNX file supplied by the user.
Boxes are [x y w h] with (x, y) the top-left pixel, 1-based.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, char
from ._cv import (
    INSTALL_HINT, VisionObject, cellstr_col, col, cv2mod, flag, fmat, fscalar, fvec, gray_u8, image_arg, method,
    opt_text, options, rgb_u8, scalar, text_list,
)
from ._registry import builtin
from ._util import is_text, text

# -- box utilities ---------------------------------------------------------------------------------------


def _boxes(v: Any, fname: str, what: str = "bounding boxes") -> np.ndarray:
    b = fmat(v, fname, what)
    if b.size == 0:
        return np.zeros((0, 4))
    if b.shape[1] != 4:
        raise MatlabError(f"{fname}: {what} must be an M-by-4 matrix of [x y width height] rows.",
                          f"mylab:{fname}:invalidBoxes")
    if np.any(b[:, 2:] < 0):
        raise MatlabError(f"{fname}: box widths and heights must not be negative.", f"mylab:{fname}:invalidBoxes")
    return b


def overlap(a: np.ndarray, b: np.ndarray, ratio: str = "union") -> np.ndarray:
    """Pairwise overlap ratios of boxes a (M) and b (N): intersection over union or over the smaller area."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    den = np.minimum(area_a, area_b) if ratio == "min" else area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, inter / den, 0.0)


@builtin("bboxOverlapRatio")
def _bbox_overlap_ratio(interp, args, nargout):
    if len(args) not in (2, 3):
        raise MatlabError("bboxOverlapRatio: expected bboxOverlapRatio(bboxA, bboxB) or (..., ratioType).",
                          "MATLAB:minrhs")
    ratio = "union"
    if len(args) == 3:
        ratio = text(args[2]).lower()
        if ratio not in ("union", "min"):
            raise MatlabError("bboxOverlapRatio: the ratio type must be 'Union' or 'Min'.")
    return overlap(_boxes(args[0], "bboxOverlapRatio", "bboxA"), _boxes(args[1], "bboxOverlapRatio", "bboxB"), ratio)


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float, ratio: str, limit: float,
        labels: np.ndarray | None = None) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices in input order."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    suppressed = np.zeros(len(scores), dtype=bool)
    ov = overlap(boxes, boxes, ratio)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        if len(keep) >= limit:
            break
        same = ov[i] > thresh
        if labels is not None:
            same &= labels == labels[i]
        suppressed |= same
    return np.sort(np.array(keep, dtype=np.int64))


def _nms_args(args, fname: str, multiclass: bool):
    npos = 3 if multiclass else 2
    pos, opts = options(list(args), ("RatioType", "OverlapThreshold", "NumStrongest"), fname, npos)
    if len(pos) != npos:
        raise MatlabError(f"{fname}: expected {fname}(bboxes, scores{', labels' if multiclass else ''}, ...).",
                          "MATLAB:minrhs")
    b = _boxes(pos[0], fname)
    s = fvec(pos[1], fname, "scores")
    if s.size != b.shape[0]:
        raise MatlabError(f"{fname}: give one score per box.", f"mylab:{fname}:sizeMismatch")
    ratio = opt_text(opts, "RatioType", "Union", ("Union", "Min"), fname).lower()
    th = fscalar(opts["OverlapThreshold"], fname, "OverlapThreshold") if "OverlapThreshold" in opts else 0.5
    lim = fscalar(opts["NumStrongest"], fname, "NumStrongest") if "NumStrongest" in opts else np.inf
    return pos, b, s, ratio, th, lim


@builtin("selectStrongestBbox")
def _select_strongest_bbox(interp, args, nargout):
    pos, b, s, ratio, th, lim = _nms_args(args, "selectStrongestBbox", False)
    k = nms(b, s, th, ratio, lim)
    sc = np.asarray(pos[1])
    return b[k], sc.reshape(-1, 1)[k] if sc.size else sc.reshape(0, 1), col(k + 1)


@builtin("selectStrongestBboxMulticlass")
def _select_strongest_bbox_multiclass(interp, args, nargout):
    pos, b, s, ratio, th, lim = _nms_args(args, "selectStrongestBboxMulticlass", True)
    lab = pos[2]
    keys = _label_keys(lab, b.shape[0], "selectStrongestBboxMulticlass")
    k = nms(b, s, th, ratio, lim, keys)
    labels_out = _index_labels(interp, lab, k)
    sc = np.asarray(pos[1]).reshape(-1, 1)
    return b[k], sc[k], labels_out, col(k + 1)


def _label_keys(lab: Any, n: int, fname: str) -> np.ndarray:
    if hasattr(lab, "codes"):
        keys = np.asarray(lab.codes).ravel()
    elif isinstance(lab, np.ndarray) and lab.dtype.kind in "biuf":
        keys = lab.ravel()
    else:
        names = text_list(lab)
        keys = np.unique(names, return_inverse=True)[1]
    if keys.size != n:
        raise MatlabError(f"{fname}: give one label per box.", f"mylab:{fname}:sizeMismatch")
    return keys


def _index_labels(interp, lab: Any, k: np.ndarray) -> Any:
    idx = (k + 1).astype(float).reshape(-1, 1)
    if hasattr(lab, "index_get"):
        return lab.index_get(interp, [idx])
    if isinstance(lab, np.ndarray):
        return lab.reshape(-1, 1)[k]
    names = text_list(lab)
    return cellstr_col([names[i] for i in k])


# -- cascade detector -------------------------------------------------------------------------------------

_CASCADE_MODELS = {
    # model name: (file in OpenCV's data folder, how it relates to the model name)
    "FrontalFaceCART": ("haarcascade_frontalface_alt2.xml", "OpenCV frontal face (Haar, tree-based stages)"),
    "FrontalFaceLBP": ("haarcascade_frontalface_default.xml", "OpenCV frontal face (Haar; no LBP face model is "
                       "bundled with opencv-python)"),
    "ProfileFace": ("haarcascade_profileface.xml", "OpenCV profile face (Haar)"),
    "UpperBody": ("haarcascade_upperbody.xml", "OpenCV upper body (Haar)"),
    "EyePairBig": ("haarcascade_eye.xml", "single eyes from OpenCV's eye cascade, paired into one box"),
    "EyePairSmall": ("haarcascade_eye.xml", "single eyes from OpenCV's eye cascade, paired into one box"),
    "LeftEye": ("haarcascade_lefteye_2splits.xml", "OpenCV left eye (Haar)"),
    "RightEye": ("haarcascade_righteye_2splits.xml", "OpenCV right eye (Haar)"),
    "LeftEyeCART": ("haarcascade_lefteye_2splits.xml", "OpenCV left eye (Haar)"),
    "RightEyeCART": ("haarcascade_righteye_2splits.xml", "OpenCV right eye (Haar)"),
}
_UNAVAILABLE = ("Mouth", "Nose")


def cascade_dirs() -> list[str]:
    dirs = []
    env = os.environ.get("MYLAB_HAARCASCADES")
    if env:
        dirs.append(env)
    try:
        import cv2  # noqa: PLC0415
        d = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if d:
            dirs.append(d)
    except ImportError:
        pass
    dirs += ["/usr/share/opencv4/haarcascades", "/usr/local/share/opencv4/haarcascades",
             "/usr/share/opencv/haarcascades", "/usr/local/share/opencv/haarcascades"]
    return dirs


def find_cascade(filename: str) -> str | None:
    for d in cascade_dirs():
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


class CascadeDetector(VisionObject):
    CLASS = "vision.CascadeObjectDetector"
    PROPS = ("ClassificationModel", "MinSize", "MaxSize", "ScaleFactor", "MergeThreshold", "UseROI",
             "TrainingSize")
    SETTABLE = frozenset(("ClassificationModel", "MinSize", "MaxSize", "ScaleFactor", "MergeThreshold", "UseROI"))
    is_handle = True

    def __init__(self, model: str):
        super().__init__(ClassificationModel=char(model), MinSize=np.zeros((0, 0)), MaxSize=np.zeros((0, 0)),
                         ScaleFactor=scalar(1.1), MergeThreshold=scalar(4), UseROI=np.array([[False]]))
        self._clf = None
        self._load(model)

    def _load(self, model: str) -> None:
        cv2 = cv2mod("vision.CascadeObjectDetector")
        self.pair = False
        if model in _UNAVAILABLE:
            raise MatlabError(f"vision.CascadeObjectDetector: no '{model}' model is available (OpenCV does not ship "
                              "one). Pass the path of a cascade XML file instead.", "mylab:vision:noModel")
        if model in _CASCADE_MODELS:
            fn = _CASCADE_MODELS[model][0]
            path = find_cascade(fn)
            if path is None:
                raise MatlabError(
                    f"vision.CascadeObjectDetector: the '{model}' model uses OpenCV's {fn}, which was not found. "
                    "opencv-python 4.x includes it in cv2.data.haarcascades (OpenCV 5 wheels do not); set the "
                    "MYLAB_HAARCASCADES environment variable to a folder that has it, or pass an XML file path.",
                    "mylab:vision:cascadeNotFound")
            self.pair = model.startswith("EyePair")
        else:
            path = model
            if not os.path.isfile(path):
                raise MatlabError(f"vision.CascadeObjectDetector: '{model}' is neither a known model name nor an "
                                  "existing cascade XML file.", "mylab:vision:cascadeNotFound")
        if not hasattr(cv2, "CascadeClassifier"):
            raise MatlabError("vision.CascadeObjectDetector: this OpenCV build "
                              f"({cv2.__version__}) has no cascade classifier. Install OpenCV 4: {INSTALL_HINT}",
                              "mylab:vision:noCascade")
        clf = cv2.CascadeClassifier()
        if not clf.load(path):
            raise MatlabError(f"vision.CascadeObjectDetector: could not read the cascade file '{path}'.",
                              "mylab:vision:badCascade")
        self._clf = clf
        self.path = path

    def prop(self, name: str) -> Any:
        if name == "TrainingSize":
            try:
                w, h = self._clf.getOriginalWindowSize()
            except Exception:  # noqa: BLE001
                w, h = 0, 0
            return np.array([[float(h), float(w)]])
        return self.p[name]

    def check_set(self, name: str, value: Any) -> Any:
        if name == "ClassificationModel":
            self._load(text(value))
            return char(text(value))
        if name in ("MinSize", "MaxSize"):
            v = np.asarray(value, dtype=float)
            if v.size not in (0, 2):
                raise MatlabError(f"vision.CascadeObjectDetector: {name} must be [height width] or [].")
            return v.reshape(1, -1) if v.size else np.zeros((0, 0))
        if name == "ScaleFactor":
            if fscalar(value, self.CLASS, name) <= 1:
                raise MatlabError("vision.CascadeObjectDetector: ScaleFactor must be greater than 1.")
            return scalar(fscalar(value, self.CLASS, name))
        if name == "MergeThreshold":
            return scalar(int(fscalar(value, self.CLASS, name)))
        if name == "UseROI":
            return np.array([[flag(value)]])
        return value

    def detect(self, args: list) -> np.ndarray:
        fname = "step"
        if not args:
            raise MatlabError("vision.CascadeObjectDetector: give an image to search.", "MATLAB:minrhs")
        g = gray_u8(args[0], fname)
        x0 = y0 = 0
        if flag(self.p["UseROI"]):
            if len(args) < 2:
                raise MatlabError("vision.CascadeObjectDetector: with UseROI true, call detector(I, roi).")
            r = fvec(args[1], fname, "ROI", 4).astype(int)
            x0, y0 = r[0] - 1, r[1] - 1
            g = np.ascontiguousarray(g[y0:y0 + r[3], x0:x0 + r[2]])
        kw = {"scaleFactor": float(self.p["ScaleFactor"].flat[0]),
              "minNeighbors": int(self.p["MergeThreshold"].flat[0])}
        if self.p["MinSize"].size:
            kw["minSize"] = (int(self.p["MinSize"].flat[1]), int(self.p["MinSize"].flat[0]))
        if self.p["MaxSize"].size:
            kw["maxSize"] = (int(self.p["MaxSize"].flat[1]), int(self.p["MaxSize"].flat[0]))
        found = self._clf.detectMultiScale(g, **kw)
        b = np.asarray(found, dtype=float).reshape(-1, 4)
        if self.pair:
            b = _pair_eyes(b)
        if b.size:
            b[:, 0] += x0 + 1
            b[:, 1] += y0 + 1
        return b


def _pair_eyes(b: np.ndarray) -> np.ndarray:
    """Join single-eye boxes of similar size and height into one box per eye pair."""
    used = np.zeros(len(b), dtype=bool)
    out = []
    order = np.argsort(b[:, 0])
    for ii, i in enumerate(order):
        if used[i]:
            continue
        for j in order[ii + 1:]:
            if used[j]:
                continue
            s = (b[i, 2] + b[j, 2]) / 2
            dy = abs((b[i, 1] + b[i, 3] / 2) - (b[j, 1] + b[j, 3] / 2))
            dx = (b[j, 0] + b[j, 2] / 2) - (b[i, 0] + b[i, 2] / 2)
            if dy < 0.5 * s and s < dx < 4 * s and 0.6 < b[i, 2] / b[j, 2] < 1.6:
                x1, y1 = min(b[i, 0], b[j, 0]), min(b[i, 1], b[j, 1])
                x2 = max(b[i, 0] + b[i, 2], b[j, 0] + b[j, 2])
                y2 = max(b[i, 1] + b[i, 3], b[j, 1] + b[j, 3])
                out.append([x1, y1, x2 - x1, y2 - y1])
                used[i] = used[j] = True
                break
    return np.array(out, dtype=float).reshape(-1, 4)


@method(CascadeDetector, "step")
def _cascade_step(self, interp, args, nargout):
    return self.detect(args)


@method(CascadeDetector, "detect")
def _cascade_detect(self, interp, args, nargout):
    return self.detect(args)


@method(CascadeDetector, "release")
def _cascade_release(self, interp, args, nargout):
    return None


def _call_object(obj, interp, args):
    """detector(I): calling a System object runs its step method."""
    return obj.call_method(interp, "step", [obj, *args], 1)[0]


def _call_object_multi(obj, interp, args, nargout):
    return obj.call_method(interp, "step", [obj, *args], max(nargout, 1))


def system_object(cls: type) -> type:
    """Make obj(args) run obj's step method, as for MATLAB System objects."""
    cls.index_get = _call_object  # type: ignore[attr-defined]
    cls.call_object = _call_object_multi  # type: ignore[attr-defined]
    return cls


system_object(CascadeDetector)


@builtin("vision.CascadeObjectDetector")
def _cascade_ctor(interp, args, nargout):
    args = list(args)
    model = "FrontalFaceCART"
    if args and is_text(args[0]) and (len(args) % 2 == 1):
        model = text(args[0])
        args = args[1:]
    _, opts = options(args, tuple(CascadeDetector.SETTABLE), "vision.CascadeObjectDetector", 0)
    if "ClassificationModel" in opts:
        model = text(opts.pop("ClassificationModel"))
    det = CascadeDetector(model)
    det.apply_options(opts)
    return det


# -- HOG people detector --------------------------------------------------------------------------------

class PeopleDetector(VisionObject):
    CLASS = "vision.PeopleDetector"
    PROPS = ("ClassificationModel", "ClassificationThreshold", "MinSize", "MaxSize", "ScaleFactor", "WindowStride",
             "MergeDetections", "UseROI")
    SETTABLE = frozenset(PROPS)
    is_handle = True

    def __init__(self, model: str = "UprightPeople_128x64", cls_name: str = "vision.PeopleDetector"):
        super().__init__(ClassificationModel=char(model), ClassificationThreshold=scalar(1.0),
                         MinSize=np.zeros((0, 0)), MaxSize=np.zeros((0, 0)), ScaleFactor=scalar(1.05),
                         WindowStride=np.array([[8.0, 8.0]]), MergeDetections=np.array([[True]]),
                         UseROI=np.array([[False]]))
        self.CLASS = cls_name
        self._make(model)

    def _make(self, model: str) -> None:
        cv2 = cv2mod(self.CLASS)
        if not hasattr(cv2, "HOGDescriptor"):
            raise MatlabError(f"{self.CLASS}: this OpenCV build ({cv2.__version__}) has no HOG people detector. "
                              f"Install OpenCV 4: {INSTALL_HINT}", "mylab:vision:noHOG")
        if model == "UprightPeople_128x64":
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif model == "UprightPeople_96x48":
            hog = cv2.HOGDescriptor((48, 96), (16, 16), (8, 8), (8, 8), 9)
            hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
        else:
            raise MatlabError(f"{self.CLASS}: the model must be 'UprightPeople_128x64' or 'UprightPeople_96x48'.",
                              "mylab:vision:invalidModel")
        self.hog = hog
        self.win = hog.winSize  # (w, h)

    def check_set(self, name: str, value: Any) -> Any:
        if name == "ClassificationModel":
            self._make(text(value))
            return char(text(value))
        if name in ("MergeDetections", "UseROI"):
            return np.array([[flag(value)]])
        return np.asarray(value, dtype=float).reshape(1, -1) if np.size(value) else np.zeros((0, 0))

    def detect(self, args: list, threshold: float | None = None):
        if not args:
            raise MatlabError(f"{self.CLASS}: give an image to search.", "MATLAB:minrhs")
        img = image_arg(args[0], self.CLASS)
        g = rgb_u8(img, self.CLASS)[:, :, ::-1].copy() if img.ndim == 3 else gray_u8(img, self.CLASS)
        x0 = y0 = 0
        if flag(self.p["UseROI"]):
            r = fvec(args[1], self.CLASS, "ROI", 4).astype(int)
            x0, y0 = r[0] - 1, r[1] - 1
            g = np.ascontiguousarray(g[y0:y0 + r[3], x0:x0 + r[2]])
        th = float(self.p["ClassificationThreshold"].flat[0]) if threshold is None else threshold
        stride = tuple(int(v) for v in self.p["WindowStride"].ravel()[:2])
        sf = float(self.p["ScaleFactor"].flat[0])
        # a smaller detection window than the image is required; very small images give no detections
        if g.shape[0] < self.win[1] or g.shape[1] < self.win[0]:
            return np.zeros((0, 4)), np.zeros((0, 1))
        rects, weights = self.hog.detectMultiScale(g, hitThreshold=th - 1.0, winStride=stride, padding=(0, 0),
                                                   scale=sf, groupThreshold=0)
        b = np.asarray(rects, dtype=float).reshape(-1, 4)
        s = np.asarray(weights, dtype=float).reshape(-1)
        keep = np.ones(len(s), dtype=bool)
        if self.p["MinSize"].size:
            keep &= (b[:, 3] >= self.p["MinSize"].flat[0]) & (b[:, 2] >= self.p["MinSize"].flat[1])
        if self.p["MaxSize"].size:
            keep &= (b[:, 3] <= self.p["MaxSize"].flat[0]) & (b[:, 2] <= self.p["MaxSize"].flat[1])
        b, s = b[keep], s[keep]
        if flag(self.p["MergeDetections"]) and len(s):
            k = nms(b, s, 0.5, "min", np.inf)
            b, s = b[k], s[k]
        order = np.argsort(-s, kind="stable")
        b, s = b[order], s[order]
        if b.size:
            b[:, 0] += x0 + 1
            b[:, 1] += y0 + 1
        return b, s.reshape(-1, 1)


@method(PeopleDetector, "step")
def _people_step(self, interp, args, nargout):
    return self.detect(args)


@method(PeopleDetector, "detect")
def _people_detect(self, interp, args, nargout):
    pos, opts = options(list(args), ("Threshold",), "detect", 2)
    th = fscalar(opts["Threshold"], "detect", "Threshold") if "Threshold" in opts else None
    return self.detect(pos, th)


@method(PeopleDetector, "release")
def _people_release(self, interp, args, nargout):
    return None


system_object(PeopleDetector)


def _people_ctor(cls_name: str):
    def fn(interp, args, nargout):
        args = list(args)
        model = "UprightPeople_128x64"
        if args and is_text(args[0]) and len(args) % 2 == 1:
            model = text(args[0])
            args = args[1:]
        _, opts = options(args, PeopleDetector.PROPS, cls_name, 0)
        if "ClassificationModel" in opts:
            model = text(opts.pop("ClassificationModel"))
        det = PeopleDetector(model, cls_name)
        det.apply_options(opts)
        return det
    builtin(cls_name)(fn)


_people_ctor("vision.PeopleDetector")
_people_ctor("peopleDetector")


# -- ONNX (YOLO-style) detectors -------------------------------------------------------------------------

_FORMATS = ("auto", "yolov5", "yolov8", "yolox")


class OnnxDetector(VisionObject):
    PROPS = ("ModelName", "ClassNames", "InputSize", "OutputFormat", "Normalize", "PreprocessMode")
    SETTABLE = frozenset(("ModelName",))

    def __init__(self, cls: str, path: str, classes: list[str] | None, input_size, fmt: str, normalize: bool,
                 mode: str):
        cv2 = cv2mod(cls)
        if not os.path.isfile(path):
            raise MatlabError(f"{cls}: the model file '{path}' was not found. mylab ships no pretrained weights; "
                              "export your detector to ONNX and pass the file name.", "mylab:vision:noModelFile")
        try:
            self.net = cv2.dnn.readNetFromONNX(path)
        except cv2.error as e:
            raise MatlabError(f"{cls}: OpenCV could not load the ONNX model '{path}': {str(e).strip()}",
                              "mylab:vision:badModel") from None
        super().__init__(ModelName=char(os.path.splitext(os.path.basename(path))[0]))
        self.CLASS = cls
        self.path = path
        self.classes = classes
        self.input_size = input_size  # (h, w)
        self.fmt = fmt
        self.normalize = normalize
        self.mode = mode

    def prop(self, name: str) -> Any:
        if name == "ClassNames":
            return cellstr_col(self.classes or [])
        if name == "InputSize":
            return np.array([[float(self.input_size[0]), float(self.input_size[1]), 3.0]])
        if name == "OutputFormat":
            return char(self.fmt)
        if name == "Normalize":
            return np.array([[self.normalize]])
        if name == "PreprocessMode":
            return char(self.mode)
        return self.p[name]

    def run(self, img: np.ndarray) -> list[np.ndarray]:
        cv2 = cv2mod(self.CLASS)
        H, W = img.shape[:2]
        ih, iw = self.input_size
        rgb = rgb_u8(img, self.CLASS)
        if self.mode == "letterbox":
            s = min(iw / W, ih / H)
            nw, nh = int(round(W * s)), int(round(H * s))
            canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
            canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sx = sy = s
            net_in = canvas
        else:
            net_in = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
            sx, sy = iw / W, ih / H
        blob = cv2.dnn.blobFromImage(net_in, 1 / 255.0 if self.normalize else 1.0, (iw, ih), swapRB=False)
        self.net.setInput(blob)
        names = self.net.getUnconnectedOutLayersNames()
        outs = self.net.forward(names)
        return list(outs), (sx, sy)

    def decode(self, outs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (cx, cy, w, h in network pixels), best class scores and class indices."""
        o = np.asarray(outs[0], dtype=float)
        o = o.reshape(o.shape[-2], o.shape[-1]) if o.ndim >= 2 else o.reshape(1, -1)
        nc = len(self.classes) if self.classes else None
        fmt = self.fmt
        if fmt == "auto":
            if nc is not None and o.shape[0] == 4 + nc and o.shape[1] != 4 + nc:
                fmt = "yolov8"
            elif nc is not None and o.shape[1] == 4 + nc:
                fmt = "yolov8t"
            else:
                fmt = "yolov5"
        if fmt == "yolov8":
            o = o.T
            fmt = "yolov8t"
        if fmt == "yolox":
            o = self._yolox_grid(o)
            fmt = "yolov5"
        if fmt == "yolov8t":
            box, cls = o[:, :4], o[:, 4:]
            obj = np.ones(len(o))
        else:
            box, obj, cls = o[:, :4], o[:, 4], o[:, 5:]
        if cls.shape[1] == 0:
            cls = np.ones((len(o), 1))
        k = cls.argmax(axis=1)
        score = obj * cls[np.arange(len(o)), k]
        return box, score, k

    def _yolox_grid(self, o: np.ndarray) -> np.ndarray:
        ih, iw = self.input_size
        grids, strides = [], []
        for s in (8, 16, 32):
            gh, gw = ih // s, iw // s
            yv, xv = np.mgrid[0:gh, 0:gw]
            grids.append(np.column_stack([xv.ravel(), yv.ravel()]))
            strides.append(np.full((gh * gw, 1), s))
        g = np.vstack(grids)
        st = np.vstack(strides)
        if g.shape[0] != o.shape[0]:
            raise MatlabError(f"{self.CLASS}: the model output has {o.shape[0]} rows, but a YOLOX grid for this "
                              f"input size has {g.shape[0]}.", "mylab:vision:badOutput")
        o = o.copy()
        o[:, :2] = (o[:, :2] + g) * st
        o[:, 2:4] = np.exp(o[:, 2:4]) * st
        return o


@method(OnnxDetector, "detect")
def _onnx_detect(self, interp, args, nargout):
    fname = "detect"
    pos, opts = options(list(args), ("Threshold", "SelectStrongest", "MinSize", "MaxSize", "ExecutionEnvironment",
                                     "MiniBatchSize", "Acceleration"), fname, 2)
    if not pos:
        raise MatlabError("detect: expected detect(detector, I, ...).", "MATLAB:minrhs")
    img = image_arg(pos[0], fname)
    th = fscalar(opts["Threshold"], fname, "Threshold") if "Threshold" in opts else 0.5
    strongest = flag(opts["SelectStrongest"]) if "SelectStrongest" in opts else True
    outs, (sx, sy) = self.run(img)
    box, score, k = self.decode(outs)
    keep = score >= th
    box, score, k = box[keep], score[keep], k[keep]
    H, W = img.shape[:2]
    x1 = (box[:, 0] - box[:, 2] / 2) / sx
    y1 = (box[:, 1] - box[:, 3] / 2) / sy
    x2 = (box[:, 0] + box[:, 2] / 2) / sx
    y2 = (box[:, 1] + box[:, 3] / 2) / sy
    x1, y1 = np.clip(x1, 0, W), np.clip(y1, 0, H)
    x2, y2 = np.clip(x2, 0, W), np.clip(y2, 0, H)
    b = np.column_stack([x1 + 1, y1 + 1, x2 - x1, y2 - y1])
    ok = (b[:, 2] > 0) & (b[:, 3] > 0)
    if "MinSize" in opts:
        ms = fvec(opts["MinSize"], fname, "MinSize", 2)
        ok &= (b[:, 3] >= ms[0]) & (b[:, 2] >= ms[1])
    if "MaxSize" in opts:
        ms = fvec(opts["MaxSize"], fname, "MaxSize", 2)
        ok &= (b[:, 3] <= ms[0]) & (b[:, 2] <= ms[1])
    b, score, k = b[ok], score[ok], k[ok]
    if strongest and len(score):
        idx = nms(b, score, 0.5, "union", np.inf, k)
        b, score, k = b[idx], score[idx], k[idx]
    order = np.argsort(-score, kind="stable")
    b, score, k = b[order], score[order], k[order]
    names = self.classes or [f"class{i + 1}" for i in range(int(k.max()) + 1 if k.size else 0)]
    if k.size and k.max() >= len(names):
        raise MatlabError(f"detect: the model predicts {int(k.max()) + 1} classes but only {len(names)} class names "
                          "were given.", "mylab:vision:classCount")
    labels = interp.call_named("categorical", [cellstr_col([names[i] for i in k]), cellstr_col(names)], 1,
                               interp.frame)[0]
    return b, col(score).astype(np.float32), labels


def _onnx_ctor(cls: str, default_fmt: str, default_norm: bool, default_mode: str):
    def fn(interp, args, nargout):
        pos, opts = options(list(args), ("InputSize", "OutputFormat", "Normalize", "PreprocessMode", "ModelName",
                                         "ClassNames"), cls, 2, lead=1)
        if not pos:
            raise MatlabError(
                f"{cls}: give the path of an ONNX model file, e.g. {cls}('model.onnx', classNames). mylab ships "
                "no pretrained networks, and the MATLAB pretrained model names are not available.",
                "mylab:vision:noModelFile")
        if is_text(pos[0]) is False and not isinstance(pos[0], np.ndarray):
            raise MatlabError(f"{cls}: the first input must be the ONNX file name.")
        path = text(pos[0])
        if not path.lower().endswith(".onnx") and not os.path.isfile(path):
            raise MatlabError(
                f"{cls}: '{path}' is not an ONNX file. mylab cannot download the pretrained MATLAB detectors; "
                "export a YOLO model to ONNX (for example with the Ultralytics or YOLOX tools) and pass its path.",
                "mylab:vision:noModelFile")
        classes = text_list(pos[1]) if len(pos) > 1 else (text_list(opts["ClassNames"]) if "ClassNames" in opts
                                                         else None)
        isz = fvec(opts["InputSize"], cls, "InputSize") if "InputSize" in opts else np.array([640.0, 640.0])
        fmt = opt_text(opts, "OutputFormat", default_fmt, _FORMATS, cls)
        norm = flag(opts["Normalize"]) if "Normalize" in opts else (False if fmt == "yolox" else default_norm)
        mode = opt_text(opts, "PreprocessMode", default_mode, ("resize", "letterbox"), cls)
        det = OnnxDetector(cls, path, classes, (int(isz[0]), int(isz[1])), fmt, norm, mode)
        if "ModelName" in opts:
            det.p["ModelName"] = char(text(opts["ModelName"]))
        return det
    builtin(cls)(fn)


_onnx_ctor("yolov4ObjectDetector", "auto", True, "resize")
_onnx_ctor("yoloxObjectDetector", "yolox", False, "letterbox")
_onnx_ctor("yolov8ObjectDetector", "yolov8", True, "letterbox")


# -- detection metrics -------------------------------------------------------------------------------------

def _per_image(v: Any, fname: str, what: str) -> list[np.ndarray]:
    if isinstance(v, Cell):
        return [_boxes(x, fname, what) for x in v.data.flatten(order="F")]
    return [_boxes(v, fname, what)]


@builtin("bboxPrecisionRecall")
def _bbox_precision_recall(interp, args, nargout):
    fname = "bboxPrecisionRecall"
    if len(args) not in (2, 3):
        raise MatlabError("bboxPrecisionRecall: expected bboxPrecisionRecall(bboxes, groundTruthBboxes, threshold).",
                          "MATLAB:minrhs")
    th = fscalar(args[2], fname, "threshold") if len(args) == 3 else 0.5
    dets = _per_image(args[0], fname, "bboxes")
    gts = _per_image(args[1], fname, "groundTruthBboxes")
    if len(dets) != len(gts):
        raise MatlabError("bboxPrecisionRecall: give the same number of images for detections and ground truth.")
    tp = nd = ng = 0
    for d, g in zip(dets, gts):
        tp += len(_greedy_match(d, np.ones(len(d)), g, th)[0])
        nd += len(d)
        ng += len(g)
    prec = tp / nd if nd else (1.0 if ng == 0 else 0.0)
    rec = tp / ng if ng else 1.0
    return scalar(prec), scalar(rec)


def _greedy_match(d: np.ndarray, s: np.ndarray, g: np.ndarray, th: float):
    """Detections (highest score first) matched to unmatched ground truth with IoU >= th."""
    ov = overlap(d, g)
    taken = np.zeros(len(g), dtype=bool)
    hits, flags = [], np.zeros(len(d), dtype=bool)
    for i in np.argsort(-s, kind="stable"):
        if len(g) == 0:
            break
        cand = np.where(taken, -1.0, ov[i])
        j = int(np.argmax(cand))
        if cand[j] >= th:
            taken[j] = True
            flags[i] = True
            hits.append((i, j))
    return hits, flags


def average_precision(dets: list[np.ndarray], scores: list[np.ndarray], gts: list[np.ndarray], th: float):
    """VOC-style AP (area under the interpolated precision-recall curve), recall and precision."""
    allscores, alltp = [], []
    ng = sum(len(g) for g in gts)
    for d, s, g in zip(dets, scores, gts):
        _, flags = _greedy_match(d, s, g, th)
        allscores.append(s)
        alltp.append(flags)
    s = np.concatenate(allscores) if allscores else np.zeros(0)
    tp = np.concatenate(alltp) if alltp else np.zeros(0, dtype=bool)
    order = np.argsort(-s, kind="stable")
    tp = tp[order].astype(float)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    recall = ctp / ng if ng else np.zeros_like(ctp)
    precision = ctp / np.maximum(ctp + cfp, 1e-300)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    # precision envelope, then sum over recall steps
    env = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum((recall[1:] - recall[:-1]) * env[1:])) if len(recall) > 1 else 0.0
    return ap, recall, precision


def _table_column(t, k: int):
    return t.columns[k]


def _boxes_scores_from(det, fname: str):
    """Per-image boxes, scores and labels from a detection-results table or a cell array of boxes."""
    if hasattr(det, "names") and hasattr(det, "columns"):
        names = [n.lower() for n in det.names]
        if "boxes" not in names or "scores" not in names:
            raise MatlabError(f"{fname}: the detection results table needs Boxes and Scores variables.")
        bc = det.columns[names.index("boxes")]
        sc = det.columns[names.index("scores")]
        lc = det.columns[names.index("labels")] if "labels" in names else None
        boxes = _per_image(bc, fname, "detected boxes") if isinstance(bc, Cell) else [_boxes(bc, fname)]
        scores = [fvec(x, fname, "scores") for x in sc.data.flatten(order="F")] if isinstance(sc, Cell) \
            else [fvec(sc, fname, "scores")]
        labels = None
        if lc is not None:
            labels = [text_list_any(x) for x in lc.data.flatten(order="F")] if isinstance(lc, Cell) \
                else [text_list_any(lc)]
        return boxes, scores, labels
    raise MatlabError(f"{fname}: the detection results must be a table with Boxes and Scores variables.")


def text_list_any(v: Any) -> list[str]:
    if hasattr(v, "codes") and hasattr(v, "categories"):
        cats = list(v.categories)
        return [cats[int(c) - 1] if c > 0 else "" for c in np.asarray(v.codes).ravel(order="F")]
    return text_list(v)


@builtin("evaluateDetectionPrecision")
def _evaluate_detection_precision(interp, args, nargout):
    fname = "evaluateDetectionPrecision"
    if len(args) not in (2, 3):
        raise MatlabError("evaluateDetectionPrecision: expected evaluateDetectionPrecision(detectionResults, "
                          "groundTruthData, threshold).", "MATLAB:minrhs")
    th = fscalar(args[2], fname, "threshold") if len(args) == 3 else 0.5
    boxes, scores, labels = _boxes_scores_from(args[0], fname)
    gt = args[1]
    if not (hasattr(gt, "names") and hasattr(gt, "columns")):
        raise MatlabError(f"{fname}: the ground truth must be a table with one variable of boxes per class.")
    classes = list(gt.names)
    aps, recs, precs = [], [], []
    for ci, cname in enumerate(classes):
        col_ = gt.columns[ci]
        g = _per_image(col_, fname, "ground truth boxes") if isinstance(col_, Cell) else [_boxes(col_, fname)]
        if len(g) != len(boxes):
            raise MatlabError(f"{fname}: the detection results and the ground truth must cover the same images.")
        if labels is None:
            if len(classes) > 1:
                raise MatlabError(f"{fname}: with several classes the detection results need a Labels variable.")
            d, s = boxes, scores
        else:
            d = [b[np.array([lab == cname for lab in L], dtype=bool)] if len(b) else b for b, L in zip(boxes, labels)]
            s = [sc[np.array([lab == cname for lab in L], dtype=bool)] if len(sc) else sc
                 for sc, L in zip(scores, labels)]
        ap, r, p = average_precision(d, s, g, th)
        aps.append(ap)
        recs.append(r.reshape(-1, 1))
        precs.append(p.reshape(-1, 1))
    if len(classes) == 1:
        return scalar(aps[0]), recs[0], precs[0]
    rc = np.empty((len(classes), 1), dtype=object)
    pc = np.empty((len(classes), 1), dtype=object)
    for i in range(len(classes)):
        rc[i, 0], pc[i, 0] = recs[i], precs[i]
    return col(aps), Cell(rc), Cell(pc)


@builtin("evaluateDetectionMissRate")
def _evaluate_detection_miss_rate(interp, args, nargout):
    fname = "evaluateDetectionMissRate"
    if len(args) not in (2, 3):
        raise MatlabError("evaluateDetectionMissRate: expected evaluateDetectionMissRate(detectionResults, "
                          "groundTruthData, threshold).", "MATLAB:minrhs")
    th = fscalar(args[2], fname, "threshold") if len(args) == 3 else 0.5
    boxes, scores, _ = _boxes_scores_from(args[0], fname)
    gt = args[1]
    col_ = gt.columns[0]
    g = _per_image(col_, fname, "ground truth boxes") if isinstance(col_, Cell) else [_boxes(col_, fname)]
    ng = sum(len(x) for x in g)
    flags, ss = [], []
    for d, s, gg in zip(boxes, scores, g):
        _, f = _greedy_match(d, s, gg, th)
        flags.append(f)
        ss.append(s)
    s = np.concatenate(ss) if ss else np.zeros(0)
    tp = np.concatenate(flags)[np.argsort(-s, kind="stable")].astype(float) if flags else np.zeros(0)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    miss = np.concatenate([[1.0], 1 - ctp / max(ng, 1)])
    fppi = np.concatenate([[0.0], cfp / max(len(boxes), 1)])
    # log-average miss rate at nine FPPI points between 1e-2 and 1
    refs = np.logspace(-2, 0, 9)
    vals = []
    for r in refs:
        idx = np.nonzero(fppi <= r)[0]
        vals.append(miss[idx[-1]] if idx.size else 1.0)
    lamr = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-10)))))
    return scalar(lamr), col(fppi), col(miss)

