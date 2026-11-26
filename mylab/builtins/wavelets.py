"""Wavelets: orthogonal wavelet filters, the discrete wavelet transform in one and two dimensions,
multilevel decompositions, thresholding and denoising, the continuous wavelet transform and the
maximal-overlap (undecimated) transform."""

from __future__ import annotations

from math import comb

import numpy as np

from ..errors import MatlabError
from ..values import Cell, char, to_int
from ._registry import builtin
from ._util import fnum, is_text, nargs, text

# -- wavelet filters ------------------------------------------------------------------------------------
# Low-pass decomposition filters. The least-asymmetric (symlet) and coiflet filters are the published
# tables of I. Daubechies ("Ten Lectures on Wavelets", 1992), normalised to sum to sqrt(2); the Daubechies
# extremal-phase filters are computed from their defining polynomial.

_SYM = {
    4: [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
        0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    5: [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936,
        0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047,
        -0.021101834024758855, 0.019538882735286728],
    6: [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633,
        0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252,
        -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    7: [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357,
        0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628,
        0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774,
        0.004010244871533663, 0.010268176708511255],
    8: [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605,
        -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235,
        0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506,
        0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
    9: [0.0014009155259146807, 0.0006197808889855868, -0.013271967781817119, -0.01152821020767923,
        0.03022487885827568, 0.0005834627461258068, -0.05456895843083407, 0.238760914607303,
        0.717897082764412, 0.6173384491409358, 0.035272488035271894, -0.19155083129728512,
        -0.018233770779395985, 0.06207778930288603, 0.008859267493400484, -0.010264064027633142,
        -0.0004731544986800831, 0.0010694900329086053],
    10: [0.0007701598091144901, 9.563267072289475e-05, -0.008641299277022422, -0.0014653825813050513,
         0.0459272392310922, 0.011609893903711381, -0.15949427888491757, -0.07088053578324385,
         0.47169066693843925, 0.7695100370211071, 0.38382676106708546, -0.03553674047381755,
         -0.0319900568824278, 0.04999497207737669, 0.005764912033581909, -0.02035493981231129,
         -0.0008043589320165449, 0.004593173585311828, 5.7036083618494284e-05, -0.0004593294210046588],
}

_COIF = {
    1: [-0.01565572813546454, -0.0727326195128539, 0.38486484686420286, 0.8525720202122554,
        0.3378976624578092, -0.0727326195128539],
    2: [-0.0007205494453645122, -0.0018232088707029932, 0.0056114348193944995, 0.023680171946334084,
        -0.0594344186464569, -0.0764885990783064, 0.41700518442169254, 0.8127236354455423,
        0.3861100668211622, -0.06737255472196302, -0.04146493678175915, 0.016387336463522112],
    3: [-3.459977283621256e-05, -7.098330313814125e-05, 0.0004662169601128863, 0.0011175187708906016,
        -0.0025745176887502236, -0.00900797613666158, 0.015880544863615904, 0.03455502757306163,
        -0.08230192710688598, -0.07179982161931202, 0.42848347637761874, 0.7937772226256206,
        0.4051769024096169, -0.06112339000267287, -0.0657719112818555, 0.023452696141836267,
        0.007782596427325418, -0.003793512864491014],
    4: [-1.7849850030882614e-06, -3.2596802368833675e-06, 3.1229875865345646e-05, 6.233903446100713e-05,
        -0.00025997455248771324, -0.0005890207562443383, 0.0012665619292989445, 0.003751436157278457,
        -0.00565828668661072, -0.015211731527946259, 0.025082261844864097, 0.03933442712333749,
        -0.09622044203398798, -0.06662747426342504, 0.4343860564914685, 0.782238930920499,
        0.41530840703043026, -0.05607731331675481, -0.08126669968087875, 0.026682300156053072,
        0.016068943964776348, -0.0073461663276420935, -0.0016294920126017326, 0.0008923136685823146],
    5: [-9.517657273819165e-08, -1.6744288576823017e-07, 2.0637618513646814e-06, 3.7346551751414047e-06,
        -2.1315026809955787e-05, -4.134043227251251e-05, 0.00014054114970203437, 0.00030225958181306315,
        -0.0006381313430451114, -0.0016628637020130838, 0.0024333732126576722, 0.006764185448053083,
        -0.009164231162481846, -0.01976177894257264, 0.03268357426711183, 0.0412892087501817,
        -0.10557420870333893, -0.06203596396290357, 0.4379916261718371, 0.7742896036529562,
        0.4215662066908515, -0.05204316317624377, -0.09192001055969624, 0.02816802897093635,
        0.023408156785839195, -0.010131117519849788, -0.004159358781386048, 0.0021782363581090178,
        0.00035858968789573785, -0.00021208083980379827],
}

_FILTER_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _db_root_groups(N: int) -> list[tuple[list[complex], list[complex]]]:
    """Roots of the Daubechies polynomial as (inside, outside) choices for every conjugate group."""
    P = [comb(N - 1 + k, k) for k in range(N)]
    ys = np.roots(P[::-1]) if N > 1 else np.zeros(0)
    groups = []
    used = np.zeros(ys.size, dtype=bool)
    for i, y in enumerate(ys):
        if used[i]:
            continue
        used[i] = True
        b = 2 - 4 * y
        z1 = (b + np.sqrt(b * b - 4 + 0j)) / 2
        z2 = 1 / z1
        zin, zout = (z1, z2) if abs(z1) < 1 else (z2, z1)
        if abs(y.imag) > 1e-10:
            j = min((k for k in range(ys.size) if not used[k]), key=lambda k: abs(ys[k] - np.conj(y)))
            used[j] = True
            groups.append(([zin, np.conj(zin)], [zout, np.conj(zout)]))
        else:
            groups.append(([complex(zin.real)], [complex(zout.real)]))
    return groups


def _scaling_from_roots(N: int, roots: list[complex]) -> np.ndarray:
    h = np.array([1.0 + 0j])
    for _ in range(N):
        h = np.convolve(h, [1.0, 1.0])
    for r in roots:
        h = np.convolve(h, [1.0, -r])
    h = np.real(h)
    return h / h.sum() * np.sqrt(2)


def _db_lor(N: int) -> np.ndarray:
    """Extremal-phase (minimum-phase) Daubechies scaling filter with N vanishing moments."""
    if N == 1:
        return np.array([1.0, 1.0]) / np.sqrt(2)
    roots = [r for inside, _out in _db_root_groups(N) for r in inside]
    return _scaling_from_roots(N, roots)


def _sym_lor(N: int) -> np.ndarray:
    """Least-asymmetric filter of order N beyond the published table: the most nearly linear phase."""
    import itertools
    groups = _db_root_groups(N)
    if len(groups) > 14:
        raise MatlabError(f"Symlets above order {N - 1} are not available.")
    best = None
    w = np.arange(1, 256) * np.pi / 256
    A = np.vstack([w, np.ones_like(w)]).T
    for choice in itertools.product((0, 1), repeat=len(groups)):
        roots = [r for g, c in zip(groups, choice) for r in g[c]]
        h = _scaling_from_roots(N, roots)
        ph = np.unwrap(np.angle(np.polyval(h[::-1], np.exp(-1j * w))))
        coef = np.linalg.lstsq(A, ph, rcond=None)[0]
        cost = float(np.sum((ph - A @ coef) ** 2))
        if best is None or cost < best[0] - 1e-9:
            best = (cost, h)
    return best[1]


def _qmf(x: np.ndarray, p: int = 0) -> np.ndarray:
    y = np.array(x[::-1], dtype=float)
    first = 1 - (p % 2)           # 0-based index of the first sign change
    y[first::2] = -y[first::2]
    return y


def wavelet_filters(wname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(LoD, HiD, LoR, HiR) of an orthogonal wavelet."""
    key = wname.strip().lower()
    if key in _FILTER_CACHE:
        return _FILTER_CACHE[key]
    fam = key.rstrip("0123456789")
    num = key[len(fam):]
    if key == "haar":
        lor = _db_lor(1)
    elif fam in ("db", "sym", "coif") and num.isdigit() and int(num) >= 1:
        n = int(num)
        if fam == "db":
            if n > 38:
                raise MatlabError(f"wfilters: '{wname}' is not available (Daubechies wavelets go up to db38).")
            lor = _db_lor(n)
        elif fam == "sym":
            if n in _SYM:
                lor = np.array(_SYM[n][::-1])
            elif n <= 3:
                lor = _db_lor(n)
            else:
                lor = _sym_lor(n)
        else:
            if n not in _COIF:
                raise MatlabError(f"wfilters: '{wname}' is not available (coiflets go from coif1 to coif5).")
            lor = np.array(_COIF[n][::-1])
    else:
        raise MatlabError(f"wfilters: the wavelet '{wname}' is not available. Supported families: 'haar', "
                          "'dbN', 'symN' and 'coifN'.")
    hir = _qmf(lor)
    out = (lor[::-1].copy(), hir[::-1].copy(), lor.copy(), hir)
    _FILTER_CACHE[key] = out
    return out


def _row(v) -> np.ndarray:
    return np.asarray(v).reshape(1, -1)


def _col(v) -> np.ndarray:
    return np.asarray(v).reshape(-1, 1)


def _f(x) -> np.ndarray:
    return np.array([[float(x)]])


def _vec(v, what: str = "argument") -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
        raise MatlabError(f"Expected {what} to be numeric.")
    return fnum(v).flatten(order="F")


def _like(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _row(r) if (x.ndim == 2 and x.shape[0] == 1) else _col(r)


@builtin("wfilters")
def _wfilters(interp, args, nargout):
    nargs(args, 1, 2, "wfilters")
    lod, hid, lor, hir = wavelet_filters(text(args[0], "wavelet name"))
    if len(args) > 1:
        kind = text(args[1]).lower()
        pairs = {"d": (lod, hid), "r": (lor, hir), "l": (lod, lor), "h": (hid, hir)}
        if kind not in pairs:
            raise MatlabError("wfilters: the type must be 'd', 'r', 'l' or 'h'.")
        a, b = pairs[kind]
        return (_row(a), _row(b)) if nargout > 1 else _row(a)
    out = (_row(lod), _row(hid), _row(lor), _row(hir))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("orthfilt")
def _orthfilt(interp, args, nargout):
    nargs(args, 1, 2, "orthfilt")
    w = np.real(_vec(args[0], "scaling filter")).astype(float)
    p = to_int(args[1], "p") if len(args) > 1 else 0
    lor = np.sqrt(2) * w / np.sum(w)
    hir = _qmf(lor, p)
    out = (_row(lor[::-1]), _row(hir[::-1]), _row(lor), _row(hir))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("qmf")
def _qmf_b(interp, args, nargout):
    nargs(args, 1, 2, "qmf")
    x = fnum(args[0])
    p = to_int(args[1], "p") if len(args) > 1 else 0
    return _like(_qmf(np.real(x).flatten(order="F"), p), x)


@builtin("wrev")
def _wrev(interp, args, nargout):
    nargs(args, 1, 1, "wrev")
    x = fnum(args[0])
    return _like(x.flatten(order="F")[::-1], x)


def _filters_from(args: list, pos: int, name: str, recon: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Filters given as a wavelet name or as two vectors starting at args[pos]; returns them and the next index."""
    if len(args) <= pos:
        raise MatlabError(f"{name}: a wavelet name or a pair of filters is required.")
    if is_text(args[pos]):
        lod, hid, lor, hir = wavelet_filters(text(args[pos]))
        return (lor, hir, pos + 1) if recon else (lod, hid, pos + 1)
    if len(args) <= pos + 1:
        raise MatlabError(f"{name}: two filters are required.")
    a = np.real(_vec(args[pos], "low-pass filter")).astype(float)
    b = np.real(_vec(args[pos + 1], "high-pass filter")).astype(float)
    if a.size != b.size:
        raise MatlabError(f"{name}: the two filters must have the same length.")
    return a, b, pos + 2


# -- signal extension ---------------------------------------------------------------------------------

_MODES = {"sym": "sym", "symh": "sym", "symw": "symw", "asym": "asym", "asymh": "asym", "asymw": "asymw",
          "zpd": "zpd", "sp0": "sp0", "sp1": "sp1", "spd": "sp1", "ppd": "ppd", "per": "per"}


def _mode_name(v, name: str) -> str:
    s = text(v).lower()
    if s not in _MODES:
        raise MatlabError(f"{name}: unknown extension mode '{text(v)}'. Use 'sym', 'symw', 'asym', 'asymw', "
                          "'zpd', 'sp0', 'sp1', 'ppd' or 'per'.")
    return _MODES[s]


def _default_mode(interp) -> str:
    return interp.state.get("dwtmode", "sym") if interp is not None else "sym"


def _take_mode(interp, args: list, name: str) -> tuple[list, str]:
    """Remove a trailing 'mode', value pair; returns the other arguments and the extension mode."""
    rest = list(args)
    mode = _default_mode(interp)
    for i in range(len(rest) - 1):
        if is_text(rest[i]) and text(rest[i]).lower() == "mode":
            mode = _mode_name(rest[i + 1], name)
            del rest[i:i + 2]
            break
    return rest, mode


def _extend(a: np.ndarray, n: int, mode: str, axis: int = 0, left: bool = True, right: bool = True) -> np.ndarray:
    """Extend an array by n samples at the chosen ends of one axis."""
    if n <= 0:
        return a
    a = np.moveaxis(a, axis, 0)
    N = a.shape[0]
    pad = [(n if left else 0, n if right else 0)] + [(0, 0)] * (a.ndim - 1)
    if mode == "per":
        if N % 2:
            a = np.concatenate([a, a[-1:]], axis=0)
        out = np.pad(a, pad, mode="wrap")
    elif mode == "ppd":
        out = np.pad(a, pad, mode="wrap")
    elif mode == "sym":
        out = np.pad(a, pad, mode="symmetric")
    elif mode == "symw":
        out = np.pad(a, pad, mode="reflect") if N > 1 else np.pad(a, pad, mode="edge")
    elif mode == "asym":
        out = np.pad(a, pad, mode="symmetric")
        if left:
            out[:n] = -out[:n]
        if right:
            out[out.shape[0] - n:] = -out[out.shape[0] - n:]
    elif mode == "asymw":
        out = np.pad(a, pad, mode="reflect", reflect_type="odd") if N > 1 else np.pad(a, pad, mode="edge")
    elif mode == "zpd":
        out = np.pad(a, pad, mode="constant")
    elif mode == "sp0":
        out = np.pad(a, pad, mode="edge")
    else:   # sp1: straight-line extrapolation from the two samples at each end
        out = np.pad(a.astype(float), pad, mode="edge")
        k = np.arange(n, 0, -1).reshape((-1,) + (1,) * (a.ndim - 1))
        slope_l = a[1] - a[0] if N > 1 else 0.0
        slope_r = a[-1] - a[-2] if N > 1 else 0.0
        if left:
            out[:n] = a[0] - k * slope_l
        if right:
            out[out.shape[0] - n:] = a[-1] + k[::-1] * slope_r
    return np.moveaxis(out, 0, axis)


@builtin("dwtmode")
def _dwtmode(interp, args, nargout):
    nargs(args, 0, 2, "dwtmode")
    if not args or text(args[0]).lower() == "status":
        mode = _default_mode(interp)
        if nargout or (len(args) > 1 and text(args[1]).lower() == "nodisp"):
            return char(mode)
        interp.out(f"DWT extension mode: {mode}\n")
        return None
    first = text(args[0]).lower()
    if first == "save":
        first = text(args[1]).lower() if len(args) > 1 else _default_mode(interp)
    interp.state["dwtmode"] = _mode_name(char(first), "dwtmode")
    return None


@builtin("wextend")
def _wextend(interp, args, nargout):
    nargs(args, 4, 5, "wextend")
    kind = text(args[0]).lower() if is_text(args[0]) else str(int(fnum(args[0]).flat[0]))
    mode = _mode_name(args[1], "wextend")
    x = fnum(args[2])
    loc = text(args[4]).lower() if len(args) > 4 else None
    lens = np.real(fnum(args[3])).flatten().astype(int)
    if kind in ("1", "1d", "1-d"):
        v = x.flatten(order="F")
        loc = loc or "b"
        if loc not in ("l", "u", "r", "d", "b"):
            raise MatlabError("wextend: the location must be 'l', 'r' or 'b'.")
        out = _extend(v, int(lens[0]), mode, 0, left=loc in ("l", "u", "b"), right=loc in ("r", "d", "b"))
        return _like(out, x)
    if kind in ("2", "2d", "2-d"):
        lr, lc = (int(lens[0]), int(lens[0])) if lens.size == 1 else (int(lens[0]), int(lens[1]))
        loc = loc or "bb"
        if len(loc) == 1:
            loc = loc * 2
        out = _extend(x, lr, mode, 0, left=loc[0] in ("u", "b", "l"), right=loc[0] in ("d", "b", "r"))
        return _extend(out, lc, mode, 1, left=loc[1] in ("l", "b", "u"), right=loc[1] in ("r", "b", "d"))
    if kind in ("ar", "addrow"):
        loc = loc or "b"
        return _extend(x, int(lens[0]), mode, 0, left=loc in ("u", "b", "l"), right=loc in ("d", "b", "r"))
    if kind in ("ac", "addcol"):
        loc = loc or "b"
        return _extend(x, int(lens[0]), mode, 1, left=loc in ("l", "b", "u"), right=loc in ("r", "b", "d"))
    raise MatlabError("wextend: the type must be '1D', '2D', 'addrow' or 'addcol'.")


@builtin("dyaddown")
def _dyaddown(interp, args, nargout):
    nargs(args, 1, 3, "dyaddown")
    x = fnum(args[0])
    rest = list(args[1:])
    kind = None
    if rest and is_text(rest[-1]):
        kind = text(rest.pop()).lower()
    even = to_int(rest[0], "evenodd") if rest else 0
    first = 1 - (even % 2)          # 0: keep the even-numbered samples (2, 4, ...), 1: the odd ones
    if x.ndim == 2 and min(x.shape) == 1 and kind is None:
        return _like(x.flatten(order="F")[first::2], x)
    kind = kind or "c"
    if kind == "c":
        return x[:, first::2]
    if kind == "r":
        return x[first::2, :]
    if kind == "m":
        return x[first::2, first::2]
    raise MatlabError("dyaddown: the type must be 'c', 'r' or 'm'.")


def _dyadup(v: np.ndarray, even: int, axis: int = 0) -> np.ndarray:
    v = np.moveaxis(v, axis, 0)
    n = v.shape[0]
    if even % 2 == 0:               # samples at odd positions: x1 0 x2 ... 0 xn
        out = np.zeros((max(2 * n - 1, 0),) + v.shape[1:], dtype=v.dtype)
        out[::2] = v
    else:                           # samples at even positions: 0 x1 0 ... xn 0
        out = np.zeros((2 * n + 1,) + v.shape[1:], dtype=v.dtype)
        out[1::2] = v
    return np.moveaxis(out, 0, axis)


@builtin("dyadup")
def _dyadup_b(interp, args, nargout):
    nargs(args, 1, 3, "dyadup")
    x = fnum(args[0])
    rest = list(args[1:])
    kind = None
    if rest and is_text(rest[-1]):
        kind = text(rest.pop()).lower()
    even = to_int(rest[0], "evenodd") if rest else 1
    if x.ndim == 2 and min(x.shape) == 1 and kind is None:
        return _like(_dyadup(x.flatten(order="F"), even), x)
    kind = kind or "c"
    if kind == "c":
        return _dyadup(x, even, 1)
    if kind == "r":
        return _dyadup(x, even, 0)
    if kind == "m":
        return _dyadup(_dyadup(x, even, 0), even, 1)
    raise MatlabError("dyadup: the type must be 'c', 'r' or 'm'.")


def _keep_centre(v: np.ndarray, length: int, axis: int = 0) -> np.ndarray:
    n = v.shape[axis]
    if length >= n:
        return v
    start = (n - length) // 2
    sl = [slice(None)] * v.ndim
    sl[axis] = slice(start, start + length)
    return v[tuple(sl)]


@builtin("wkeep")
def _wkeep(interp, args, nargout):
    nargs(args, 2, 4, "wkeep")
    x = fnum(args[0])
    lens = np.real(fnum(args[1])).flatten().astype(int)
    opt = args[2] if len(args) > 2 else None
    if x.ndim == 2 and min(x.shape) == 1:
        v = x.flatten(order="F")
        L = int(lens[0])
        if L >= v.size:
            return x
        if opt is None or (is_text(opt) and text(opt).lower() == "c"):
            out = _keep_centre(v, L)
        elif is_text(opt) and text(opt).lower() == "l":
            out = v[:L]
        elif is_text(opt) and text(opt).lower() == "r":
            out = v[v.size - L:]
        else:
            first = to_int(opt, "first index")
            out = v[first - 1:first - 1 + L]
        return _like(out, x)
    out = _keep_centre(_keep_centre(x, int(lens[0]), 0), int(lens[-1] if lens.size > 1 else lens[0]), 1)
    return out


@builtin("wmaxlev")
def _wmaxlev(interp, args, nargout):
    nargs(args, 2, 2, "wmaxlev")
    size = np.real(fnum(args[0])).flatten()
    lf = wavelet_filters(text(args[1]))[0].size
    lx = float(np.min(size))
    if lx < 1:
        return 0.0
    lev = int(np.floor(np.log2(lx / (lf - 1)))) if lf > 1 else 0
    return float(max(lev, 0))


# -- one-dimensional transform --------------------------------------------------------------------------

def _conv_valid(y: np.ndarray, f: np.ndarray, axis: int = 0) -> np.ndarray:
    """'valid' convolution with f along one axis."""
    y = np.moveaxis(y, axis, 0)
    n = y.shape[0] - f.size + 1
    out = np.zeros((n,) + y.shape[1:], dtype=np.result_type(y, f))
    for j, c in enumerate(f):
        if c != 0:
            out += c * y[f.size - 1 - j:f.size - 1 - j + n]
    return np.moveaxis(out, 0, axis)


def _conv_full(y: np.ndarray, f: np.ndarray, axis: int = 0) -> np.ndarray:
    y = np.moveaxis(y, axis, 0)
    n = y.shape[0]
    out = np.zeros((n + f.size - 1,) + y.shape[1:], dtype=np.result_type(y, f))
    for j, c in enumerate(f):
        if c != 0:
            out[j:j + n] += c * y
    return np.moveaxis(out, 0, axis)


def _analysis(x: np.ndarray, f: np.ndarray, mode: str, axis: int = 0) -> np.ndarray:
    """Filter with f and keep every second sample (one branch of the forward transform)."""
    lf = f.size
    lx = x.shape[axis]
    if mode == "per":
        y = _extend(x, lf // 2, "per", axis)
        z = _conv_valid(y, f, axis)
        last = lx + (lx % 2)
        sl = [slice(None)] * z.ndim
        sl[axis] = slice(1, last, 2)
        return z[tuple(sl)]
    y = _extend(x, lf - 1, mode, axis)
    z = _conv_valid(y, f, axis)
    sl = [slice(None)] * z.ndim
    sl[axis] = slice(1, None, 2)
    return z[tuple(sl)]


def _synthesis(c: np.ndarray, f: np.ndarray, mode: str, length: int | None, axis: int = 0) -> np.ndarray:
    """Insert zeros, filter with f and keep the centre (one branch of the inverse transform)."""
    lf = f.size
    lc = c.shape[axis]
    if mode == "per":
        n = 2 * lc
        cm = np.moveaxis(c, axis, 0)
        out = np.zeros((n,) + cm.shape[1:], dtype=np.result_type(c, f))
        k = np.arange(lc)
        for j in range(lf):
            # the analysis step reads x[(2k + lf/2 - j) mod n] with weight LoD[j] = f[lf-1-j]
            idx = (2 * k + lf // 2 - j) % n
            np.add.at(out, idx, f[lf - 1 - j] * cm)
        out = np.moveaxis(out, 0, axis)
        if length is not None and length < n:
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(0, length)
            out = out[tuple(sl)]
        return out
    up = _dyadup(c, 0, axis)
    full = _conv_full(up, f, axis)
    want = 2 * lc - lf + 2 if length is None else length
    return _keep_centre(full, want, axis)


def _signal(v, name: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
        raise MatlabError(f"{name}: the signal must be numeric.")
    x = fnum(v)
    if x.ndim != 2 or min(x.shape) > 1:
        raise MatlabError(f"{name}: the signal must be a vector.")
    return x


@builtin("dwt")
def _dwt(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "dwt")
    nargs(rest, 2, 3, "dwt")
    x = _signal(rest[0], "dwt")
    lod, hid, _ = _filters_from(rest, 1, "dwt", False)
    v = x.flatten(order="F")
    ca, cd = _analysis(v, lod, mode), _analysis(v, hid, mode)
    return (_like(ca, x), _like(cd, x)) if nargout > 1 else _like(ca, x)


@builtin("idwt")
def _idwt(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "idwt")
    nargs(rest, 3, 5, "idwt")
    ca, cd = fnum(rest[0]), fnum(rest[1])
    lor, hir, pos = _filters_from(rest, 2, "idwt", True)
    length = to_int(rest[pos], "length") if len(rest) > pos else None
    a, d = ca.flatten(order="F"), cd.flatten(order="F")
    if a.size == 0 and d.size == 0:
        raise MatlabError("idwt: at least one of the coefficient vectors must not be empty.")
    parts = []
    if a.size:
        parts.append(_synthesis(a, lor, mode, length))
    if d.size:
        parts.append(_synthesis(d, hir, mode, length))
    if len(parts) == 2 and parts[0].size != parts[1].size:
        raise MatlabError("idwt: the approximation and detail coefficients must have the same length.")
    out = parts[0] + parts[1] if len(parts) == 2 else parts[0]
    return _like(out, ca if a.size else cd)


def _wavedec(v: np.ndarray, n: int, lod: np.ndarray, hid: np.ndarray, mode: str):
    details, lens = [], [v.size]
    a = v
    for _ in range(n):
        d = _analysis(a, hid, mode)
        a = _analysis(a, lod, mode)
        details.append(d)
        lens.append(d.size)
    c = np.concatenate([a] + details[::-1]) if n else a
    book = np.array([a.size] + lens[:0:-1] + [v.size], dtype=float) if n else np.array([v.size, v.size], float)
    return c, book


@builtin("wavedec")
def _wavedec_b(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "wavedec")
    nargs(rest, 3, 4, "wavedec")
    x = _signal(rest[0], "wavedec")
    n = to_int(rest[1], "level")
    if n < 1:
        raise MatlabError("wavedec: the level must be a positive integer.")
    lod, hid, _ = _filters_from(rest, 2, "wavedec", False)
    c, book = _wavedec(x.flatten(order="F"), n, lod, hid, mode)
    return (_like(c, x), _like(book, x)) if nargout > 1 else _like(c, x)


def _book(args: list, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = fnum(args[0])
    l = np.real(fnum(args[1])).flatten(order="F").astype(int)
    cv = c.flatten(order="F")
    if l.size < 2 or int(np.sum(l[:-1])) != cv.size:
        raise MatlabError(f"{name}: the bookkeeping vector does not match the coefficient vector.")
    return c, cv, l


def _detail(cv: np.ndarray, l: np.ndarray, level: int) -> np.ndarray:
    nmax = l.size - 2
    if not 1 <= level <= nmax:
        raise MatlabError(f"The level must be between 1 and {nmax}.")
    k = nmax - level + 1                 # position of this level's details in l
    first = int(np.sum(l[:k]))
    return cv[first:first + l[k]]


def _appcoef(cv: np.ndarray, l: np.ndarray, lor: np.ndarray, hir: np.ndarray, level: int, mode: str) -> np.ndarray:
    nmax = l.size - 2
    if not 0 <= level <= nmax:
        raise MatlabError(f"The level must be between 0 and {nmax}.")
    a = cv[:l[0]]
    for lev in range(nmax, level, -1):
        d = _detail(cv, l, lev)
        length = int(l[nmax - lev + 2])
        a = _synthesis(a, lor, mode, length) + _synthesis(d, hir, mode, length)
    return a


@builtin("waverec")
def _waverec(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "waverec")
    nargs(rest, 3, 4, "waverec")
    c, cv, l = _book(rest, "waverec")
    lor, hir, _ = _filters_from(rest, 2, "waverec", True)
    return _like(_appcoef(cv, l, lor, hir, 0, mode), c)


@builtin("appcoef")
def _appcoef_b(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "appcoef")
    nargs(rest, 3, 5, "appcoef")
    c, cv, l = _book(rest, "appcoef")
    lor, hir, pos = _filters_from(rest, 2, "appcoef", True)
    level = to_int(rest[pos], "level") if len(rest) > pos else l.size - 2
    return _like(_appcoef(cv, l, lor, hir, level, mode), c)


@builtin("detcoef")
def _detcoef(interp, args, nargout):
    nargs(args, 2, 4, "detcoef")
    rest = [a for a in args if not is_text(a)]
    flags = [text(a).lower() for a in args if is_text(a)]
    c, cv, l = _book(rest, "detcoef")
    nmax = l.size - 2
    levels = np.real(fnum(rest[2])).flatten().astype(int) if len(rest) > 2 else np.array([nmax])
    outs = [_like(_detail(cv, l, int(k)), c) for k in levels]
    if "cells" in flags or (levels.size > 1 and nargout <= 1):
        return Cell.from_list(outs)
    return tuple(outs[:max(nargout, 1)]) if len(outs) > 1 else outs[0]


@builtin("wrcoef")
def _wrcoef(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "wrcoef")
    nargs(rest, 4, 6, "wrcoef")
    kind = text(rest[0]).lower()
    if kind not in ("a", "d"):
        raise MatlabError("wrcoef: the type must be 'a' (approximation) or 'd' (detail).")
    c, cv, l = _book(rest[1:], "wrcoef")
    lor, hir, pos = _filters_from(rest, 3, "wrcoef", True)
    nmax = l.size - 2
    level = to_int(rest[pos], "level") if len(rest) > pos else nmax
    if kind == "a":
        if not 0 <= level <= nmax:
            raise MatlabError(f"wrcoef: the level must be between 0 and {nmax}.")
        a = _appcoef(cv, l, lor, hir, level, mode)
        first = lor
    else:
        a = _detail(cv, l, level)
        first = hir
    # climb back to the signal: the first step uses the branch's own filter, the others the low-pass
    for lev in range(level, 0, -1):
        length = int(l[nmax - lev + 2])
        a = _synthesis(a, first, mode, length)
        first = lor
    return _like(a, c)


@builtin("upcoef")
def _upcoef(interp, args, nargout):
    nargs(args, 3, 6, "upcoef")
    kind = text(args[0]).lower()
    if kind not in ("a", "d"):
        raise MatlabError("upcoef: the type must be 'a' or 'd'.")
    x = fnum(args[1])
    lor, hir, pos = _filters_from(list(args), 2, "upcoef", True)
    n = to_int(args[pos], "level") if len(args) > pos else 1
    length = to_int(args[pos + 1], "length") if len(args) > pos + 1 else None
    v = x.flatten(order="F")
    f = lor if kind == "a" else hir
    for _ in range(n):
        v = _conv_full(_dyadup(v, 0), f)
        f = lor
    if length is not None:
        v = _keep_centre(v, length)
    return _like(v, x)


@builtin("wenergy")
def _wenergy(interp, args, nargout):
    nargs(args, 2, 2, "wenergy")
    c, cv, l = _book(list(args), "wenergy")
    total = float(np.sum(np.abs(cv) ** 2))
    if total == 0:
        return (0.0, _row(np.zeros(l.size - 2))) if nargout > 1 else 0.0
    ea = 100 * float(np.sum(np.abs(cv[:l[0]]) ** 2)) / total
    ed = [100 * float(np.sum(np.abs(_detail(cv, l, k)) ** 2)) / total for k in range(1, l.size - 1)]
    return (ea, _row(ed)) if nargout > 1 else ea


# -- two-dimensional transform --------------------------------------------------------------------------

def _dwt2(X: np.ndarray, lod: np.ndarray, hid: np.ndarray, mode: str):
    lo = _analysis(X, lod, mode, axis=1)
    hi = _analysis(X, hid, mode, axis=1)
    return (_analysis(lo, lod, mode, 0), _analysis(lo, hid, mode, 0), _analysis(hi, lod, mode, 0),
            _analysis(hi, hid, mode, 0))


def _idwt2(a, h, v, d, lor: np.ndarray, hir: np.ndarray, mode: str, size: tuple[int, int] | None):
    parts = [(a, lor, lor), (h, hir, lor), (v, lor, hir), (d, hir, hir)]
    out = None
    for coef, fcol, frow in parts:
        if coef is None or coef.size == 0:
            continue
        r = _synthesis(coef, fcol, mode, size[0] if size else None, axis=0)
        r = _synthesis(r, frow, mode, size[1] if size else None, axis=1)
        out = r if out is None else out + r
    if out is None:
        raise MatlabError("idwt2: at least one coefficient matrix must not be empty.")
    return out


def _image(v, name: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
        raise MatlabError(f"{name}: the input must be a numeric matrix.")
    x = fnum(v)
    if x.ndim != 2:
        raise MatlabError(f"{name}: the input must be a matrix.")
    return x.astype(complex if np.iscomplexobj(x) else float)


@builtin("dwt2")
def _dwt2_b(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "dwt2")
    nargs(rest, 2, 3, "dwt2")
    X = _image(rest[0], "dwt2")
    lod, hid, _ = _filters_from(rest, 1, "dwt2", False)
    out = _dwt2(X, lod, hid, mode)
    return out[:nargout] if nargout > 1 else out[0]


@builtin("idwt2")
def _idwt2_b(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "idwt2")
    nargs(rest, 5, 7, "idwt2")
    coefs = [fnum(c) for c in rest[:4]]
    lor, hir, pos = _filters_from(rest, 4, "idwt2", True)
    size = None
    if len(rest) > pos:
        s = np.real(fnum(rest[pos])).flatten().astype(int)
        size = (int(s[0]), int(s[1]))
    return _idwt2(*coefs, lor, hir, mode, size)


@builtin("wavedec2")
def _wavedec2(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "wavedec2")
    nargs(rest, 3, 4, "wavedec2")
    X = _image(rest[0], "wavedec2")
    n = to_int(rest[1], "level")
    if n < 1:
        raise MatlabError("wavedec2: the level must be a positive integer.")
    lod, hid, _ = _filters_from(rest, 2, "wavedec2", False)
    pieces: list[np.ndarray] = []
    sizes = [X.shape]
    a = X
    for _ in range(n):
        a, h, v, d = _dwt2(a, lod, hid, mode)
        pieces = [h.flatten(order="F"), v.flatten(order="F"), d.flatten(order="F")] + pieces
        sizes.insert(0, h.shape)
    c = np.concatenate([a.flatten(order="F")] + pieces)
    s = np.array([a.shape] + sizes, dtype=float)
    return (_row(c), s) if nargout > 1 else _row(c)


def _book2(args: list, name: str):
    cv = fnum(args[0]).flatten(order="F")
    s = np.real(fnum(args[1])).astype(int)
    if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] < 3:
        raise MatlabError(f"{name}: the bookkeeping matrix must have two columns and at least three rows.")
    need = int(np.prod(s[0]) + 3 * np.sum(np.prod(s[1:-1], axis=1)))
    if need != cv.size:
        raise MatlabError(f"{name}: the bookkeeping matrix does not match the coefficient vector.")
    return cv, s


def _detail2(cv: np.ndarray, s: np.ndarray, level: int):
    nmax = s.shape[0] - 2
    if not 1 <= level <= nmax:
        raise MatlabError(f"The level must be between 1 and {nmax}.")
    k = nmax - level + 1
    first = int(np.prod(s[0]) + 3 * np.sum(np.prod(s[1:k], axis=1)))
    cnt = int(np.prod(s[k]))
    shape = (int(s[k, 0]), int(s[k, 1]))
    return tuple(cv[first + i * cnt:first + (i + 1) * cnt].reshape(shape, order="F") for i in range(3))


def _appcoef2(cv, s, lor, hir, level: int, mode: str) -> np.ndarray:
    nmax = s.shape[0] - 2
    if not 0 <= level <= nmax:
        raise MatlabError(f"The level must be between 0 and {nmax}.")
    a = cv[:int(np.prod(s[0]))].reshape((int(s[0, 0]), int(s[0, 1])), order="F")
    for lev in range(nmax, level, -1):
        h, v, d = _detail2(cv, s, lev)
        nxt = s[nmax - lev + 2]
        a = _idwt2(a, h, v, d, lor, hir, mode, (int(nxt[0]), int(nxt[1])))
    return a


@builtin("waverec2")
def _waverec2(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "waverec2")
    nargs(rest, 3, 4, "waverec2")
    cv, s = _book2(rest, "waverec2")
    lor, hir, _ = _filters_from(rest, 2, "waverec2", True)
    return _appcoef2(cv, s, lor, hir, 0, mode)


@builtin("appcoef2")
def _appcoef2_b(interp, args, nargout):
    rest, mode = _take_mode(interp, list(args), "appcoef2")
    nargs(rest, 3, 5, "appcoef2")
    cv, s = _book2(rest, "appcoef2")
    lor, hir, pos = _filters_from(rest, 2, "appcoef2", True)
    level = to_int(rest[pos], "level") if len(rest) > pos else s.shape[0] - 2
    return _appcoef2(cv, s, lor, hir, level, mode)


@builtin("detcoef2")
def _detcoef2(interp, args, nargout):
    nargs(args, 4, 4, "detcoef2")
    kind = text(args[0]).lower()
    cv, s = _book2(list(args[1:]), "detcoef2")
    h, v, d = _detail2(cv, s, to_int(args[3], "level"))
    if kind in ("h", "v", "d"):
        return {"h": h, "v": v, "d": d}[kind]
    if kind in ("all", "a"):
        return (h, v, d)[:max(nargout, 1)] if nargout > 1 else h
    if kind in ("c", "compact"):
        return _row(np.concatenate([h.flatten(order="F"), v.flatten(order="F"), d.flatten(order="F")]))
    raise MatlabError("detcoef2: the type must be 'h', 'v', 'd', 'all' or 'compact'.")


# -- thresholding and denoising -------------------------------------------------------------------------

def _wthresh(x: np.ndarray, kind: str, t) -> np.ndarray:
    if kind == "s":
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)
    if kind == "h":
        return x * (np.abs(x) > t)
    raise MatlabError("wthresh: the thresholding type must be 's' (soft) or 'h' (hard).")


@builtin("wthresh")
def _wthresh_b(interp, args, nargout):
    nargs(args, 3, 3, "wthresh")
    x = fnum(args[0])
    t = float(np.real(fnum(args[2]).flat[0]))
    if t < 0:
        raise MatlabError("wthresh: the threshold must not be negative.")
    return _wthresh(x, text(args[1]).lower(), t)


def _thselect(x: np.ndarray, rule: str) -> float:
    x = np.real(x).flatten()
    n = x.size
    if n == 0:
        return 0.0
    if rule == "sqtwolog":
        return float(np.sqrt(2 * np.log(n)))
    if rule == "minimaxi":
        return 0.0 if n <= 32 else float(0.3936 + 0.1829 * np.log2(n))
    if rule == "rigrsure":
        sx2 = np.sort(x * x)
        k = np.arange(1, n + 1)
        risks = (n - 2 * k + np.cumsum(sx2) + (n - k) * sx2) / n
        return float(np.sqrt(sx2[int(np.argmin(risks))]))
    if rule == "heursure":
        hthr = np.sqrt(2 * np.log(n))
        eta = (np.sum(x * x) - n) / n
        crit = np.log2(n) ** 1.5 / np.sqrt(n)
        if eta < crit:
            return float(hthr)
        return float(min(_thselect(x, "rigrsure"), hthr))
    raise MatlabError("The threshold rule must be 'rigrsure', 'heursure', 'sqtwolog' or 'minimaxi'.")


@builtin("thselect")
def _thselect_b(interp, args, nargout):
    nargs(args, 2, 2, "thselect")
    return _thselect(fnum(args[0]), text(args[1]).lower())


def _mad_sigma(d: np.ndarray) -> float:
    return float(np.median(np.abs(d)) / 0.6745) if d.size else 0.0


@builtin("wnoisest")
def _wnoisest(interp, args, nargout):
    nargs(args, 2, 3, "wnoisest")
    c, cv, l = _book(list(args), "wnoisest")
    levels = np.real(fnum(args[2])).flatten().astype(int) if len(args) > 2 else np.array([1])
    return _row([_mad_sigma(_detail(cv, l, int(k))) for k in levels])


@builtin("ddencmp")
def _ddencmp(interp, args, nargout):
    nargs(args, 3, 3, "ddencmp")
    kind, what = text(args[0]).lower(), text(args[1]).lower()
    if kind not in ("den", "cmp") or what not in ("wv", "wp"):
        raise MatlabError("ddencmp: expected ddencmp('den' or 'cmp', 'wv' or 'wp', x).")
    x = fnum(args[2])
    lod, hid, _lor, _hir = wavelet_filters("db1")
    if x.ndim == 2 and min(x.shape) == 1:
        det = _analysis(x.flatten(order="F"), hid, "sym")
        n = x.size
    else:
        _a, h, v, d = _dwt2(x.astype(float), lod, hid, "sym")
        det = np.concatenate([h.ravel(), v.ravel(), d.ravel()])
        n = x.size
    med = float(np.median(np.abs(det)))
    if med == 0:
        med = 0.05 * float(np.max(np.abs(det))) if det.size else 0.0
    if kind == "den":
        thr = np.sqrt(2 * np.log(n)) * med / 0.6745
        sorh = "s"
    else:
        thr = med
        sorh = "h"
    out = [float(thr), char(sorh), 1.0]
    if what == "wp":
        out.append(char("sure" if kind == "den" else "threshold"))
    return tuple(out[:nargout]) if nargout > 1 else out[0]


def _denoise_levels(cv: np.ndarray, l: np.ndarray, rule: str, sorh: str, scal: str):
    nmax = l.size - 2
    out = cv.copy()
    if scal == "one":
        sig = np.ones(nmax)
    elif scal == "sln":
        sig = np.full(nmax, _mad_sigma(_detail(cv, l, 1)))
    elif scal == "mln":
        sig = np.array([_mad_sigma(_detail(cv, l, k)) for k in range(1, nmax + 1)])
    else:
        raise MatlabError("wden: the scaling option must be 'one', 'sln' or 'mln'.")
    for k in range(1, nmax + 1):
        pos = nmax - k + 1
        first = int(np.sum(l[:pos]))
        d = cv[first:first + l[pos]]
        s = sig[k - 1]
        if s < np.sqrt(np.finfo(float).eps) * max(float(np.max(np.abs(d))) if d.size else 0.0, 1e-300):
            thr = 0.0
        else:
            thr = _thselect(d / s, rule) * s
        out[first:first + l[pos]] = _wthresh(d, sorh, thr)
    return out


@builtin("wden")
def _wden(interp, args, nargout):
    nargs(args, 6, 7, "wden")
    if is_text(args[2]) and not is_text(args[1]):
        c, cv, l = _book(list(args[:2]), "wden")
        rule, sorh, scal = (text(a).lower() for a in args[2:5])
        n = to_int(args[5], "level")
        _lod, _hid, lor, hir = wavelet_filters(text(args[6]))
        x_like = c
        del n
    else:
        x = _signal(args[0], "wden")
        rule, sorh, scal = (text(a).lower() for a in args[1:4])
        n = to_int(args[4], "level")
        lod, hid, lor, hir = wavelet_filters(text(args[5]))
        cv, l = _wavedec(np.real(x).flatten(order="F").astype(float), n, lod, hid, "sym")
        l = l.astype(int)
        x_like = x
    cxd = _denoise_levels(cv, l, rule, sorh, scal)
    xd = _appcoef(cxd, l, lor, hir, 0, "sym")
    out = (_like(xd, x_like), _like(cxd, x_like), _like(l.astype(float), x_like))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("wdencmp")
def _wdencmp(interp, args, nargout):
    nargs(args, 6, 7, "wdencmp")
    opt = text(args[0]).lower()
    if opt not in ("gbl", "lvd"):
        raise MatlabError("wdencmp: the first input must be 'gbl' or 'lvd'.")
    x = fnum(args[1])
    lod, hid, lor, hir = wavelet_filters(text(args[2]))
    n = to_int(args[3], "level")
    thr = np.real(fnum(args[4])).astype(float)
    sorh = text(args[5]).lower()
    keepapp = bool(fnum(args[6]).flat[0]) if len(args) > 6 else True
    one_d = x.ndim == 2 and min(x.shape) == 1
    if one_d:
        cv, l = _wavedec(np.real(x).flatten(order="F").astype(float), n, lod, hid, "sym")
        l = l.astype(int)
        napp = int(l[0])
        out = cv.copy()
        if opt == "gbl":
            t = float(thr.flat[0])
            if keepapp:
                out[napp:] = _wthresh(cv[napp:], sorh, t)
            else:
                out = _wthresh(cv, sorh, t)
        else:
            tv = thr.flatten()
            if tv.size != n:
                raise MatlabError("wdencmp: give one threshold per level.")
            for k in range(1, n + 1):
                pos = n - k + 1
                first = int(np.sum(l[:pos]))
                out[first:first + l[pos]] = _wthresh(cv[first:first + l[pos]], sorh, tv[k - 1])
        xc = _like(_appcoef(out, l, lor, hir, 0, "sym"), x)
        book = _like(l.astype(float), x)
        cout = _like(out, x)
    else:
        pieces = []
        sizes = [x.shape]
        a = x.astype(float)
        for _ in range(n):
            a, h, v, d = _dwt2(a, lod, hid, "sym")
            pieces.insert(0, (h, v, d))
            sizes.insert(0, h.shape)
        if opt == "gbl":
            t = float(thr.flat[0])
            pieces = [tuple(_wthresh(p, sorh, t) for p in trio) for trio in pieces]
            if not keepapp:
                a = _wthresh(a, sorh, t)
        else:
            if thr.shape != (3, n):
                raise MatlabError("wdencmp: for images the level thresholds must be a 3-by-N matrix.")
            pieces = [tuple(_wthresh(p, sorh, thr[i, n - 1 - j]) for i, p in enumerate(trio))
                      for j, trio in enumerate(pieces)]
        flat = [a.flatten(order="F")] + [p.flatten(order="F") for trio in pieces for p in trio]
        out = np.concatenate(flat)
        cv = None
        s = np.array([a.shape] + sizes)
        xc = _appcoef2(out, s, lor, hir, 0, "sym")
        book = s.astype(float)
        cout = _row(out)
    if nargout > 3:
        if cv is None:
            a0 = x.astype(float)
            orig = []
            for _ in range(n):
                a0, h, v, d = _dwt2(a0, lod, hid, "sym")
                orig = [h.ravel(), v.ravel(), d.ravel()] + orig
            cv = np.concatenate([a0.ravel()] + orig)
        perf0 = 100 * float(np.sum(out == 0)) / out.size
        e0 = float(np.sum(cv ** 2))
        perfl2 = 100 * float(np.sum(out ** 2)) / e0 if e0 else 100.0
        return (xc, cout, book, perf0, perfl2)[:nargout]
    res = (xc, cout, book)
    return res[:nargout] if nargout > 1 else xc


# empirical-Bayes thresholding (Johnstone and Silverman): a point mass at zero mixed with a heavy-tailed
# "quasi-Cauchy" density, the weight chosen by marginal maximum likelihood, the posterior median as estimate

def _qc_prior(u: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    a = np.abs(u)
    return (1 - a * norm.sf(a) / np.maximum(norm.pdf(a), 1e-300)) / np.sqrt(2 * np.pi)


def _qc_beta(x: np.ndarray) -> np.ndarray:
    """g(x)/phi(x) - 1 for the quasi-Cauchy marginal density g."""
    a = np.minimum(np.abs(x), 35.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b = np.expm1(a * a / 2) / (a * a) - 1
    return np.where(a < 1e-4, -0.5 + a * a / 8, b)


def _eb_weight(z: np.ndarray) -> float:
    n = z.size
    beta = _qc_beta(z)

    def score(w):
        return float(np.sum(beta / (1 + w * beta)))

    lo = max(1.0 / n, 1e-10)
    if score(1.0) >= 0:
        return 1.0
    if score(lo) <= 0:
        return lo
    hi = 1.0
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _eb_estimate(z: np.ndarray, w: float, rule: str) -> np.ndarray:
    """Posterior median (or mean) of the signal under the mixture prior, for unit-variance data."""
    from scipy.stats import norm
    zmax = float(np.max(np.abs(z))) if z.size else 0.0
    grid = np.linspace(0.0, max(zmax, 1.0) + 1e-9, 600)
    mu = np.linspace(-12.0, max(zmax, 1.0) + 12.0, 6001)
    dm = mu[1] - mu[0]
    prior = _qc_prior(mu)
    like = norm.pdf(grid[:, None] - mu[None, :])
    dens = w * prior[None, :] * like                       # continuous part of the posterior
    cum = np.cumsum(dens, axis=1) * dm
    slab = cum[:, -1]
    spike = (1 - w) * norm.pdf(grid)
    total = slab + spike
    est = np.zeros(grid.size)
    if rule == "mean":
        est = (dens * mu[None, :]).sum(axis=1) * dm / total
    else:
        zero = int(np.searchsorted(mu, 0.0))
        below = cum[:, zero] / total                       # P(mu < 0 | x)
        upto = below + spike / total                       # P(mu <= 0 | x)
        for i in range(grid.size):
            if upto[i] >= 0.5:
                est[i] = 0.0 if below[i] <= 0.5 else mu[int(np.searchsorted(cum[i] / total[i], 0.5))]
            else:
                target = 0.5 * total[i] - spike[i]
                j = int(np.searchsorted(cum[i], target))
                j = min(max(j, 1), mu.size - 1)
                c0, c1 = cum[i, j - 1], cum[i, j]
                est[i] = mu[j - 1] + (target - c0) / (c1 - c0) * dm if c1 > c0 else mu[j]
        est = np.maximum(est, 0.0)
    out = np.interp(np.abs(z), grid, est)
    if rule in ("soft", "hard"):
        # the threshold is where the posterior median leaves zero
        nz = np.flatnonzero(est > 0)
        thr = grid[nz[0]] if nz.size else grid[-1]
        return _wthresh(z, "s" if rule == "soft" else "h", thr)
    return np.sign(z) * out


_WD_OPTS = ("Wavelet", "DenoisingMethod", "ThresholdRule", "NoiseEstimate")


@builtin("wdenoise")
def _wdenoise(interp, args, nargout):
    nargs(args, 1, None, "wdenoise")
    x = fnum(args[0])
    if np.iscomplexobj(x) or x.ndim != 2:
        raise MatlabError("wdenoise: the data must be a real vector or matrix.")
    pos = 1
    level = None
    if len(args) > 1 and not is_text(args[1]):
        level = to_int(args[1], "level")
        pos = 2
    opts: dict = {}
    while pos < len(args):
        key = text(args[pos]).lower()
        if key not in [o.lower() for o in _WD_OPTS] or pos + 1 >= len(args):
            raise MatlabError(f"wdenoise: unknown or incomplete option '{text(args[pos])}'.")
        opts[key] = args[pos + 1]
        pos += 2
    wname = text(opts["wavelet"]) if "wavelet" in opts else "sym4"
    method = text(opts["denoisingmethod"]).lower() if "denoisingmethod" in opts else "bayes"
    if method not in ("bayes", "universalthreshold", "sure", "minimax"):
        raise MatlabError(f"wdenoise: the denoising method '{method}' is not supported. Use 'Bayes', "
                          "'UniversalThreshold', 'SURE' or 'Minimax'.")
    rule = text(opts["thresholdrule"]).lower() if "thresholdrule" in opts else ("median" if method == "bayes"
                                                                               else "soft")
    if method == "bayes" and rule not in ("median", "mean", "soft", "hard"):
        raise MatlabError("wdenoise: with 'Bayes' the threshold rule must be 'Median', 'Mean', 'Soft' or 'Hard'.")
    if method != "bayes" and rule not in ("soft", "hard"):
        raise MatlabError("wdenoise: the threshold rule must be 'Soft' or 'Hard'.")
    noise = text(opts["noiseestimate"]).lower() if "noiseestimate" in opts else "levelindependent"
    if noise not in ("levelindependent", "leveldependent"):
        raise MatlabError("wdenoise: 'NoiseEstimate' must be 'LevelIndependent' or 'LevelDependent'.")
    lod, hid, lor, hir = wavelet_filters(wname)
    row = x.shape[0] == 1
    cols = x.reshape(-1, 1) if row else x
    n = cols.shape[0]
    maxlev = int(np.floor(np.log2(n / (lod.size - 1)))) if lod.size > 1 and n >= lod.size - 1 else 0
    if level is None:
        level = max(min(int(np.floor(np.log2(n))), maxlev), 1)
    if level < 1:
        raise MatlabError("wdenoise: the level must be a positive integer.")
    outs, den_c, orig_c = [], [], []
    for j in range(cols.shape[1]):
        cv, l = _wavedec(cols[:, j].astype(float), level, lod, hid, "per")
        l = l.astype(int)
        new = cv.copy()
        sig1 = _mad_sigma(_detail(cv, l, 1))
        for k in range(1, level + 1):
            p = level - k + 1
            first = int(np.sum(l[:p]))
            d = cv[first:first + l[p]]
            sig = _mad_sigma(d) if noise == "leveldependent" else sig1
            if sig <= 0:
                continue
            z = d / sig
            if method == "bayes":
                est = _eb_estimate(z, _eb_weight(z), rule)
            else:
                name = {"universalthreshold": "sqtwolog", "sure": "rigrsure", "minimax": "minimaxi"}[method]
                thr = np.sqrt(2 * np.log(n)) if method == "universalthreshold" else _thselect(z, name)
                est = _wthresh(z, "s" if rule == "soft" else "h", thr)
            new[first:first + l[p]] = est * sig
        outs.append(_appcoef(new, l, lor, hir, 0, "per")[:n])
        den_c.append(new)
        orig_c.append(cv)
    y = np.stack(outs, axis=1)
    y = y.reshape(1, -1) if row else y
    if nargout > 1:
        def pack(cs):
            if len(cs) == 1:
                return _like(cs[0], x)
            return Cell.from_list([_col(c) for c in cs])
        return (y, pack(den_c), pack(orig_c))[:nargout]
    return y


# -- continuous wavelet transform -----------------------------------------------------------------------

def _morse_props(gamma: float, beta: float):
    """Peak frequency, 50% cutoff frequency and time spread of the Morse wavelet w^beta exp(-w^gamma)."""
    from scipy.special import gammaln
    wpk = (beta / gamma) ** (1 / gamma)

    def psi(w):
        return np.exp(beta * np.log(w) - w ** gamma - (beta * np.log(wpk) - wpk ** gamma))

    lo, hi = wpk, wpk * 2
    while psi(hi) > 0.5:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if psi(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    cutoff = 0.5 * (lo + hi)

    def lint(a):      # log of the integral of w^a exp(-2 w^gamma) over w > 0
        return -np.log(gamma) - (a + 1) / gamma * np.log(2) + gammaln((a + 1) / gamma)

    num = (beta ** 2 * np.exp(lint(2 * beta - 2) - lint(2 * beta))
           - 2 * beta * gamma * np.exp(lint(2 * beta + gamma - 2) - lint(2 * beta))
           + gamma ** 2 * np.exp(lint(2 * beta + 2 * gamma - 2) - lint(2 * beta)))
    sigma_t = float(np.sqrt(max(num, 0.0)))
    return wpk, cutoff, sigma_t


def _cwt_wavelet(name: str, gamma: float, p2: float):
    """Returns (psi_hat(s*w) evaluator, peak frequency, cutoff frequency, time spread)."""
    if name == "morse":
        beta = p2 / gamma
        wpk, cutoff, sig = _morse_props(gamma, beta)
        logpk = beta * np.log(wpk) - wpk ** gamma

        def fn(sw):
            out = np.zeros(sw.shape)
            pos = sw > 0
            out[pos] = 2 * np.exp(beta * np.log(sw[pos]) - sw[pos] ** gamma - logpk)
            return out
        return fn, wpk, cutoff, sig
    if name == "amor":
        w0 = 6.0

        def fn(sw):
            return np.where(sw > 0, 2 * np.exp(-0.5 * (sw - w0) ** 2), 0.0)
        cutoff = w0 + np.sqrt(2 * np.log(10))
        return fn, w0, cutoff, 1.0 / np.sqrt(2)
    if name == "bump":
        mu, sg = 5.0, 0.6

        def fn(sw):
            u = (sw - mu) / sg
            out = np.zeros(sw.shape)
            ok = np.abs(u) < 1
            out[ok] = 2 * np.exp(1 - 1 / (1 - u[ok] ** 2))
            return out
        cutoff = mu + sg * np.sqrt(1 - 1 / (1 + np.log(10)))
        return fn, mu, cutoff, 5.847705
    raise MatlabError(f"cwt: the wavelet '{name}' is not available. Use 'morse', 'amor' or 'bump'.")


_CWT_OPTS = ("WaveletParameters", "TimeBandwidth", "VoicesPerOctave", "ExtendSignal", "FrequencyLimits",
             "NumOctaves")


@builtin("cwt")
def _cwt(interp, args, nargout):
    nargs(args, 1, None, "cwt")
    x = _signal(args[0], "cwt")
    v = x.flatten(order="F")
    n = v.size
    if n < 4:
        raise MatlabError("cwt: the signal must have at least four samples.")
    rest = list(args[1:])
    wname = "morse"
    fs = None
    opts: dict = {}
    i = 0
    while i < len(rest):
        a = rest[i]
        if is_text(a):
            s = text(a)
            if s.lower() in ("morse", "amor", "bump"):
                wname = s.lower()
                i += 1
                continue
            if s.lower() not in [o.lower() for o in _CWT_OPTS] or i + 1 >= len(rest):
                raise MatlabError(f"cwt: unknown or incomplete option '{s}'.")
            opts[s.lower()] = rest[i + 1]
            i += 2
            continue
        if fs is not None:
            raise MatlabError("cwt: unexpected numeric argument.")
        fs = float(np.real(fnum(a).flat[0]))
        if fs <= 0:
            raise MatlabError("cwt: the sample rate must be positive.")
        i += 1
    gamma, p2 = 3.0, 60.0
    if "waveletparameters" in opts:
        wp = np.real(fnum(opts["waveletparameters"])).flatten()
        gamma, p2 = float(wp[0]), float(wp[1])
    if "timebandwidth" in opts:
        p2 = float(np.real(fnum(opts["timebandwidth"]).flat[0]))
    nv = to_int(opts["voicesperoctave"], "VoicesPerOctave") if "voicesperoctave" in opts else 10
    if not 1 <= nv <= 48:
        raise MatlabError("cwt: 'VoicesPerOctave' must be between 1 and 48.")
    extend = True
    if "extendsignal" in opts:
        e = opts["extendsignal"]
        extend = text(e).lower() in ("true", "on") if is_text(e) else bool(np.ravel(e)[0])
    psi, wpk, cutoff, sig_t = _cwt_wavelet(wname, gamma, p2)
    # scales: from the one whose spectrum has fallen to the cutoff level (50% of its peak for Morse
    # wavelets, 10% for the others) at the Nyquist frequency up to the one whose
    # wavelet (two standard deviations wide) spans the signal
    s0 = cutoff / np.pi
    smax = n / (2 * sig_t)
    if smax < s0 * 2 ** (1 / nv):
        smax = s0 * 2 ** (1 / nv)
    octaves = max(np.log2(smax / s0), 1 / nv)
    if "numoctaves" in opts:
        octaves = min(octaves, float(to_int(opts["numoctaves"], "NumOctaves")))
    scales = s0 * 2 ** (np.arange(int(np.floor(nv * octaves)) + 1) / nv)
    fsv = fs if fs is not None else 1.0
    freqs = wpk / scales / (2 * np.pi) * fsv
    if "frequencylimits" in opts:
        lim = np.real(fnum(opts["frequencylimits"])).flatten()
        keep = (freqs >= lim[0] - 1e-12) & (freqs <= lim[1] + 1e-12)
        if not np.any(keep):
            raise MatlabError("cwt: no wavelet frequencies fall inside 'FrequencyLimits'.")
        scales, freqs = scales[keep], freqs[keep]
    xm = v - np.mean(v)
    if extend:
        pad = n // 2
        xe = np.concatenate([np.conj(xm[pad - 1::-1]) if pad else xm[:0], xm, np.conj(xm[:n - pad - 1:-1])
                             if pad else xm[:0]])
    else:
        pad = 0
        xe = xm
    m = xe.size
    omega = 2 * np.pi * np.fft.fftfreq(m)
    X = np.fft.fft(xe)
    W = np.fft.ifft(X[None, :] * psi(scales[:, None] * omega[None, :]), axis=1)
    wt = W[:, pad:pad + n]
    if nargout == 0:
        _plot = interp.call_named
        t = np.arange(n) / fsv
        _plot("imagesc", [_row(t), _row(np.arange(freqs.size)), np.abs(wt)], 0, interp.frame)
        _plot("xlabel", [char("Time (s)" if fs is not None else "Samples")], 0, interp.frame)
        _plot("ylabel", [char("Wavelet scale index (high to low frequency)")], 0, interp.frame)
        _plot("colorbar", [], 0, interp.frame)
        return None
    out = [wt.astype(complex), _col(freqs)]
    if nargout > 2:
        samples = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)
        out.append(_col(sig_t * wpk / (2 * np.pi) * fsv / samples))
    return tuple(out[:nargout]) if nargout > 1 else out[0]


# -- maximal-overlap transform ----------------------------------------------------------------------------

def _modwt_filters(args: list, pos: int, n: int, name: str):
    wname = "sym4"
    level = None
    rest = [a for a in args[pos:]]
    filt = None
    for a in rest:
        if is_text(a):
            s = text(a)
            if s.lower() in ("reflection", "periodic"):
                if s.lower() == "reflection":
                    raise MatlabError(f"{name}: the 'reflection' boundary is not supported; the transform is "
                                      "periodic.")
                continue
            wname = s
        elif filt is None and isinstance(a, np.ndarray) and a.size > 1:
            filt = np.real(fnum(a)).flatten()
        else:
            level = to_int(a, "level")
    _lod, _hid, lor, hir = wavelet_filters(wname)
    return lor / np.sqrt(2), hir / np.sqrt(2), level


@builtin("modwt")
def _modwt(interp, args, nargout):
    nargs(args, 1, 4, "modwt")
    x = _signal(args[0], "modwt")
    v = np.real(x).flatten(order="F").astype(float)
    n = v.size
    if n < 2:
        raise MatlabError("modwt: the signal must have at least two samples.")
    g, h, level = _modwt_filters(list(args), 1, n, "modwt")
    if level is None:
        level = int(np.floor(np.log2(n)))
    if level < 1 or level > int(np.floor(np.log2(n))):
        raise MatlabError(f"modwt: the level must be between 1 and {int(np.floor(np.log2(n)))}.")
    k = np.arange(n)
    G = np.fft.fft(np.bincount(np.arange(g.size) % n, weights=g, minlength=n))
    H = np.fft.fft(np.bincount(np.arange(h.size) % n, weights=h, minlength=n))
    V = np.fft.fft(v)
    rows = []
    for j in range(1, level + 1):
        idx = (2 ** (j - 1) * k) % n
        rows.append(np.real(np.fft.ifft(H[idx] * V)))
        V = G[idx] * V
    rows.append(np.real(np.fft.ifft(V)))
    return np.array(rows)


@builtin("imodwt")
def _imodwt(interp, args, nargout):
    nargs(args, 1, 4, "imodwt")
    w = np.real(fnum(args[0])).astype(float)
    if w.ndim != 2 or w.shape[0] < 2:
        raise MatlabError("imodwt: the input must be the matrix returned by modwt.")
    n = w.shape[1]
    g, h, _level = _modwt_filters([a for a in args if is_text(a) or a is args[0]], 1, n, "imodwt")
    lowest = 0
    for a in args[1:]:
        if not is_text(a):
            lowest = to_int(a, "level")
    k = np.arange(n)
    G = np.fft.fft(np.bincount(np.arange(g.size) % n, weights=g, minlength=n))
    H = np.fft.fft(np.bincount(np.arange(h.size) % n, weights=h, minlength=n))
    J = w.shape[0] - 1
    V = np.fft.fft(w[J])
    for j in range(J, lowest, -1):
        idx = (2 ** (j - 1) * k) % n
        V = np.conj(G[idx]) * V + np.conj(H[idx]) * np.fft.fft(w[j - 1])
    return _row(np.real(np.fft.ifft(V)))


@builtin("modwtmra")
def _modwtmra(interp, args, nargout):
    nargs(args, 1, 3, "modwtmra")
    w = np.real(fnum(args[0])).astype(float)
    if w.ndim != 2 or w.shape[0] < 2:
        raise MatlabError("modwtmra: the input must be the matrix returned by modwt.")
    n = w.shape[1]
    g, h, _level = _modwt_filters(list(args), 1, n, "modwtmra")
    k = np.arange(n)
    G = np.fft.fft(np.bincount(np.arange(g.size) % n, weights=g, minlength=n))
    H = np.fft.fft(np.bincount(np.arange(h.size) % n, weights=h, minlength=n))
    J = w.shape[0] - 1
    out = np.zeros_like(w)
    for row in range(J + 1):
        if row < J:      # detail of level row+1: go up through the wavelet filter once, then scaling filters
            lev = row + 1
            V = np.conj(H[(2 ** (lev - 1) * k) % n]) * np.fft.fft(w[row])
            start = lev - 1
        else:
            V = np.fft.fft(w[J])
            start = J
        for j in range(start, 0, -1):
            V = np.conj(G[(2 ** (j - 1) * k) % n]) * V
        out[row] = np.real(np.fft.ifft(V))
    return out
