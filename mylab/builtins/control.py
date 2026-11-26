"""Control-system models: tf, zpk, ss, frd, pid and pidstd objects.

The models are value objects. Transfer functions keep numerator and denominator polynomials,
zero-pole-gain models keep roots, state-space models keep matrices and frequency-response
models keep samples. Arithmetic (``+ - * / \\ ^``), concatenation, indexing and the interconnection
functions (series, parallel, feedback, append, connect, lft) keep the type with the highest
precedence: frd, then ss, zpk, tf and pid.

The analysis functions are in :mod:`control_analysis` and the design functions in
:mod:`control_design`; all of them register in :data:`LTI_FUNCS`, which is also the method table
of the objects (so ``step(sys)`` reaches the control-system function even though ``step`` is
a built-in with another meaning for other arguments).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, StringArray, Struct, char, is_char, to_matlab, to_str
from . import _ctrl_num as cn
from ._registry import BUILTINS
from ._util import bool_scalar, is_text, text

LTI_FUNCS: dict[str, Callable] = {}

PADE_HINT = ("Replace the delay with a rational approximation first, for example "
             "pade(sys, 3), or work in discrete time where delays are extra poles at z = 0.")


def ltifn(*names: str, builtin: bool = True):
    """Register a function for model arguments; also as a built-in when the name is free."""

    def deco(fn: Callable) -> Callable:
        for n in names:
            LTI_FUNCS[n] = fn
            if builtin and n not in BUILTINS:
                BUILTINS[n] = fn
        return fn

    return deco


# -- small argument helpers -------------------------------------------------------------------

def is_num(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind in "biufc"


def as_matrix(v: Any, what: str = "argument") -> np.ndarray:
    if hasattr(v, "to_dense"):
        v = v.to_dense()
    if not is_num(v):
        raise MatlabError(f"The {what} must be numeric.")
    a = np.asarray(v)
    if a.dtype.kind in "biu":
        a = a.astype(float)
    if a.ndim > 2:
        raise MatlabError(f"The {what} must be a matrix.")
    return np.atleast_2d(a)


def as_vector(v: Any, what: str = "argument") -> np.ndarray:
    a = as_matrix(v, what)
    return a.ravel(order="F")


def as_scalar(v: Any, what: str = "argument") -> float:
    a = as_matrix(v, what)
    if a.size != 1:
        raise MatlabError(f"The {what} must be a scalar.")
    x = a.flat[0]
    return float(np.real(x))


def is_empty_num(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.size == 0 and v.dtype.kind != "U"


def row(v) -> np.ndarray:
    return np.asarray(v).reshape(1, -1)


def col(v) -> np.ndarray:
    return np.asarray(v).reshape(-1, 1)


def names_from(v: Any) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) if not is_empty_num(x) else "" for x in v.values_f()]
    if isinstance(v, StringArray):
        return [x or "" for x in v.values_f()]
    if is_empty_num(v):
        return []
    if is_char(v) and v.ndim == 2 and v.shape[0] > 1:
        return ["".join(r).rstrip() for r in v.tolist()]
    return [text(v)]


def names_cell(names: list[str]) -> Cell:
    return Cell.from_list([char(n) for n in names], column=True)


def fmt_num(x: float, digits: int = 4) -> str:
    """Number formatting of model displays: four significant digits, compact exponents."""
    if isinstance(x, complex) or np.iscomplexobj(x):
        x = complex(x)
        if x.imag == 0:
            return fmt_num(x.real, digits)
        sign = "+" if x.imag >= 0 else "-"
        return f"({fmt_num(x.real, digits)}{sign}{fmt_num(abs(x.imag), digits)}i)"
    x = float(x)
    if x == 0:
        return "0"
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    s = f"{x:.{digits}g}"
    if "e" in s:
        mant, exp = s.split("e")
        s = mant + "e" + ("-" if exp[0] == "-" else "") + exp[1:].rjust(2, "0")
    return s


# -- the base class ---------------------------------------------------------------------------

_COMMON_PROPS = ["InputDelay", "OutputDelay", "Ts", "TimeUnit", "InputName", "InputUnit", "InputGroup",
                 "OutputName", "OutputUnit", "OutputGroup", "Notes", "UserData", "Name", "SamplingGrid"]
_ALIASES = {"num": "Numerator", "den": "Denominator", "u": "InputName", "y": "OutputName",
            "ioDelay": "IODelay", "inputdelay": "InputDelay", "outputdelay": "OutputDelay",
            "Units": "FrequencyUnit", "Response": "ResponseData"}


class LTI(MObject):
    """Common part of all model objects: sample time, delays, channel names."""

    kind = "lti"
    rank = 0
    PROPS: list[str] = []

    def _init_common(self, ny: int, nu: int, ts: float = 0.0) -> None:
        self.ts = float(ts)
        self.in_delay = np.zeros(nu)
        self.out_delay = np.zeros(ny)
        self.io_delay = np.zeros((ny, nu))
        self.in_name = [""] * nu
        self.out_name = [""] * ny
        self.in_unit = [""] * nu
        self.out_unit = [""] * ny
        self.in_group: dict[str, np.ndarray] = {}
        self.out_group: dict[str, np.ndarray] = {}
        self.name = ""
        self.notes: Any = None
        self.userdata: Any = None
        self.time_unit = "seconds"

    # -- identity -----------------------------------------------------------------------------
    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        return name in (self.kind, "lti", "DynamicSystem", "InputOutputModel", "numlti") or \
            (name == "genfrd" and self.kind == "frd")

    @property
    def ny(self) -> int:
        raise NotImplementedError

    @property
    def nu(self) -> int:
        raise NotImplementedError

    def matlab_shape(self) -> tuple:
        return (self.ny, self.nu)

    def clone(self) -> "LTI":
        return copy.copy(self)

    @property
    def is_siso(self) -> bool:
        return self.ny == 1 and self.nu == 1

    @property
    def is_dt(self) -> bool:
        return self.ts != 0

    def tsv(self) -> float:
        """The sample time used in formulas (1 when unspecified)."""
        return self.ts if self.ts > 0 else 1.0

    def has_delay(self) -> bool:
        return bool(np.any(self.in_delay) or np.any(self.out_delay) or np.any(self.io_delay))

    def total_delay(self) -> np.ndarray:
        return self.out_delay[:, None] + self.io_delay + self.in_delay[None, :]

    def is_static(self) -> bool:
        return False

    def copy_meta(self, other: "LTI", inputs: bool = True, outputs: bool = True) -> "LTI":
        """Take channel names, units and the general information from ``other``."""
        if inputs and other.nu == self.nu:
            self.in_name, self.in_unit, self.in_group = list(other.in_name), list(other.in_unit), dict(other.in_group)
        if outputs and other.ny == self.ny:
            self.out_name, self.out_unit = list(other.out_name), list(other.out_unit)
            self.out_group = dict(other.out_group)
        self.time_unit = other.time_unit
        return self

    def copy_delays(self, other: "LTI") -> "LTI":
        self.in_delay, self.out_delay, self.io_delay = other.in_delay, other.out_delay, other.io_delay
        return self

    def object_equals(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False
        from ..values import values_equal

        def same(a, b):
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a, b = np.asarray(a), np.asarray(b)
                return a.shape == b.shape and bool(np.all(a == b))
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
            if isinstance(a, (str, float, int, bool)) or a is None:
                return a == b
            return values_equal(a, b)

        return self.__dict__.keys() == other.__dict__.keys() and \
            all(same(v, other.__dict__[k]) for k, v in self.__dict__.items())

    # -- methods ------------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        return name in LTI_FUNCS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        res = LTI_FUNCS[name](interp, list(args), nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    # -- properties ---------------------------------------------------------------------------
    def prop_names(self) -> list[str]:
        return self.PROPS + _COMMON_PROPS

    def resolve_prop(self, name: str) -> str:
        names = self.prop_names()
        if name in names:
            return name
        low = name.lower()
        for alias, target in _ALIASES.items():
            if low == alias.lower() and target in names:
                return target
        exact = [n for n in names if n.lower() == low]
        if exact:
            return exact[0]
        part = [n for n in names if n.lower().startswith(low)]
        if len(part) == 1:
            return part[0]
        if len(part) > 1:
            raise MatlabError(f"The property name '{name}' is ambiguous for a {self.kind} model: it matches "
                              + ", ".join(part) + ".")
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.",
                          "MATLAB:noSuchMethodOrField")

    def get_field(self, interp, name: str) -> Any:
        try:
            prop = self.resolve_prop(name)
        except MatlabError:
            if name in LTI_FUNCS:
                res = self.call_method(interp, name, [self], 1)
                if res:
                    return res[0]
            raise
        return to_matlab(self.get_prop(prop))

    def set_field(self, interp, name: str, value: Any) -> "LTI":
        out = self.clone()
        out.set_prop(out.resolve_prop(name), value)
        return out

    def get_prop(self, prop: str) -> Any:
        if prop == "Ts":
            return float(self.ts)
        if prop == "InputDelay":
            return col(self.in_delay.astype(float))
        if prop == "OutputDelay":
            return col(self.out_delay.astype(float))
        if prop == "IODelay":
            return self.io_delay.astype(float)
        if prop == "TimeUnit":
            return char(self.time_unit)
        if prop == "InputName":
            return names_cell(self.in_name)
        if prop == "OutputName":
            return names_cell(self.out_name)
        if prop == "InputUnit":
            return names_cell(self.in_unit)
        if prop == "OutputUnit":
            return names_cell(self.out_unit)
        if prop in ("InputGroup", "OutputGroup"):
            g = self.in_group if prop == "InputGroup" else self.out_group
            return Struct.scalar({k: row(np.asarray(v, dtype=float) + 1) for k, v in g.items()})
        if prop == "Notes":
            return self.notes if self.notes is not None else Cell(np.empty((0, 0), dtype=object))
        if prop == "UserData":
            return self.userdata if self.userdata is not None else np.zeros((0, 0))
        if prop == "Name":
            return char(self.name)
        if prop == "SamplingGrid":
            return Struct.scalar({})
        raise MatlabError(f"Unrecognized property '{prop}' for class '{self.kind}'.")

    def set_prop(self, prop: str, value: Any) -> None:
        if prop == "Ts":
            ts = as_scalar(value, "sample time Ts")
            if ts < 0 and ts != -1:
                raise MatlabError("The sample time Ts must be 0 (continuous), positive, or -1 (unspecified).")
            self._set_ts(ts)
        elif prop == "InputDelay":
            self.in_delay = self._delay_vec(value, self.nu, "InputDelay")
        elif prop == "OutputDelay":
            self.out_delay = self._delay_vec(value, self.ny, "OutputDelay")
        elif prop == "IODelay":
            d = as_matrix(value, "IODelay").real.astype(float)
            if d.size == 1:
                d = np.full((self.ny, self.nu), float(d.flat[0]))
            if d.shape != (self.ny, self.nu):
                raise MatlabError(f"IODelay must be a {self.ny}-by-{self.nu} matrix (one entry per "
                                  "output/input pair) or a scalar.")
            self._check_delay(d, "IODelay")
            self.io_delay = d
        elif prop == "TimeUnit":
            unit = text(value)
            allowed = ("nanoseconds", "microseconds", "milliseconds", "seconds", "minutes", "hours", "days",
                       "weeks", "months", "years")
            if unit not in allowed:
                raise MatlabError("TimeUnit must be one of: " + ", ".join(allowed) + ".")
            self.time_unit = unit
        elif prop in ("InputName", "OutputName", "InputUnit", "OutputUnit"):
            n = self.nu if prop.startswith("Input") else self.ny
            names = names_from(value)
            if len(names) == 1 and n > 1 and prop.endswith("Name") and names[0]:
                names = [f"{names[0]}({k + 1})" for k in range(n)]
            elif len(names) == 1 and n > 1:
                names = names * n
            elif not names:
                names = [""] * n
            if len(names) != n:
                kind = "inputs" if prop.startswith("Input") else "outputs"
                raise MatlabError(f"{prop} needs {n} name(s), one for each of the {n} {kind}; got {len(names)}.")
            attr = {"InputName": "in_name", "OutputName": "out_name", "InputUnit": "in_unit",
                    "OutputUnit": "out_unit"}[prop]
            setattr(self, attr, names)
        elif prop in ("InputGroup", "OutputGroup"):
            if not isinstance(value, Struct):
                raise MatlabError(f"{prop} must be a struct whose fields are group names holding channel indices.")
            n = self.nu if prop == "InputGroup" else self.ny
            groups = {}
            for f in value.fields:
                idx = as_vector(value.get_field(0, f), f"group {f}").real.astype(int) - 1
                if idx.size and (idx.min() < 0 or idx.max() >= n):
                    raise MatlabError(f"Group '{f}' refers to a channel outside 1..{n}.")
                groups[f] = idx
            if prop == "InputGroup":
                self.in_group = groups
            else:
                self.out_group = groups
        elif prop == "Notes":
            self.notes = value
        elif prop == "UserData":
            self.userdata = value
        elif prop == "Name":
            self.name = text(value)
        elif prop == "SamplingGrid":
            if not isinstance(value, Struct) or value.fields:
                raise MatlabError("SamplingGrid is only supported as an empty struct (model arrays are not "
                                  "available).")
        else:
            raise MatlabError(f"The property '{prop}' of a {self.kind} model cannot be set.")

    def _set_ts(self, ts: float) -> None:
        was_dt = self.is_dt
        self.ts = float(ts)
        if was_dt != self.is_dt and hasattr(self, "var"):
            self.var = "z" if self.is_dt else "s"
        self._check_all_delays()

    def _check_delay(self, d: np.ndarray, what: str) -> None:
        if np.any(~np.isfinite(d)) or np.any(d < 0):
            raise MatlabError(f"{what} must be non-negative and finite.")
        if self.is_dt and np.any(np.abs(d - np.round(d)) > 0):
            raise MatlabError(f"{what} of a discrete-time model is counted in sample periods and must be "
                              "a whole number.")

    def _check_all_delays(self) -> None:
        for d, what in ((self.in_delay, "InputDelay"), (self.out_delay, "OutputDelay"), (self.io_delay, "IODelay")):
            self._check_delay(d, what)

    def _delay_vec(self, value: Any, n: int, what: str) -> np.ndarray:
        d = as_vector(value, what).real.astype(float)
        if d.size == 1 and n != 1:
            d = np.full(n, float(d[0]))
        if d.size != n:
            raise MatlabError(f"{what} must have one entry per channel ({n}) or be a scalar.")
        self._check_delay(d, what)
        return d

    # -- indexing -----------------------------------------------------------------------------
    def chan_index(self, idx: Any, outputs: bool) -> np.ndarray:
        n = self.ny if outputs else self.nu
        names = self.out_name if outputs else self.in_name
        groups = self.out_group if outputs else self.in_group
        if idx is ix.COLON:
            return np.arange(n)
        if is_text(idx) or isinstance(idx, (Cell, StringArray)):
            out: list[int] = []
            for nm in names_from(idx):
                if nm in groups:
                    out.extend(int(k) for k in groups[nm])
                elif nm in names and nm:
                    out.extend(k for k, x in enumerate(names) if x == nm)
                else:
                    what = "output" if outputs else "input"
                    raise MatlabError(f"No {what} channel or group is named '{nm}'.")
            return np.array(out, dtype=int)
        pos, _ = ix._to_positions(idx, n, for_set=False)
        return np.asarray(pos, dtype=int).ravel()

    def index_get(self, interp, args: list) -> Any:
        if len(args) == 1:
            if ix_is_one(args[0]) and self.is_siso:
                return self
            if args[0] is ix.COLON and self.is_siso:
                return self
            raise MatlabError("Use two subscripts to pick outputs and inputs of a model, e.g. sys(1, 2) for "
                              "the response from input 2 to output 1.")
        if len(args) > 2:
            if not all(ix_is_one(a) or a is ix.COLON for a in args[2:]):
                raise MatlabError("Arrays of models are not supported; a model has only output and input "
                                  "dimensions.")
        rows = self.chan_index(args[0], True)
        cols = self.chan_index(args[1], False)
        return self.sub(rows, cols)

    def index_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 2:
            raise MatlabError("Use two subscripts to assign part of a model, e.g. sys(1, 2) = tf(1, [1 1]).")
        rows = self._grow_index(args[0], self.ny)
        cols = self._grow_index(args[1], self.nu)
        ny, nu = max(self.ny, rows.max() + 1 if rows.size else 0), max(self.nu, cols.max() + 1 if cols.size else 0)
        target, val = promote_pair(self, value)
        if val.ny == 1 and val.nu == 1 and (rows.size, cols.size) != (1, 1):
            val = lti_vcat([lti_hcat([val] * cols.size)] * rows.size)
        if (val.ny, val.nu) != (rows.size, cols.size):
            raise MatlabError(f"The right-hand side has {val.ny} output(s) and {val.nu} input(s) but the "
                              f"indexed part has {rows.size} and {cols.size}.")
        return target.assign(rows, cols, val, ny, nu)

    def _grow_index(self, idx: Any, n: int) -> np.ndarray:
        if idx is ix.COLON:
            return np.arange(n)
        if is_text(idx) or isinstance(idx, (Cell, StringArray)):
            return self.chan_index(idx, n == self.ny)
        a = np.asarray(idx)
        if a.dtype == bool:
            return np.flatnonzero(a.ravel())
        pos = np.real(a).astype(int).ravel() - 1
        if pos.size and pos.min() < 0:
            raise MatlabError("Index must be a positive integer.")
        return pos

    def index_delete(self, interp, args: list) -> Any:
        if len(args) != 2:
            raise MatlabError("Use two subscripts to delete channels, e.g. sys(:, 2) = [].")
        if args[0] is ix.COLON:
            drop = set(self.chan_index(args[1], False).tolist())
            return self.sub(np.arange(self.ny), np.array([j for j in range(self.nu) if j not in drop], dtype=int))
        if args[1] is ix.COLON:
            drop = set(self.chan_index(args[0], True).tolist())
            return self.sub(np.array([i for i in range(self.ny) if i not in drop], dtype=int), np.arange(self.nu))
        raise MatlabError("Only whole rows (outputs) or whole columns (inputs) of a model can be deleted.")

    def sub_meta(self, src: "LTI", rows: np.ndarray, cols: np.ndarray) -> "LTI":
        self.in_delay = src.in_delay[cols]
        self.out_delay = src.out_delay[rows]
        self.io_delay = src.io_delay[np.ix_(rows, cols)]
        self.in_name = [src.in_name[j] for j in cols]
        self.out_name = [src.out_name[i] for i in rows]
        self.in_unit = [src.in_unit[j] for j in cols]
        self.out_unit = [src.out_unit[i] for i in rows]
        self.in_group, self.out_group = {}, {}
        self.name, self.notes, self.userdata, self.time_unit = src.name, src.notes, src.userdata, src.time_unit
        return self

    # -- operators ----------------------------------------------------------------------------
    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op == "+":
            return lti_add(left, right)
        if op == "-":
            return lti_add(left, right, -1.0)
        if op == "*":
            return lti_mul(left, right)
        if op == "/":
            return lti_mul(left, lti_inv(right))
        if op == "\\":
            return lti_mul(lti_inv(left), right)
        if op == "^":
            return lti_power(left, right)
        if op == ".*":
            return lti_times(left, right)
        if op == "./":
            return lti_times(left, right, divide=True)
        if op == ".^":
            return lti_power(left, right, elementwise=True)
        return NotImplemented

    def unary_op(self, interp, op: str) -> Any:
        if op == "-":
            return lti_neg(self)
        if op == "+":
            return self
        if op == "'":
            return lti_ctranspose(self)
        if op == ".'":
            return lti_transpose(self)
        return NotImplemented

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        items = [i for i in items if not is_empty_num(i)]
        if axis == 1:
            return lti_hcat(items)
        if axis == 0:
            return lti_vcat(items)
        raise MatlabError("Arrays of models (concatenation along a third dimension) are not supported.")

    # -- display ------------------------------------------------------------------------------
    def format_display(self, name: str) -> str:
        return f"{name} =\n \n" + "\n".join(self.display_lines()) + "\n\n"

    def summary_text(self, in_cell: bool) -> str:
        s = f"{self.ny}x{self.nu} {self.kind}"
        return s if in_cell else f"[{s}]"

    def time_label(self) -> str:
        if self.ts == 0:
            return "Continuous-time"
        return "Discrete-time"

    def ts_line(self) -> list[str]:
        if self.ts > 0:
            return [f"Sample time: {fmt_num(self.ts)} {self.time_unit}"]
        if self.ts < 0:
            return ["Sample time: unspecified"]
        return []


def ix_is_one(a: Any) -> bool:
    return isinstance(a, np.ndarray) and a.size == 1 and a.dtype != bool and float(np.real(a.flat[0])) == 1


def _grid(ny: int, nu: int, fn: Callable) -> list:
    return [[fn(i, j) for j in range(nu)] for i in range(ny)]


def _pad_delay(d: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n)
    out[:d.size] = d
    return out


def _assign_meta(target: LTI, val: LTI, rows, cols, ny: int, nu: int) -> None:
    """Grow and overwrite the delays and names of ``target`` for sys(rows, cols) = val."""
    in_d, out_d = _pad_delay(target.in_delay, nu), _pad_delay(target.out_delay, ny)
    io = np.zeros((ny, nu))
    io[:target.ny, :target.nu] = target.io_delay
    # delays of the new block go to the I/O delay matrix so that other channels stay as they were
    block = val.total_delay() - out_d[rows][:, None] - in_d[cols][None, :]
    if np.any(block < 0):
        # keep it simple: fold the input and output delays of the whole model into the I/O matrix
        io = io + out_d[:, None] + in_d[None, :]
        in_d, out_d = np.zeros(nu), np.zeros(ny)
        block = val.total_delay()
    io[np.ix_(rows, cols)] = block
    names_in = list(target.in_name) + [""] * (nu - target.nu)
    names_out = list(target.out_name) + [""] * (ny - target.ny)
    units_in = list(target.in_unit) + [""] * (nu - target.nu)
    units_out = list(target.out_unit) + [""] * (ny - target.ny)
    target.in_delay, target.out_delay, target.io_delay = in_d, out_d, io
    target.in_name, target.out_name, target.in_unit, target.out_unit = names_in, names_out, units_in, units_out


# -- transfer functions -----------------------------------------------------------------------

def _tf_elem_add(n1, d1, n2, d2, sign: float = 1.0):
    if cn.piszero(n2):
        return n1, d1
    if cn.piszero(n1):
        return (n2 if sign == 1 else -n2), d2
    if cn.peq(d1, d2):
        return cn.padd(n1, sign * n2), d1
    return cn.padd(cn.pmul(n1, d2), sign * cn.pmul(n2, d1)), cn.pmul(d1, d2)


def _tf_elem_mul(n1, d1, n2, d2):
    if cn.piszero(n1) or cn.piszero(n2):
        return np.zeros(1), np.ones(1)
    return cn.pmul(n1, n2), cn.pmul(d1, d2)


class TF(LTI):
    kind = "tf"
    rank = 2
    PROPS = ["Numerator", "Denominator", "Variable", "IODelay"]

    def __init__(self, num: list, den: list, ts: float = 0.0, var: str | None = None):
        ny = len(num)
        nu = len(num[0]) if ny else 0
        self.num = [[cn.ptrim(num[i][j]) for j in range(nu)] for i in range(ny)]
        self.den = [[cn.ptrim(den[i][j]) for j in range(nu)] for i in range(ny)]
        self._nu = nu
        for i in range(ny):
            for j in range(nu):
                if cn.piszero(self.den[i][j]):
                    raise MatlabError("The denominator of a transfer function cannot be zero.")
        self._init_common(ny, nu, ts)
        self.var = var or ("z" if ts != 0 else "s")

    @classmethod
    def siso(cls, num, den, ts: float = 0.0, var: str | None = None) -> "TF":
        return cls([[num]], [[den]], ts, var)

    @classmethod
    def gain(cls, k: np.ndarray, ts: float = 0.0) -> "TF":
        k = np.atleast_2d(k)
        return cls(_grid(k.shape[0], k.shape[1], lambda i, j: np.array([k[i, j]])),
                   _grid(k.shape[0], k.shape[1], lambda i, j: np.ones(1)), ts)

    @property
    def ny(self) -> int:
        return len(self.num)

    @property
    def nu(self) -> int:
        return self._nu

    def is_static(self) -> bool:
        return all(self.num[i][j].size == 1 and self.den[i][j].size == 1 or cn.piszero(self.num[i][j])
                   for i in range(self.ny) for j in range(self.nu)) and not self.has_delay()

    def padded(self, i: int, j: int):
        """Numerator and denominator padded with leading zeros to the same length."""
        n, d = self.num[i][j], self.den[i][j]
        m = max(n.size, d.size)
        dtype = np.result_type(n, d)
        pn, pd = np.zeros(m, dtype=dtype), np.zeros(m, dtype=dtype)
        pn[m - n.size:] = n
        pd[m - d.size:] = d
        return pn, pd

    def sub(self, rows, cols) -> "TF":
        out = TF([[self.num[i][j] for j in cols] for i in rows], [[self.den[i][j] for j in cols] for i in rows],
                 self.ts, self.var)
        if not len(rows):
            out._nu = len(cols)
        return out.sub_meta(self, rows, cols)

    def assign(self, rows, cols, val: "TF", ny: int, nu: int) -> "TF":
        num = _grid(ny, nu, lambda i, j: self.num[i][j] if i < self.ny and j < self.nu else np.zeros(1))
        den = _grid(ny, nu, lambda i, j: self.den[i][j] if i < self.ny and j < self.nu else np.ones(1))
        for a, i in enumerate(rows):
            for b, j in enumerate(cols):
                num[i][j], den[i][j] = val.num[a][b], val.den[a][b]
        out = TF(num, den, common_ts(self, val), self.var)
        out.copy_meta(self).copy_delays(self)
        out.name, out.notes, out.userdata = self.name, self.notes, self.userdata
        _assign_meta(out, val, rows, cols, ny, nu)
        return out

    def get_prop(self, prop: str) -> Any:
        if prop in ("Numerator", "Denominator"):
            data = np.empty((self.ny, self.nu), dtype=object)
            for i in range(self.ny):
                for j in range(self.nu):
                    data[i, j] = row(self.padded(i, j)[0 if prop == "Numerator" else 1])
            return Cell(data)
        if prop == "Variable":
            return char(self.var)
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop in ("Numerator", "Denominator"):
            polys = poly_grid(value, prop.lower())
            if len(polys) != self.ny or (self.ny and len(polys[0]) != self.nu):
                raise MatlabError(f"The {prop} must describe {self.ny} output(s) and {self.nu} input(s).")
            if prop == "Numerator":
                self.num = [[cn.ptrim(p) for p in r] for r in polys]
            else:
                den = [[cn.ptrim(p) for p in r] for r in polys]
                if any(cn.piszero(p) for r in den for p in r):
                    raise MatlabError("The denominator of a transfer function cannot be zero.")
                self.den = den
        elif prop == "Variable":
            v = text(value)
            allowed = ("z", "q", "z^-1", "q^-1") if self.is_dt else ("s", "p")
            if v not in allowed:
                raise MatlabError(f"Variable must be one of {', '.join(allowed)} for a "
                                  f"{'discrete' if self.is_dt else 'continuous'}-time model.")
            self.var = v
        else:
            super().set_prop(prop, value)

    def display_lines(self) -> list[str]:
        return _tf_display(self)


def poly_grid(value: Any, what: str) -> list:
    """Numerator/denominator data as a nested list [output][input] of coefficient vectors."""
    if isinstance(value, Cell):
        data = value.data
        if data.ndim != 2:
            raise MatlabError(f"The {what} cell array must be 2-D: one row per output, one column per input.")
        return [[_poly_vec(data[i, j], what) for j in range(data.shape[1])] for i in range(data.shape[0])]
    return [[_poly_vec(value, what)]]


def _poly_vec(v: Any, what: str) -> np.ndarray:
    a = as_matrix(v, what)
    if a.size == 0:
        raise MatlabError(f"The {what} cannot be empty.")
    if a.shape[0] != 1 and a.shape[1] != 1:
        raise MatlabError(f"The {what} of each channel must be a row vector of polynomial coefficients; use a "
                          "cell array {...} with one vector per output/input pair for MIMO models.")
    return a.ravel()


# -- zero-pole-gain models --------------------------------------------------------------------

class ZPK(LTI):
    kind = "zpk"
    rank = 3
    PROPS = ["Z", "P", "K", "DisplayFormat", "Variable", "IODelay"]

    def __init__(self, z: list, p: list, k, ts: float = 0.0, var: str | None = None):
        k = np.atleast_2d(np.asarray(k))
        if k.dtype.kind not in "fc":
            k = k.astype(float)
        ny, nu = k.shape
        self.z = [[np.asarray(z[i][j]).ravel() for j in range(nu)] for i in range(ny)]
        self.p = [[np.asarray(p[i][j]).ravel() for j in range(nu)] for i in range(ny)]
        self.k = k
        self._init_common(ny, nu, ts)
        self.var = var or ("z" if ts != 0 else "s")
        self.display_format = "roots"

    @classmethod
    def siso(cls, z, p, k, ts: float = 0.0, var: str | None = None) -> "ZPK":
        return cls([[z]], [[p]], [[k]], ts, var)

    @property
    def ny(self) -> int:
        return self.k.shape[0]

    @property
    def nu(self) -> int:
        return self.k.shape[1]

    def is_static(self) -> bool:
        return all((self.z[i][j].size == 0 and self.p[i][j].size == 0) or self.k[i, j] == 0
                   for i in range(self.ny) for j in range(self.nu)) and not self.has_delay()

    def sub(self, rows, cols) -> "ZPK":
        out = ZPK([[self.z[i][j] for j in cols] for i in rows], [[self.p[i][j] for j in cols] for i in rows],
                  self.k[np.ix_(rows, cols)], self.ts, self.var)
        out.display_format = self.display_format
        return out.sub_meta(self, rows, cols)

    def assign(self, rows, cols, val: "ZPK", ny: int, nu: int) -> "ZPK":
        z = _grid(ny, nu, lambda i, j: self.z[i][j] if i < self.ny and j < self.nu else np.zeros(0))
        p = _grid(ny, nu, lambda i, j: self.p[i][j] if i < self.ny and j < self.nu else np.zeros(0))
        k = np.zeros((ny, nu), dtype=np.result_type(self.k, val.k))
        k[:self.ny, :self.nu] = self.k
        for a, i in enumerate(rows):
            for b, j in enumerate(cols):
                z[i][j], p[i][j], k[i, j] = val.z[a][b], val.p[a][b], val.k[a, b]
        out = ZPK(z, p, k, common_ts(self, val), self.var)
        out.copy_meta(self).copy_delays(self)
        out.name, out.notes, out.userdata, out.display_format = self.name, self.notes, self.userdata, \
            self.display_format
        _assign_meta(out, val, rows, cols, ny, nu)
        return out

    def get_prop(self, prop: str) -> Any:
        if prop in ("Z", "P"):
            src = self.z if prop == "Z" else self.p
            data = np.empty((self.ny, self.nu), dtype=object)
            for i in range(self.ny):
                for j in range(self.nu):
                    v = np.asarray(src[i][j])
                    data[i, j] = col(cn.real_if_close(v)) if v.size else np.zeros((0, 1))
            return Cell(data)
        if prop == "K":
            return self.k
        if prop == "DisplayFormat":
            return char(self.display_format)
        if prop == "Variable":
            return char(self.var)
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop in ("Z", "P"):
            roots = root_grid(value, self.ny, self.nu, "zeros" if prop == "Z" else "poles")
            if prop == "Z":
                self.z = roots
            else:
                self.p = roots
        elif prop == "K":
            k = as_matrix(value, "gain K")
            if k.shape != (self.ny, self.nu):
                raise MatlabError(f"K must be a {self.ny}-by-{self.nu} matrix.")
            self.k = k
        elif prop == "DisplayFormat":
            v = text(value).lower()
            if v not in ("roots", "frequency", "time constant"):
                raise MatlabError("DisplayFormat must be 'roots', 'frequency' or 'time constant'.")
            if v != "roots" and self.is_dt:
                raise MatlabError("Only the 'roots' display format is available for discrete-time models.")
            self.display_format = v
        elif prop == "Variable":
            TF.set_prop(self, prop, value)  # same rule as for transfer functions
        else:
            super().set_prop(prop, value)

    def display_lines(self) -> list[str]:
        return _zpk_display(self)


def root_grid(value: Any, ny: int, nu: int, what: str) -> list:
    if isinstance(value, Cell):
        data = value.data
        if data.shape != (ny, nu):
            raise MatlabError(f"The {what} cell array must be {ny}-by-{nu}, like the gain matrix.")
        return [[_root_vec(data[i, j], what) for j in range(nu)] for i in range(ny)]
    if (ny, nu) != (1, 1):
        raise MatlabError(f"For a MIMO model give the {what} as a {ny}-by-{nu} cell array of vectors.")
    return [[_root_vec(value, what)]]


def _root_vec(v: Any, what: str) -> np.ndarray:
    if is_empty_num(v):
        return np.zeros(0)
    a = as_matrix(v, what)
    if a.shape[0] != 1 and a.shape[1] != 1:
        raise MatlabError(f"The {what} of each channel must be a vector.")
    return a.ravel()


# -- state-space models -----------------------------------------------------------------------

class SS(LTI):
    kind = "ss"
    rank = 4
    PROPS = ["A", "B", "C", "D", "E", "Scaled", "StateName", "StatePath", "StateUnit", "InternalDelay"]

    def __init__(self, a, b, c, d, ts: float = 0.0):
        a, b, c, d = (cn.mat(m) for m in (a, b, c, d))
        n = a.shape[0]
        if a.shape != (n, n):
            raise MatlabError("The A matrix must be square.")
        if a.size == 0:
            a = np.zeros((0, 0))
            n = 0
        ny, nu = d.shape
        if b.size == 0 and b.shape != (n, nu):
            b = np.zeros((n, nu))
        if c.size == 0 and c.shape != (ny, n):
            c = np.zeros((ny, n))
        if b.shape[0] != n:
            raise MatlabError(f"The A and B matrices must have the same number of rows (A is "
                              f"{a.shape[0]}-by-{a.shape[1]}, B is {b.shape[0]}-by-{b.shape[1]}).")
        if c.shape[1] != n:
            raise MatlabError(f"The A and C matrices must have the same number of columns (A is "
                              f"{a.shape[0]}-by-{a.shape[1]}, C is {c.shape[0]}-by-{c.shape[1]}).")
        if d.shape != (c.shape[0], b.shape[1]):
            raise MatlabError(f"The D matrix must be {c.shape[0]}-by-{b.shape[1]} (rows of C by columns of B); "
                              f"it is {d.shape[0]}-by-{d.shape[1]}.")
        self.a, self.b, self.c, self.d = a, b, c, d
        self._init_common(c.shape[0], b.shape[1], ts)
        self.state_name = [""] * n
        self.state_unit = [""] * n

    @property
    def ny(self) -> int:
        return self.d.shape[0]

    @property
    def nu(self) -> int:
        return self.d.shape[1]

    @property
    def nx(self) -> int:
        return self.a.shape[0]

    @property
    def abcd(self):
        return self.a, self.b, self.c, self.d

    def is_static(self) -> bool:
        return self.nx == 0 and not self.has_delay()

    def sub(self, rows, cols) -> "SS":
        out = SS(self.a, self.b[:, cols], self.c[rows, :], self.d[np.ix_(rows, cols)], self.ts)
        out.state_name, out.state_unit = list(self.state_name), list(self.state_unit)
        return out.sub_meta(self, rows, cols)

    def assign(self, rows, cols, val: "SS", ny: int, nu: int) -> "SS":
        # sys(rows, cols) = val: remove the old block, then add the new one in parallel
        b = np.zeros((self.nx, nu), dtype=self.b.dtype)
        b[:, :self.nu] = self.b
        c = np.zeros((ny, self.nx), dtype=self.c.dtype)
        c[:self.ny, :] = self.c
        d = np.zeros((ny, nu), dtype=np.result_type(self.d, val.d))
        d[:self.ny, :self.nu] = self.d
        keep_rows = np.array([i for i in range(ny) if i not in set(rows.tolist())], dtype=int)
        keep_cols = np.array([j for j in range(nu) if j not in set(cols.tolist())], dtype=int)
        d[np.ix_(rows, cols)] = 0
        # split the old dynamics into the part seen outside the block ...
        pieces = []
        if self.nx:
            c_in = c.copy()
            c_in[keep_rows, :] = 0  # rows of the block, towards columns outside the block
            b_out = b.copy()
            b_out[:, cols] = 0
            c_out = c.copy()
            c_out[rows, :] = 0
            pieces.append((self.a, b, c_out, np.zeros((ny, nu))))        # other rows, all columns
            pieces.append((self.a, b_out, c_in, np.zeros((ny, nu))))     # block rows, other columns
        bv = np.zeros((val.nx, nu), dtype=val.b.dtype)
        bv[:, cols] = val.b
        cv = np.zeros((ny, val.nx), dtype=val.c.dtype)
        cv[rows, :] = val.c
        d[np.ix_(rows, cols)] = val.d
        pieces.append((val.a, bv, cv, np.zeros((ny, nu))))
        a_new = cn.blkdiag(*[p[0] for p in pieces])
        b_new = np.vstack([p[1] for p in pieces])
        c_new = np.hstack([p[2] for p in pieces])
        del keep_cols
        out = SS(a_new, b_new, c_new, d, common_ts(self, val))
        idx = cn.sminreal_states(out.a, out.b, out.c)
        out = SS(out.a[np.ix_(idx, idx)], out.b[idx, :], out.c[:, idx], out.d, out.ts)
        out.copy_meta(self).copy_delays(self)
        out.name, out.notes, out.userdata = self.name, self.notes, self.userdata
        _assign_meta(out, val, rows, cols, ny, nu)
        if np.any(out.io_delay):
            raise MatlabError("This assignment would give a state-space model different delays for each "
                              "input/output pair, which is not supported. " + PADE_HINT)
        return out

    def get_prop(self, prop: str) -> Any:
        if prop in ("A", "B", "C", "D"):
            return getattr(self, prop.lower())
        if prop == "E":
            return np.zeros((0, 0))
        if prop == "Scaled":
            return np.array([[False]])
        if prop == "StateName":
            return names_cell(self.state_name)
        if prop == "StateUnit":
            return names_cell(self.state_unit)
        if prop == "StatePath":
            return names_cell([""] * self.nx)
        if prop == "InternalDelay":
            return np.zeros((0, 1))
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop in ("A", "B", "C", "D"):
            m = as_matrix(value, f"{prop} matrix")
            parts = {"A": self.a, "B": self.b, "C": self.c, "D": self.d}
            parts[prop] = m
            new = SS(parts["A"], parts["B"], parts["C"], parts["D"], self.ts)
            if (new.ny, new.nu) != (self.ny, self.nu):
                raise MatlabError("Assigning this matrix would change the number of inputs or outputs.")
            self.a, self.b, self.c, self.d = new.abcd
            if len(self.state_name) != new.nx:
                self.state_name, self.state_unit = [""] * new.nx, [""] * new.nx
        elif prop == "E":
            if not is_empty_num(value):
                e = as_matrix(value, "E matrix")
                if e.shape != self.a.shape or np.linalg.matrix_rank(e) < e.shape[0]:
                    raise MatlabError("Only nonsingular E matrices of the same size as A are supported.")
                self.a, self.b = np.linalg.solve(e, self.a), np.linalg.solve(e, self.b)
        elif prop in ("StateName", "StateUnit"):
            names = names_from(value)
            if not names:
                names = [""] * self.nx
            if len(names) != self.nx:
                raise MatlabError(f"{prop} needs {self.nx} entries, one per state.")
            if prop == "StateName":
                self.state_name = names
            else:
                self.state_unit = names
        elif prop == "IODelay":
            raise MatlabError("State-space models have InputDelay and OutputDelay but no IODelay property.")
        else:
            super().set_prop(prop, value)

    def prop_names(self) -> list[str]:
        return self.PROPS + _COMMON_PROPS

    def display_lines(self) -> list[str]:
        return _ss_display(self)


# -- frequency-response data ------------------------------------------------------------------

class FRD(LTI):
    kind = "frd"
    rank = 5
    PROPS = ["Frequency", "FrequencyUnit", "ResponseData", "IODelay"]

    def __init__(self, resp: np.ndarray, freq: np.ndarray, ts: float = 0.0, unit: str = "rad/TimeUnit"):
        resp = np.asarray(resp, dtype=complex)
        freq = np.asarray(freq, dtype=float).ravel()
        if resp.ndim != 3 or resp.shape[2] != freq.size:
            raise MatlabError("The response data must have one value (or one ny-by-nu matrix) per frequency.")
        if np.any(freq < 0) or np.any(~np.isfinite(freq)):
            raise MatlabError("Frequencies must be real, finite and non-negative.")
        self.resp, self.freq = resp, freq
        self._init_common(resp.shape[0], resp.shape[1], ts)
        self.freq_unit = unit

    @property
    def ny(self) -> int:
        return self.resp.shape[0]

    @property
    def nu(self) -> int:
        return self.resp.shape[1]

    def rad_freq(self) -> np.ndarray:
        """Frequencies in rad per time unit."""
        return self.freq * freq_unit_factor(self.freq_unit)

    def sub(self, rows, cols) -> "FRD":
        out = FRD(self.resp[np.ix_(rows, cols)], self.freq, self.ts, self.freq_unit)
        return out.sub_meta(self, rows, cols)

    def assign(self, rows, cols, val: "FRD", ny: int, nu: int) -> "FRD":
        resp = np.zeros((ny, nu, self.freq.size), dtype=complex)
        resp[:self.ny, :self.nu] = self.resp
        resp[np.ix_(rows, cols)] = val.resp
        out = FRD(resp, self.freq, common_ts(self, val), self.freq_unit)
        out.copy_meta(self)
        return out

    def with_resp(self, resp: np.ndarray) -> "FRD":
        return FRD(resp, self.freq, self.ts, self.freq_unit)

    def get_prop(self, prop: str) -> Any:
        if prop == "Frequency":
            return col(self.freq)
        if prop == "FrequencyUnit":
            return char(self.freq_unit)
        if prop == "ResponseData":
            return cn.real_if_close(self.resp, 0) if not np.any(self.resp.imag) else self.resp
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop == "Frequency":
            f = as_vector(value, "Frequency").real
            if f.size != self.freq.size:
                raise MatlabError("The number of frequencies must match the response data.")
            self.freq = f.astype(float)
        elif prop == "FrequencyUnit":
            u = text(value)
            freq_unit_factor(u)
            self.freq_unit = u
        elif prop == "ResponseData":
            r = np.asarray(value)
            if r.ndim == 2 and self.is_siso and r.size == self.freq.size:
                r = r.reshape(1, 1, -1)
            if r.shape != self.resp.shape:
                raise MatlabError("ResponseData must keep its size (outputs-by-inputs-by-frequencies).")
            self.resp = r.astype(complex)
        elif prop in ("InputDelay", "OutputDelay", "IODelay"):
            super().set_prop(prop, value)
        else:
            super().set_prop(prop, value)

    def display_lines(self) -> list[str]:
        return _frd_display(self)


def freq_unit_factor(unit: str) -> float:
    table = {"rad/timeunit": 1.0, "rad/s": 1.0, "rad/sec": 1.0, "rad/second": 1.0, "hz": 2 * np.pi,
             "cycles/timeunit": 2 * np.pi, "khz": 2e3 * np.pi, "mhz": 2e6 * np.pi, "ghz": 2e9 * np.pi,
             "rpm": 2 * np.pi / 60, "rad/minute": 1 / 60.0, "rad/hour": 1 / 3600.0}
    key = unit.lower()
    if key not in table:
        raise MatlabError(f"Unknown frequency unit '{unit}'. Use 'rad/TimeUnit', 'rad/s', 'Hz', 'kHz', 'MHz', "
                          "'GHz', 'rpm' or 'cycles/TimeUnit'.")
    return table[key]


# -- PID controllers --------------------------------------------------------------------------

_FORMULAS = {"forwardeuler": "ForwardEuler", "backwardeuler": "BackwardEuler", "trapezoidal": "Trapezoidal"}


class PID(LTI):
    """PID controller in parallel form: Kp + Ki/s + Kd*s/(Tf*s + 1)."""

    kind = "pid"
    rank = 1
    PROPS = ["Kp", "Ki", "Kd", "Tf", "IFormula", "DFormula"]

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, tf: float = 0.0, ts: float = 0.0):
        self.kp, self.ki, self.kd, self.tf = float(kp), float(ki), float(kd), float(tf)
        self._init_common(1, 1, ts)
        self.iformula = "ForwardEuler"
        self.dformula = "ForwardEuler"
        self.check()

    def check(self) -> None:
        for v, nm in ((self.kp, "Kp"), (self.ki, "Ki"), (self.kd, "Kd"), (self.tf, "Tf")):
            if not np.isfinite(v):
                raise MatlabError(f"{nm} must be a finite real number.")
        if self.tf < 0:
            raise MatlabError("The derivative filter time constant Tf must not be negative.")

    @property
    def ny(self) -> int:
        return 1

    @property
    def nu(self) -> int:
        return 1

    def is_static(self) -> bool:
        return self.ki == 0 and self.kd == 0 and not self.has_delay()

    def sub(self, rows, cols) -> "LTI":
        if len(rows) == 1 and len(cols) == 1:
            return self
        return to_tf(self).sub(rows, cols)

    def gains(self):
        """Kp, Ki, Kd, Tf of the parallel form."""
        return self.kp, self.ki, self.kd, self.tf

    def type_name(self) -> str:
        kp, ki, kd, tf = self.gains()
        s = ("P" if kp != 0 else "") + ("I" if ki != 0 else "") + ("D" if kd != 0 else "")
        if kd != 0 and tf != 0:
            s += "F"
        return s or "P"

    def get_prop(self, prop: str) -> Any:
        if prop in ("Kp", "Ki", "Kd", "Tf"):
            return float(getattr(self, prop.lower()))
        if prop == "IFormula":
            return char(self.iformula)
        if prop == "DFormula":
            return char(self.dformula)
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop in ("Kp", "Ki", "Kd", "Tf"):
            setattr(self, prop.lower(), as_scalar(value, prop))
            self.check()
        elif prop in ("IFormula", "DFormula"):
            v = _FORMULAS.get(text(value).lower())
            if v is None:
                raise MatlabError(f"{prop} must be 'ForwardEuler', 'BackwardEuler' or 'Trapezoidal'.")
            setattr(self, prop.lower(), v)
        else:
            super().set_prop(prop, value)

    def prop_names(self) -> list[str]:
        return self.PROPS + _COMMON_PROPS

    def display_lines(self) -> list[str]:
        return _pid_display(self)


class PIDSTD(PID):
    """PID controller in standard form: Kp * (1 + 1/(Ti*s) + Td*s/(Td/N*s + 1))."""

    kind = "pidstd"
    PROPS = ["Kp", "Ti", "Td", "N", "IFormula", "DFormula"]

    def __init__(self, kp: float = 1.0, ti: float = np.inf, td: float = 0.0, n: float = np.inf, ts: float = 0.0):
        self.kp, self.ti, self.td, self.n = float(kp), float(ti), float(td), float(n)
        self._init_common(1, 1, ts)
        self.iformula = "ForwardEuler"
        self.dformula = "ForwardEuler"
        self.check()

    def check(self) -> None:
        if not np.isfinite(self.kp):
            raise MatlabError("Kp must be a finite real number.")
        if not (self.ti > 0):
            raise MatlabError("The integral time Ti must be positive (Inf turns integral action off).")
        if not (self.td >= 0 and np.isfinite(self.td)):
            raise MatlabError("The derivative time Td must be finite and not negative.")
        if not (self.n > 0):
            raise MatlabError("The filter divisor N must be positive (Inf means no derivative filter).")

    def gains(self):
        ki = 0.0 if np.isinf(self.ti) else self.kp / self.ti
        kd = self.kp * self.td
        tf = 0.0 if np.isinf(self.n) else self.td / self.n
        return self.kp, ki, kd, tf

    def is_static(self) -> bool:
        return np.isinf(self.ti) and self.td == 0 and not self.has_delay()

    def get_prop(self, prop: str) -> Any:
        if prop in ("Kp", "Ti", "Td", "N"):
            return float(getattr(self, prop.lower()))
        return super().get_prop(prop)

    def set_prop(self, prop: str, value: Any) -> None:
        if prop in ("Kp", "Ti", "Td", "N"):
            setattr(self, prop.lower(), as_scalar(value, prop))
            self.check()
        else:
            super().set_prop(prop, value)


# -- conversions between model types ----------------------------------------------------------

def pid_to_tf(c: PID) -> TF:
    kp, ki, kd, tf = c.gains()
    ts = c.ts
    if ts == 0:
        num, den = np.array([kp]), np.ones(1)
        if ki != 0:
            num, den = _tf_elem_add(num, den, np.array([ki]), np.array([1.0, 0.0]))
        if kd != 0:
            dden = np.array([tf, 1.0]) if tf != 0 else np.ones(1)
            num, den = _tf_elem_add(num, den, np.array([kd, 0.0]), dden)
    else:
        h = c.tsv()
        num, den = np.array([kp]), np.ones(1)
        if ki != 0:
            inum = {"ForwardEuler": np.array([1.0]), "BackwardEuler": np.array([1.0, 0.0]),
                    "Trapezoidal": np.array([0.5, 0.5])}[c.iformula] * ki * h
            num, den = _tf_elem_add(num, den, inum, np.array([1.0, -1.0]))
        if kd != 0:
            dden = {"ForwardEuler": np.array([tf, h - tf]), "BackwardEuler": np.array([tf + h, -tf]),
                    "Trapezoidal": np.array([tf + h / 2, h / 2 - tf])}[c.dformula]
            num, den = _tf_elem_add(num, den, kd * np.array([1.0, -1.0]), dden)
    out = TF.siso(num, den, ts)
    out.copy_meta(c)
    out.in_delay, out.out_delay = c.in_delay, c.out_delay
    out.name, out.notes, out.userdata = c.name, c.notes, c.userdata
    return out


def tf_to_pid(g: LTI, like: PID | None = None, standard: bool = False) -> PID | None:
    """The PID controller equal to ``g``, or None when ``g`` does not have PID structure."""
    g = to_tf(g)
    if not g.is_siso or np.any(g.io_delay):
        return None
    num, den = g.num[0][0], g.den[0][0]
    if np.iscomplexobj(num) or np.iscomplexobj(den):
        return None
    ts = g.ts
    iform = like.iformula if like is not None else "ForwardEuler"
    dform = like.dformula if like is not None else "ForwardEuler"
    h = g.tsv()
    poles = cn.proots(den)
    int_pole = 1.0 if ts != 0 else 0.0
    scale = max(1.0, float(np.max(np.abs(poles)))) if poles.size else 1.0
    is_int = np.abs(poles - int_pole) <= 1e-8 * scale
    if np.sum(is_int) > 1 or poles.size - np.sum(is_int) > 1:
        return None
    rest = poles[~is_int]
    has_i = bool(np.any(is_int))
    tf_ = 0.0
    if rest.size:
        p0 = rest[0]
        if abs(np.imag(p0)) > 1e-10 * scale:
            return None
        p0 = float(np.real(p0))
        if ts == 0:
            if p0 >= 0:
                return None
            tf_ = -1.0 / p0
        else:
            if dform == "ForwardEuler":
                tf_ = h / (1 - p0)
            elif dform == "BackwardEuler":
                tf_ = h * p0 / (1 - p0)
            else:
                tf_ = (h / 2) * (1 + p0) / (1 - p0)
            if abs(tf_) < 1e-12 * h:
                tf_ = 0.0
            if tf_ < 0 or not np.isfinite(tf_):
                return None
    # with the pole structure fixed the controller is linear in Kp, Ki and Kd
    basis = [PID(1, 0, 0, 0, ts), PID(0, 1, 0, 0, ts), PID(0, 0, 1, tf_, ts)]
    for bsys in basis:
        bsys.iformula, bsys.dformula = iform, dform
    roots_all = np.concatenate([poles, cn.proots(num)])
    base = np.array([0.37 + 0.61j, 1.3 - 0.4j, -0.9 + 1.1j, 2.2 + 0.3j, 0.11 - 1.7j, -1.6 - 0.5j, 0.7 + 2.3j])
    if ts != 0:
        pts = 1.0 + base
    else:
        pts = base * (max(1.0, float(np.max(np.abs(roots_all)))) if roots_all.size else 1.0)
    target = cn.poly_freqresp(num, den, pts)
    cols_resp = []
    for bsys in basis:
        t = pid_to_tf(bsys)
        cols_resp.append(cn.poly_freqresp(t.num[0][0], t.den[0][0], pts))
    m = np.array(cols_resp).T
    mm = np.vstack([m.real, m.imag])
    tt = np.concatenate([target.real, target.imag])
    sol, *_ = np.linalg.lstsq(mm, tt, rcond=None)
    if np.linalg.norm(mm @ sol - tt) > 1e-8 * max(float(np.linalg.norm(tt)), 1e-300):
        return None
    kp, ki, kd = (float(v) for v in sol)
    big = max(abs(kp), abs(ki), abs(kd), 1e-300)
    kp, ki, kd = (0.0 if abs(v) < 1e-12 * big else v for v in (kp, ki, kd))
    if ki != 0 and not has_i:
        return None
    if kd == 0:
        tf_ = 0.0
    if standard:
        out: PID = pid_to_std(kp, ki, kd, tf_, ts)
    else:
        out = PID(kp, ki, kd, tf_, ts)
    out.iformula, out.dformula = iform, dform
    out.copy_meta(g)
    out.in_delay, out.out_delay = g.in_delay, g.out_delay
    out.name, out.notes, out.userdata = g.name, g.notes, g.userdata
    return out


def pid_to_std(kp: float, ki: float, kd: float, tf: float, ts: float) -> "PIDSTD":
    if kp == 0:
        raise MatlabError("A controller without proportional gain (Kp = 0) cannot be written in standard "
                          "form; use the parallel form pid(...).")
    ti = np.inf if ki == 0 else kp / ki
    td = kd / kp
    if ti <= 0 or td < 0:
        raise MatlabError("This controller cannot be written in standard form: Ti and Td would be negative.")
    n = np.inf if tf == 0 else td / tf
    return PIDSTD(kp, ti, td, n, ts)


def to_tf(sys: Any) -> TF:
    if isinstance(sys, TF):
        return sys
    if isinstance(sys, PID):
        return pid_to_tf(sys)
    if isinstance(sys, ZPK):
        num = _grid(sys.ny, sys.nu, lambda i, j: sys.k[i, j] * cn.rpoly(sys.z[i][j]))
        den = _grid(sys.ny, sys.nu, lambda i, j: cn.rpoly(sys.p[i][j]))
        out = TF(num, den, sys.ts, sys.var)
        _carry(out, sys)
        return out
    if isinstance(sys, SS):
        num = _grid(sys.ny, sys.nu, lambda i, j: None)
        den = _grid(sys.ny, sys.nu, lambda i, j: None)
        real = not any(np.iscomplexobj(m) for m in sys.abcd)
        for i in range(sys.ny):
            for j in range(sys.nu):
                idx = cn.sminreal_states(sys.a, sys.b[:, [j]], sys.c[[i], :])
                nmr, dn = cn.ss_numden(sys.a[np.ix_(idx, idx)], sys.b[idx][:, [j]], sys.c[[i]][:, idx],
                                       sys.d[[i]][:, [j]])
                n_ij = _clean_poly(nmr[0, 0])
                if cn.piszero(n_ij):
                    num[i][j], den[i][j] = np.zeros(1), np.ones(1)
                else:
                    num[i][j], den[i][j] = (n_ij.real if real and np.iscomplexobj(n_ij) else n_ij), dn
        out = TF(num, den, sys.ts)
        if sys.ny == 0:
            out._nu = sys.nu
        _carry(out, sys)
        return out
    if isinstance(sys, FRD):
        raise MatlabError("A frequency-response (frd) model holds samples only and cannot be converted to a "
                          "transfer function.")
    if is_num(sys):
        return TF.gain(as_matrix(sys, "gain"))
    raise MatlabError("A control-system model (tf, zpk, ss or pid) or a numeric gain is expected.")


def _clean_poly(num: np.ndarray) -> np.ndarray:
    """Zero the coefficients that are rounding noise relative to the largest one."""
    scale = max(float(np.max(np.abs(num))) if num.size else 0.0, 1e-300)
    out = num.copy()
    out[np.abs(out) < 1e-13 * scale] = 0
    return out


def _carry(out: LTI, src: LTI) -> LTI:
    """Copy everything except the model data: names, delays, notes."""
    out.copy_meta(src)
    out.in_delay, out.out_delay = src.in_delay, src.out_delay
    if not isinstance(out, SS):
        out.io_delay = src.io_delay
    out.name, out.notes, out.userdata = src.name, src.notes, src.userdata
    return out


def to_zpk(sys: Any) -> ZPK:
    if isinstance(sys, ZPK):
        return sys
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    if isinstance(sys, TF):
        z = _grid(sys.ny, sys.nu, lambda i, j: cn.proots(sys.num[i][j]))
        p = _grid(sys.ny, sys.nu, lambda i, j: cn.proots(sys.den[i][j]))
        k = np.zeros((sys.ny, sys.nu), dtype=complex)
        for i in range(sys.ny):
            for j in range(sys.nu):
                k[i, j] = 0 if cn.piszero(sys.num[i][j]) else sys.num[i][j][0] / sys.den[i][j][0]
                if k[i, j] == 0:
                    z[i][j], p[i][j] = np.zeros(0), np.zeros(0)
        out = ZPK(z, p, k.real.copy() if not np.any(k.imag) else k, sys.ts, sys.var)
        _carry(out, sys)
        return out
    if isinstance(sys, SS):
        z = _grid(sys.ny, sys.nu, lambda i, j: np.zeros(0))
        p = _grid(sys.ny, sys.nu, lambda i, j: np.zeros(0))
        cplx = any(np.iscomplexobj(m) for m in sys.abcd)
        k = np.zeros((sys.ny, sys.nu), dtype=complex if cplx else float)
        for i in range(sys.ny):
            for j in range(sys.nu):
                idx = cn.sminreal_states(sys.a, sys.b[:, [j]], sys.c[[i], :])
                a = sys.a[np.ix_(idx, idx)]
                zz, kk = cn.siso_zeros_gain(a, sys.b[idx][:, [j]], sys.c[[i]][:, idx], sys.d[i, j])
                k[i, j] = kk
                if kk != 0:
                    z[i][j] = zz
                    p[i][j] = cn.eigvals(a) if cplx else cn.pair_conjugates(cn.eigvals(a))
        out = ZPK(z, p, k, sys.ts)
        _carry(out, sys)
        return out
    if isinstance(sys, FRD):
        raise MatlabError("A frequency-response (frd) model holds samples only and cannot be converted to a "
                          "zero-pole-gain model.")
    if is_num(sys):
        return to_zpk(TF.gain(as_matrix(sys, "gain")))
    raise MatlabError("A control-system model (tf, zpk, ss or pid) or a numeric gain is expected.")


def _poly_lcm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least common multiple of two polynomials (monic), matching roots with a tolerance."""
    a, b = a / a[0], b / b[0]
    if cn.peq(a, b) or b.size == 1:
        return a
    if a.size == 1:
        return b
    ra, rb = list(cn.proots(a)), cn.proots(b)
    extra = []
    for r in rb:
        if ra:
            dist = [abs(r - x) for x in ra]
            k = int(np.argmin(dist))
            if dist[k] <= 1e-7 * max(1.0, abs(r)):
                ra.pop(k)
                continue
        extra.append(r)
    if len(extra) == rb.size:
        return cn.pmul(a, b)
    if not extra:
        return a
    return cn.pmul(a, cn.rpoly(cn.pair_conjugates(np.array(extra))))


def _column_realization(nums: list, dens: list):
    """One realization (controller form) of a column of transfer functions sharing an input."""
    lcd = np.ones(1)
    for n, d in zip(nums, dens):
        if not cn.piszero(n):
            lcd = _poly_lcm(lcd, d)
    n_states = lcd.size - 1
    dtype = np.result_type(lcd, *nums, float)
    a = np.zeros((n_states, n_states), dtype=dtype)
    b = np.zeros((n_states, 1), dtype=dtype)
    c = np.zeros((len(nums), n_states), dtype=dtype)
    dd = np.zeros((len(nums), 1), dtype=dtype)
    if n_states:
        a[0, :] = -lcd[1:]
        if n_states > 1:
            a[1:, :-1] = np.eye(n_states - 1)
        b[0, 0] = 1.0
    for i, (n, d) in enumerate(zip(nums, dens)):
        if cn.piszero(n):
            continue
        if cn.peq(d / d[0], lcd):
            full = n / d[0]
        else:
            q, _ = np.polydiv(lcd, d / d[0])
            full = cn.pmul(n / d[0], q)
        if full.size - 1 > n_states:
            raise MatlabError("An improper transfer function (more zeros than poles) has no state-space "
                              "realization here; multiply it with other blocks first or add a fast pole.")
        padded = np.zeros(n_states + 1, dtype=dtype)
        padded[n_states + 1 - full.size:] = full
        dd[i, 0] = padded[0]
        if n_states:
            c[i, :] = padded[1:] - padded[0] * lcd[1:]
    return a, b, c, dd


def _zpk_sections(z: np.ndarray, p: np.ndarray, k):
    """Split a SISO zero-pole-gain model into real first- and second-order sections."""
    def factors(r):
        r = cn.pair_conjugates(r)
        out, used = [], np.zeros(len(r), dtype=bool)
        for i, v in enumerate(r):
            if used[i]:
                continue
            used[i] = True
            if np.imag(v) == 0:
                out.append(np.array([1.0, -float(np.real(v))]))
                continue
            partner = [j for j in range(len(r)) if not used[j] and r[j] == np.conj(v)]
            if not partner:
                return None
            used[partner[0]] = True
            out.append(np.array([1.0, -2 * float(np.real(v)), float(abs(v) ** 2)]))
        return out

    pf, zf = factors(p), factors(z)
    if pf is None or zf is None:
        return None
    # second-order zero factors need second-order pole sections: merge first-order pole factors
    need2 = sum(1 for f in zf if f.size == 3)
    ones = [f for f in pf if f.size == 2]
    twos = [f for f in pf if f.size == 3]
    while need2 > len(twos) and len(ones) >= 2:
        twos.append(cn.pmul(ones.pop(), ones.pop()))
    if need2 > len(twos):
        return None
    sections = [[np.ones(1), f] for f in twos + ones]
    for f in sorted(zf, key=lambda f: -f.size):
        for sec in sections:
            if sec[0].size - 1 + f.size - 1 <= sec[1].size - 1 and (f.size == 2 or sec[1].size == 3):
                sec[0] = cn.pmul(sec[0], f)
                break
        else:
            return None
    if sections:
        sections[0][0] = sections[0][0] * k
    return sections


def _siso_zpk_ss(z, p, k):
    if len(z) > len(p):
        raise MatlabError("An improper model (more zeros than poles) has no state-space realization here; "
                          "multiply it with other blocks first or add a fast pole.")
    if not (np.iscomplexobj(k) and np.imag(k) != 0):
        secs = _zpk_sections(np.asarray(z), np.asarray(p), float(np.real(k)))
        if secs is not None:
            if not secs:
                return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[float(np.real(k))]])
            real = cn.tf2ss_siso(*secs[0])
            for sec in secs[1:]:
                real = cn.ss_mul(real, cn.tf2ss_siso(*sec))
            return real
    return cn.tf2ss_siso(k * cn.rpoly(z), cn.rpoly(p))


def to_ss(sys: Any) -> SS:
    if isinstance(sys, SS):
        return sys
    if isinstance(sys, FRD):
        raise MatlabError("A frequency-response (frd) model holds samples only and cannot be converted to "
                          "state space.")
    if is_num(sys):
        d = as_matrix(sys, "gain")
        return SS(np.zeros((0, 0)), np.zeros((0, d.shape[1])), np.zeros((d.shape[0], 0)), d)
    if not isinstance(sys, LTI):
        raise MatlabError("A control-system model (tf, zpk, ss or pid) or a numeric gain is expected.")
    src = sys
    if isinstance(sys, ZPK) and sys.is_siso:
        out = SS(*_siso_zpk_ss(sys.z[0][0], sys.p[0][0], sys.k[0, 0]), sys.ts)
    else:
        g = to_tf(sys)
        if g.is_siso:
            out = SS(*cn.tf2ss_siso(g.num[0][0], g.den[0][0]), g.ts)
        elif g.ny == 0 or g.nu == 0:
            out = SS(np.zeros((0, 0)), np.zeros((0, g.nu)), np.zeros((g.ny, 0)), np.zeros((g.ny, g.nu)), g.ts)
        else:
            cols_ = [_column_realization([g.num[i][j] for i in range(g.ny)], [g.den[i][j] for i in range(g.ny)])
                     for j in range(g.nu)]
            rows_ = [_column_realization([g.num[i][j] for j in range(g.nu)], [g.den[i][j] for j in range(g.nu)])
                     for i in range(g.ny)]
            ncol = sum(r[0].shape[0] for r in cols_)
            nrow = sum(r[0].shape[0] for r in rows_)
            if ncol <= nrow:
                a = cn.blkdiag(*[r[0] for r in cols_])
                b = cn.blkdiag(*[r[1] for r in cols_])
                c = np.hstack([r[2] for r in cols_])
                d = np.hstack([r[3] for r in cols_])
            else:
                a = cn.blkdiag(*[r[0].T for r in rows_])
                c = cn.blkdiag(*[r[1].T for r in rows_])
                b = np.vstack([r[2].T for r in rows_])
                d = np.vstack([r[3].T for r in rows_])
            out = SS(a, b, c, d, g.ts)
    out.copy_meta(src)
    out.name, out.notes, out.userdata = src.name, src.notes, src.userdata
    out.in_delay, out.out_delay = src.in_delay, src.out_delay
    if np.any(src.io_delay):
        ind, outd, io = split_delays(src.total_delay(), nonzero_mask(src))
        if np.any(io):
            if src.is_dt:
                return to_ss(absorb_delay(src))
            raise MatlabError("This model has a different delay for each input/output pair, which a "
                              "state-space model cannot hold without internal delays. " + PADE_HINT)
        out.in_delay, out.out_delay = ind, outd
    return out


def nonzero_mask(sys: LTI) -> np.ndarray:
    """True for the input/output pairs whose transfer is not identically zero."""
    m = np.ones((sys.ny, sys.nu), dtype=bool)
    if isinstance(sys, TF):
        for i in range(sys.ny):
            for j in range(sys.nu):
                m[i, j] = not cn.piszero(sys.num[i][j])
    elif isinstance(sys, ZPK):
        m = sys.k != 0
    return m


def split_delays(total: np.ndarray, mask: np.ndarray):
    """Write a matrix of total delays as output delays + input delays + a remainder."""
    ny, nu = total.shape
    t = np.where(mask, total, np.inf)
    outd = np.min(t, axis=1) if nu else np.zeros(ny)
    outd = np.where(np.isfinite(outd), outd, 0.0)
    rem = t - outd[:, None]
    ind = np.min(rem, axis=0) if ny else np.zeros(nu)
    ind = np.where(np.isfinite(ind), ind, 0.0)
    io = np.where(mask, rem - ind[None, :], 0.0)
    io[np.abs(io) < 1e-12 * max(1.0, float(np.max(total)) if total.size else 1.0)] = 0.0
    return ind, outd, io


def absorb_delay(sys: LTI) -> LTI:
    """Discrete-time delays turned into poles at z = 0 (a delay of d samples is z^-d)."""
    if not sys.has_delay():
        return sys
    if not sys.is_dt:
        raise MatlabError("absorbDelay works on discrete-time models. For a continuous-time model use "
                          "pade(sys, N) to approximate the delays.")
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    total = np.round(sys.total_delay()).astype(int)
    if isinstance(sys, TF):
        den = _grid(sys.ny, sys.nu, lambda i, j: np.concatenate([sys.den[i][j], np.zeros(total[i, j])]))
        out: LTI = TF(sys.num, den, sys.ts, sys.var)
    elif isinstance(sys, ZPK):
        p = _grid(sys.ny, sys.nu, lambda i, j: np.concatenate([sys.p[i][j], np.zeros(total[i, j])]))
        out = ZPK(sys.z, p, sys.k, sys.ts, sys.var)
        out.display_format = sys.display_format
    elif isinstance(sys, SS):
        core = sys.abcd
        ind, outd = np.round(sys.in_delay).astype(int), np.round(sys.out_delay).astype(int)
        if np.any(ind):
            core = cn.ss_mul(core, cn.ss_append(*[_delay_chain(d) for d in ind]))
        if np.any(outd):
            core = cn.ss_mul(cn.ss_append(*[_delay_chain(d) for d in outd]), core)
        out = SS(*core, sys.ts)
    else:
        raise MatlabError("absorbDelay does not apply to frequency-response data.")
    out.copy_meta(sys)
    out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
    return out


def _delay_chain(d: int):
    """State-space model of z^-d."""
    if d == 0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.ones((1, 1))
    a = np.zeros((d, d))
    a[1:, :-1] = np.eye(d - 1)
    b = np.zeros((d, 1))
    b[0, 0] = 1.0
    c = np.zeros((1, d))
    c[0, -1] = 1.0
    return a, b, c, np.zeros((1, 1))


def lti_freqresp(sys: LTI, w, with_delay: bool = True) -> np.ndarray:
    """Frequency response (ny, nu, nw) at real frequencies ``w`` in rad per time unit."""
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    if isinstance(sys, FRD):
        return frd_interp(sys, w)
    s = np.exp(1j * w * sys.tsv()) if sys.is_dt else 1j * w
    return lti_evalfr(sys, s, with_delay)


def lti_evalfr(sys: LTI, s, with_delay: bool = True) -> np.ndarray:
    """The transfer matrix at complex points ``s`` (or z): array (ny, nu, len(s))."""
    s = np.atleast_1d(np.asarray(s, dtype=complex)).ravel()
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    if isinstance(sys, FRD):
        raise MatlabError("A frequency-response model can only be evaluated at real frequencies; use "
                          "freqresp(sys, w).")
    out = np.empty((sys.ny, sys.nu, s.size), dtype=complex)
    if isinstance(sys, TF):
        for i in range(sys.ny):
            for j in range(sys.nu):
                out[i, j, :] = cn.poly_freqresp(sys.num[i][j], sys.den[i][j], s)
    elif isinstance(sys, ZPK):
        for i in range(sys.ny):
            for j in range(sys.nu):
                out[i, j, :] = cn.zpk_freqresp(sys.z[i][j], sys.p[i][j], sys.k[i, j], s)
    else:
        out = cn.ss_freqresp(sys.a, sys.b, sys.c, sys.d, s)
    if with_delay and sys.has_delay():
        total = sys.total_delay()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if sys.is_dt:
                out = out * s[None, None, :] ** (-total[:, :, None])
            else:
                out = out * np.exp(-s[None, None, :] * total[:, :, None])
    return out


def frd_interp(sys: FRD, w: np.ndarray) -> np.ndarray:
    f = sys.rad_freq()
    order = np.argsort(f)
    f, r = f[order], sys.resp[:, :, order]
    if w.size and f.size and (w.min() < f.min() * (1 - 1e-12) or w.max() > f.max() * (1 + 1e-12)):
        raise MatlabError("The requested frequencies lie outside the range of the frequency-response data; "
                          "frd models cannot be extrapolated.")
    out = np.empty((sys.ny, sys.nu, w.size), dtype=complex)
    for i in range(sys.ny):
        for j in range(sys.nu):
            out[i, j] = np.interp(w, f, r[i, j].real) + 1j * np.interp(w, f, r[i, j].imag)
    return out


def to_frd(sys: Any, ref: FRD) -> FRD:
    if isinstance(sys, FRD):
        return sys
    if is_num(sys):
        k = as_matrix(sys, "gain")
        return ref.with_resp(np.repeat(k[:, :, None].astype(complex), ref.freq.size, axis=2))
    resp = lti_freqresp(sys, ref.rad_freq())
    out = FRD(resp, ref.freq, sys.ts if not (sys.is_static() and sys.ts == 0) else ref.ts, ref.freq_unit)
    out.copy_meta(sys)
    return out


def convert(sys: LTI, rank: int, ref: LTI | None = None) -> LTI:
    if sys.rank >= rank and not isinstance(sys, PID):
        return sys
    if rank == 5:
        return to_frd(sys, ref)  # type: ignore[arg-type]
    if rank == 4:
        return to_ss(sys)
    if rank == 3:
        return to_zpk(sys)
    return to_tf(sys)


def back_to(sys: LTI, like: LTI) -> LTI:
    """Convert a result computed in state space back to the type of the operands."""
    if isinstance(like, ZPK):
        out: LTI = to_zpk(sys)
        out.var, out.display_format = like.var, like.display_format  # type: ignore[attr-defined]
        return out
    if isinstance(like, (TF, PID)):
        out = to_tf(sys)
        if hasattr(like, "var"):
            out.var = like.var  # type: ignore[attr-defined]
        return out
    return sys


def as_lti(v: Any, like: LTI | None = None) -> LTI:
    if isinstance(v, LTI):
        return v
    if hasattr(v, "to_dense"):
        v = v.to_dense()
    if is_num(v):
        k = np.asarray(v)
        if k.ndim > 2:
            raise MatlabError("A gain must be a scalar or a matrix.")
        if k.dtype.kind in "biu":
            k = k.astype(float)
        return TF.gain(np.atleast_2d(k), like.ts if like is not None else 0.0)
    from ..values import class_name
    raise MatlabError("Operands must be control-system models (tf, zpk, ss, frd, pid) or numeric gains; got a "
                      f"value of class '{class_name(v)}'.")


def common_ts(a: LTI, b: LTI) -> float:
    ta, tb = a.ts, b.ts
    if ta == tb:
        return ta
    if ta == 0 and a.is_static():
        return tb
    if tb == 0 and b.is_static():
        return ta
    if ta == -1 and tb > 0:
        return tb
    if tb == -1 and ta > 0:
        return ta

    def say(t):
        return "continuous-time" if t == 0 else ("discrete with unspecified sample time" if t < 0
                                                 else f"discrete with Ts = {fmt_num(t)}")
    raise MatlabError(f"Sampling times must agree: one model is {say(ta)} and the other is {say(tb)}. "
                      "Convert one of them with c2d, d2c or d2d first.")


def promote_pair(a: Any, b: Any):
    """Both operands as models of the same class (the one with the higher precedence)."""
    la, lb = isinstance(a, LTI), isinstance(b, LTI)
    a2 = as_lti(a, b if lb else None)
    b2 = as_lti(b, a if la else None)
    if isinstance(a2, PID):
        a2 = pid_to_tf(a2)
    if isinstance(b2, PID):
        b2 = pid_to_tf(b2)
    rank = max(a2.rank, b2.rank)
    ref = a2 if isinstance(a2, FRD) else b2
    return convert(a2, rank, ref), convert(b2, rank, ref)


def _pid_template(a: Any, b: Any) -> PID | None:
    """The controller whose type a result should keep (both operands PID, or PID with a number)."""
    pa, pb = isinstance(a, PID), isinstance(b, PID)
    if pa and (pb or (is_num(b) and np.asarray(b).size == 1)):
        return a
    if pb and is_num(a) and np.asarray(a).size == 1:
        return b
    return None


def _finish_pid(result: LTI, template: PID | None) -> LTI:
    if template is None or not isinstance(result, TF):
        return result
    try:
        c = tf_to_pid(result, template, isinstance(template, PIDSTD))
    except MatlabError:
        c = None
    return c if c is not None else result


def _var_of(a: LTI, b: LTI, ts: float) -> str:
    dt_vars = ("z", "q", "z^-1", "q^-1")
    for static_ok in (False, True):
        for s in (a, b):
            v = getattr(s, "var", None)
            if v and (v in dt_vars) == (ts != 0) and (static_ok or not s.is_static()):
                return v
    return "z" if ts != 0 else "s"


def _names_merge(x: list[str], y: list[str]) -> list[str]:
    if len(x) != len(y):
        return x if any(x) else y
    return [p if p == q or not q else (q if not p else "") for p, q in zip(x, y)]


def _expand(a: LTI, ny: int, nu: int) -> LTI:
    """A SISO model repeated into an ny-by-nu model (every entry equal)."""
    return lti_vcat([lti_hcat([a] * nu)] * ny) if (ny, nu) != (1, 1) else a


def _need_no_delay(what: str, *systems: LTI) -> list:
    """Delays must be gone before an operation that would put them inside a loop."""
    out = []
    for s in systems:
        if s.has_delay():
            if s.is_dt:
                s = absorb_delay(s)
            else:
                raise MatlabError(f"{what} with time delays would need internal delays, which are not "
                                  "supported. " + PADE_HINT)
        out.append(s)
    return out


# -- arithmetic -------------------------------------------------------------------------------

def lti_neg(a: LTI) -> LTI:
    out = a.clone()
    if isinstance(a, PIDSTD):
        out.kp = -a.kp
    elif isinstance(a, PID):
        out.kp, out.ki, out.kd = -a.kp, -a.ki, -a.kd
    elif isinstance(a, TF):
        out.num = [[-n for n in r] for r in a.num]
    elif isinstance(a, ZPK):
        out.k = -a.k
    elif isinstance(a, SS):
        out.c, out.d = -a.c, -a.d
    elif isinstance(a, FRD):
        out.resp = -a.resp
    return out


def _same(x: np.ndarray, y: np.ndarray) -> bool:
    return x.shape == y.shape and bool(np.all(np.abs(x - y) <= 1e-12 * np.maximum(1.0, np.abs(x))))


def _same_freq(a: FRD, b: FRD) -> None:
    fa, fb = a.rad_freq(), b.rad_freq()
    if fa.size != fb.size or not np.allclose(fa, fb, rtol=1e-10, atol=0):
        raise MatlabError("The two frequency-response models are sampled at different frequencies; "
                          "interpolate one of them first with interp(sys, w).")


def lti_add(x: Any, y: Any, sign: float = 1.0) -> LTI:
    template = _pid_template(x, y)
    a, b = promote_pair(x, y)
    ts = common_ts(a, b)
    if a.is_siso and not b.is_siso:
        a = _expand(a, b.ny, b.nu)
    elif b.is_siso and not a.is_siso:
        b = _expand(b, a.ny, a.nu)
    if (a.ny, a.nu) != (b.ny, b.nu):
        raise MatlabError(f"Models with different numbers of inputs and outputs cannot be added: one is "
                          f"{a.ny}-by-{a.nu} and the other {b.ny}-by-{b.nu}.")
    if isinstance(a, FRD):
        _same_freq(a, b)
        out: LTI = a.with_resp(a.resp + sign * b.resp)
        out.ts = ts
    else:
        ma, mb = nonzero_mask(a), nonzero_mask(b)
        if a.has_delay() or b.has_delay():
            ta, tb = a.total_delay(), b.total_delay()
            clash = bool(np.any(ma & mb & (np.abs(ta - tb) > 1e-12 * np.maximum(1.0, np.abs(ta)))))
            if isinstance(a, SS):
                clash = not (_same(a.in_delay, b.in_delay) and _same(a.out_delay, b.out_delay))
                if clash and _same(ta, tb):   # the same delays, split differently between input and output
                    b = b.clone()
                    b.in_delay, b.out_delay = a.in_delay, a.out_delay
                    clash = False
            if clash:
                a, b = _need_no_delay("Adding models", a, b)
        ta, tb = a.total_delay(), b.total_delay()
        total = np.where(ma, ta, tb)
        if isinstance(a, TF):
            num = _grid(a.ny, a.nu, lambda i, j: None)
            den = _grid(a.ny, a.nu, lambda i, j: None)
            for i in range(a.ny):
                for j in range(a.nu):
                    num[i][j], den[i][j] = _tf_elem_add(a.num[i][j], a.den[i][j], b.num[i][j], b.den[i][j], sign)
            out = TF(num, den, ts, _var_of(a, b, ts))
        elif isinstance(a, ZPK):
            z = _grid(a.ny, a.nu, lambda i, j: None)
            p = _grid(a.ny, a.nu, lambda i, j: None)
            k = np.zeros((a.ny, a.nu), dtype=np.result_type(a.k, b.k))
            for i in range(a.ny):
                for j in range(a.nu):
                    z[i][j], p[i][j], k[i, j] = _zpk_elem_add(a.z[i][j], a.p[i][j], a.k[i, j], b.z[i][j],
                                                              b.p[i][j], sign * b.k[i, j])
            out = ZPK(z, p, k, ts, _var_of(a, b, ts))
            out.display_format = a.display_format
        else:
            out = SS(*cn.ss_add(a.abcd, b.abcd, sign), ts)
            out.state_name = list(a.state_name) + list(b.state_name)
        if isinstance(out, SS):
            out.in_delay, out.out_delay = a.in_delay, a.out_delay
        else:
            ind = a.in_delay if _same(a.in_delay, b.in_delay) else np.zeros(a.nu)
            outd = a.out_delay if _same(a.out_delay, b.out_delay) else np.zeros(a.ny)
            io = total - outd[:, None] - ind[None, :]
            if np.any(io < -1e-12):
                ind, outd, io = np.zeros(a.nu), np.zeros(a.ny), total
            out.in_delay, out.out_delay, out.io_delay = ind, outd, np.maximum(io, 0.0)
    out.in_name, out.out_name = _names_merge(a.in_name, b.in_name), _names_merge(a.out_name, b.out_name)
    out.time_unit = a.time_unit
    return _finish_pid(out, template)


def _zpk_elem_add(z1, p1, k1, z2, p2, k2):
    if k2 == 0:
        return z1, p1, k1
    if k1 == 0:
        return z2, p2, k2
    z1, p1, z2, p2 = (np.asarray(v) for v in (z1, p1, z2, p2))
    if p1.size == p2.size and np.array_equal(np.sort_complex(p1.astype(complex)),
                                             np.sort_complex(p2.astype(complex))):
        num = cn.padd(k1 * cn.rpoly(z1), k2 * cn.rpoly(z2))
        p = p1
    else:
        num = cn.padd(k1 * cn.pmul(cn.rpoly(z1), cn.rpoly(p2)), k2 * cn.pmul(cn.rpoly(z2), cn.rpoly(p1)))
        p = np.concatenate([p1, p2])
    if cn.piszero(num):
        return np.zeros(0), np.zeros(0), 0.0
    roots = cn.proots(num)
    return (roots if np.iscomplexobj(num) else cn.pair_conjugates(roots)), p, num[0]


def lti_mul(x: Any, y: Any) -> LTI:
    """The product x*y: y acts first, then x."""
    template = _pid_template(x, y)
    a, b = promote_pair(x, y)
    ts = common_ts(a, b)
    if a.nu != b.ny:
        if a.is_siso:
            if isinstance(a, SS):
                if b.ny <= b.nu:
                    a = lti_append([a] * b.ny)
                else:
                    return lti_mul(b, lti_append([a] * b.nu))
            else:
                return lti_times(_expand(a, b.ny, b.nu), b)
        elif b.is_siso:
            if isinstance(b, SS):
                if a.nu <= a.ny:
                    b = lti_append([b] * a.nu)
                else:
                    return lti_mul(lti_append([b] * a.ny), a)
            else:
                return lti_times(a, _expand(b, a.ny, a.nu))
        else:
            raise MatlabError(f"Model dimensions do not agree for multiplication: the left model has {a.nu} "
                              f"input(s) but the right model has {b.ny} output(s).")
    ny, nu, nk = a.ny, b.nu, a.nu
    if isinstance(a, FRD):
        _same_freq(a, b)
        out: LTI = a.with_resp(np.einsum("ikf,kjf->ijf", a.resp, b.resp))
        out.ts = ts
    elif isinstance(a, SS):
        mid = a.in_delay + b.out_delay
        if nk and not _same(mid, np.full(nk, mid[0])):
            a, b = _need_no_delay("Connecting these models in series", a, b)
            mid = np.zeros(nk)
        out = SS(*cn.ss_mul(a.abcd, b.abcd), ts)
        out.state_name = list(a.state_name) + list(b.state_name)
        out.in_delay = b.in_delay + (mid[0] if nk else 0.0)
        out.out_delay = a.out_delay
    else:
        total = np.zeros((ny, nu))
        if a.has_delay() or b.has_delay():
            ta, tb = a.total_delay(), b.total_delay()
            ma, mb = nonzero_mask(a), nonzero_mask(b)
            ok = True
            for i in range(ny):
                for j in range(nu):
                    vals = [ta[i, k] + tb[k, j] for k in range(nk) if ma[i, k] and mb[k, j]]
                    if vals:
                        total[i, j] = vals[0]
                        if max(vals) - min(vals) > 1e-12 * max(1.0, max(vals)):
                            ok = False
            if not ok:
                a, b = _need_no_delay("Multiplying these models", a, b)
                total = np.zeros((ny, nu))
        if isinstance(a, TF):
            num = _grid(ny, nu, lambda i, j: np.zeros(1))
            den = _grid(ny, nu, lambda i, j: np.ones(1))
            for i in range(ny):
                for j in range(nu):
                    for k in range(nk):
                        n, d = _tf_elem_mul(a.num[i][k], a.den[i][k], b.num[k][j], b.den[k][j])
                        num[i][j], den[i][j] = _tf_elem_add(num[i][j], den[i][j], n, d)
            out = TF(num, den, ts, _var_of(a, b, ts))
            if ny == 0:
                out._nu = nu
        else:
            z = _grid(ny, nu, lambda i, j: np.zeros(0))
            p = _grid(ny, nu, lambda i, j: np.zeros(0))
            kk = np.zeros((ny, nu), dtype=np.result_type(a.k, b.k))
            for i in range(ny):
                for j in range(nu):
                    for k in range(nk):
                        g = a.k[i, k] * b.k[k, j]
                        if g == 0:
                            continue
                        zz = np.concatenate([np.asarray(a.z[i][k]), np.asarray(b.z[k][j])])
                        pp = np.concatenate([np.asarray(a.p[i][k]), np.asarray(b.p[k][j])])
                        z[i][j], p[i][j], kk[i, j] = _zpk_elem_add(z[i][j], p[i][j], kk[i, j], zz, pp, g)
            out = ZPK(z, p, kk, ts, _var_of(a, b, ts))
            out.display_format = a.display_format
        if a.is_siso and b.is_siso:
            out.in_delay = a.in_delay + b.in_delay
            out.out_delay = a.out_delay + b.out_delay
            out.io_delay = a.io_delay + b.io_delay
        else:
            outd, ind = a.out_delay, b.in_delay
            io = total - outd[:, None] - ind[None, :]
            out.in_delay, out.out_delay, out.io_delay = ind, outd, np.where(io < 0, 0.0, io)
    out.in_name, out.in_unit = list(b.in_name), list(b.in_unit)
    out.out_name, out.out_unit = list(a.out_name), list(a.out_unit)
    out.in_group, out.out_group = dict(b.in_group), dict(a.out_group)
    out.time_unit = a.time_unit
    return _finish_pid(out, template)


def lti_times(x: Any, y: Any, divide: bool = False) -> LTI:
    """Element-by-element product (or quotient) of two models of the same size."""
    a, b = promote_pair(x, y)
    ts = common_ts(a, b)
    if a.is_siso and not b.is_siso:
        a = _expand(a, b.ny, b.nu)
    elif b.is_siso and not a.is_siso:
        b = _expand(b, a.ny, a.nu)
    if (a.ny, a.nu) != (b.ny, b.nu):
        raise MatlabError("Element-wise operations need models of the same size (or a SISO model).")
    if isinstance(a, FRD):
        _same_freq(a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            out: LTI = a.with_resp(a.resp / b.resp if divide else a.resp * b.resp)
        out.ts = ts
        return out
    if a.is_siso:
        return lti_mul(a, lti_inv(b)) if divide else lti_mul(a, b)
    rows = []
    for i in range(a.ny):
        entries = []
        for j in range(a.nu):
            p, q = a.sub(np.array([i]), np.array([j])), b.sub(np.array([i]), np.array([j]))
            entries.append(lti_mul(p, lti_inv(q)) if divide else lti_mul(p, q))
        rows.append(lti_hcat(entries))
    out = lti_vcat(rows)
    out.in_name, out.out_name = _names_merge(a.in_name, b.in_name), _names_merge(a.out_name, b.out_name)
    return out


def lti_inv(x: Any) -> LTI:
    if is_num(x):
        m = as_matrix(x, "gain")
        if m.shape[0] != m.shape[1]:
            raise MatlabError("Only square gains can be inverted.")
        return TF.gain(np.linalg.inv(m))
    a = as_lti(x)
    if isinstance(a, PID):
        a = pid_to_tf(a)
    if a.ny != a.nu:
        raise MatlabError(f"Only square models can be inverted; this one has {a.ny} output(s) and "
                          f"{a.nu} input(s).")
    if a.has_delay():
        raise MatlabError("A model with time delays cannot be inverted: the inverse would have to "
                          "predict the future.")
    if isinstance(a, FRD):
        resp = np.empty_like(a.resp)
        for f in range(a.freq.size):
            resp[:, :, f] = np.linalg.inv(a.resp[:, :, f])
        out: LTI = a.with_resp(resp)
    elif isinstance(a, TF) and a.is_siso:
        if cn.piszero(a.num[0][0]):
            raise MatlabError("Cannot invert a transfer function that is identically zero.")
        out = TF.siso(a.den[0][0], a.num[0][0], a.ts, a.var)
    elif isinstance(a, ZPK) and a.is_siso:
        if a.k[0, 0] == 0:
            raise MatlabError("Cannot invert a model that is identically zero.")
        out = ZPK.siso(a.p[0][0], a.z[0][0], 1.0 / a.k[0, 0], a.ts, a.var)
        out.display_format = a.display_format
    elif isinstance(a, SS):
        out = SS(*cn.ss_inv(a.abcd), a.ts)
    else:
        out = back_to(SS(*cn.ss_inv(to_ss(a).abcd), a.ts), a)
    out.in_name, out.out_name = list(a.out_name), list(a.in_name)
    out.in_unit, out.out_unit = list(a.out_unit), list(a.in_unit)
    out.time_unit = a.time_unit
    return out


def lti_power(x: Any, n: Any, elementwise: bool = False) -> LTI:
    if not isinstance(x, LTI) or not is_num(n) or np.asarray(n).size != 1:
        raise MatlabError("Only integer powers of a model are defined, for example s^2 or sys^-1.")
    k = float(np.real(np.asarray(n).flat[0]))
    if k != np.round(k):
        raise MatlabError("The exponent of a model must be an integer.")
    k = int(k)
    if not elementwise and x.ny != x.nu:
        raise MatlabError("Only square models can be raised to a power; use .^ for element-wise powers.")
    if elementwise and not x.is_siso:
        rows = [lti_hcat([lti_power(x.sub(np.array([i]), np.array([j])), n) for j in range(x.nu)])
                for i in range(x.ny)]
        return lti_vcat(rows)
    if k == 0:
        one = as_lti(np.eye(x.ny), x)
        return one if isinstance(x, PID) else promote_pair(x, one)[1]
    base = lti_inv(x) if k < 0 else x
    k = abs(k)
    result: LTI | None = None
    while k:
        if k & 1:
            result = base if result is None else lti_mul(result, base)
        k >>= 1
        if k:
            base = lti_mul(base, base)
    return result  # type: ignore[return-value]


def lti_transpose(a: LTI) -> LTI:
    if isinstance(a, PID):
        return a
    rows, cols_ = range(a.ny), range(a.nu)
    if isinstance(a, TF):
        out: LTI = TF([[a.num[i][j] for i in rows] for j in cols_], [[a.den[i][j] for i in rows] for j in cols_],
                      a.ts, a.var)
        out._nu = a.ny
    elif isinstance(a, ZPK):
        out = ZPK([[a.z[i][j] for i in rows] for j in cols_], [[a.p[i][j] for i in rows] for j in cols_],
                  a.k.T, a.ts, a.var)
        out.display_format = a.display_format
    elif isinstance(a, SS):
        out = SS(a.a.T, a.c.T, a.b.T, a.d.T, a.ts)
        out.state_name = list(a.state_name)
    else:
        out = a.with_resp(np.transpose(a.resp, (1, 0, 2)))  # type: ignore[attr-defined]
    out.in_delay, out.out_delay, out.io_delay = a.out_delay, a.in_delay, a.io_delay.T
    out.in_name, out.out_name = list(a.out_name), list(a.in_name)
    out.in_unit, out.out_unit = list(a.out_unit), list(a.in_unit)
    out.time_unit = a.time_unit
    return out


def lti_ctranspose(a: LTI) -> LTI:
    """The conjugate system G(-s)' in continuous time, G(1/z)' in discrete time."""
    if a.has_delay():
        raise MatlabError("The conjugate of a model with delays would have to predict the future.")
    if isinstance(a, PID):
        a = pid_to_tf(a)
    t = lti_transpose(a)
    if isinstance(t, FRD):
        return t.with_resp(np.conj(t.resp))
    if isinstance(t, SS):
        if not t.is_dt:
            return _carry(SS(-t.a.conj(), -t.b.conj(), t.c.conj(), t.d.conj(), 0.0), t)
        if t.nx and np.linalg.matrix_rank(t.a) < t.nx:
            raise MatlabError("The conjugate of a discrete-time model with poles at z = 0 is not causal and "
                              "has no state-space form.")
        # with (A, B, C, D) of the transposed model:
        #   D + C (z^-1 I - A)^-1 B = D - C A^-1 B - C A^-1 (zI - A^-1)^-1 A^-1 B
        ai = np.linalg.inv(t.a.conj()) if t.nx else t.a
        return _carry(SS(ai, ai @ t.b.conj(), -t.c.conj() @ ai, t.d.conj() - t.c.conj() @ ai @ t.b.conj(), t.ts), t)
    if isinstance(t, ZPK):
        if not t.is_dt:
            out = t.clone()
            out.z = [[-np.conj(v) for v in r] for r in t.z]
            out.p = [[-np.conj(v) for v in r] for r in t.p]
            k = np.conj(t.k).copy()
            for i in range(t.ny):
                for j in range(t.nu):
                    k[i, j] *= (-1.0) ** (len(t.z[i][j]) - len(t.p[i][j]))
            out.k = k
            return out
        res = to_zpk(lti_ctranspose(to_tf(a)))
        res.display_format = t.display_format
        return res
    num = _grid(t.ny, t.nu, lambda i, j: None)
    den = _grid(t.ny, t.nu, lambda i, j: None)
    for i in range(t.ny):
        for j in range(t.nu):
            n, d = np.conj(t.num[i][j]), np.conj(t.den[i][j])
            if not t.is_dt:
                n = n * (-1.0) ** np.arange(n.size - 1, -1, -1)
                d = d * (-1.0) ** np.arange(d.size - 1, -1, -1)
            else:  # p(1/z) = z^-deg * reversed(p)(z)
                dn, dd = n.size - 1, d.size - 1
                n, d = n[::-1], d[::-1]
                if dd > dn:
                    n = np.concatenate([n, np.zeros(dd - dn)])
                elif dn > dd:
                    d = np.concatenate([d, np.zeros(dn - dd)])
            num[i][j], den[i][j] = n, d
    res2 = TF(num, den, t.ts, t.var)
    res2._nu = t.nu
    return _carry(res2, t)


# -- concatenation ----------------------------------------------------------------------------

def _common_kind(items: list) -> list:
    models = [as_lti(i) for i in items]
    models = [pid_to_tf(m) if isinstance(m, PID) else m for m in models]
    rank = max(m.rank for m in models)
    ref = next((m for m in models if isinstance(m, FRD)), None)
    return [convert(m, rank, ref) for m in models]


def _concat_ts(models: list) -> float:
    ref = models[0]
    ts = ref.ts
    for m in models[1:]:
        ts = common_ts(ref, m)
        if ref.is_static() and ref.ts == 0 or ref.ts == -1:
            ref = m
    return ts


def lti_hcat(items: list) -> LTI:
    models = _common_kind(items)
    if len(models) == 1:
        return models[0]
    ny = models[0].ny
    if any(m.ny != ny for m in models):
        raise MatlabError("Models placed side by side must have the same number of outputs.")
    ts = _concat_ts(models)
    first = models[0]
    out_d = np.min(np.array([m.out_delay for m in models]), axis=0) if ny else np.zeros(0)
    if isinstance(first, FRD):
        for m in models[1:]:
            _same_freq(first, m)
        out: LTI = first.with_resp(np.concatenate([m.resp for m in models], axis=1))
        out.ts = ts
    elif isinstance(first, SS):
        fixed = []
        for m in models:
            extra = m.out_delay - out_d
            if np.any(extra):
                if _same(extra, np.full(ny, extra[0])):
                    m = m.clone()
                    m.in_delay, m.out_delay = m.in_delay + extra[0], out_d
                else:
                    m = _need_no_delay("Concatenating these models", m)[0]
            fixed.append(m)
        if any(not _same(m.out_delay, fixed[0].out_delay) for m in fixed):
            fixed = _need_no_delay("Concatenating these models", *fixed)
        models = fixed
        out = SS(cn.blkdiag(*[m.a for m in models]), cn.blkdiag(*[m.b for m in models]),
                 np.hstack([m.c for m in models]), np.hstack([m.d for m in models]), ts)
        out.state_name = [n for m in models for n in m.state_name]
        out.in_delay = np.concatenate([m.in_delay for m in models])
        out.out_delay = models[0].out_delay
    else:
        if isinstance(first, TF):
            out = TF([[m.num[i][j] for m in models for j in range(m.nu)] for i in range(ny)],
                     [[m.den[i][j] for m in models for j in range(m.nu)] for i in range(ny)], ts,
                     _var_of(first, models[1], ts))
            out._nu = sum(m.nu for m in models)
        else:
            out = ZPK([[m.z[i][j] for m in models for j in range(m.nu)] for i in range(ny)],
                      [[m.p[i][j] for m in models for j in range(m.nu)] for i in range(ny)],
                      np.hstack([m.k for m in models]), ts, _var_of(first, models[1], ts))
            out.display_format = first.display_format
        out.in_delay = np.concatenate([m.in_delay for m in models])
        out.out_delay = out_d
        out.io_delay = np.hstack([m.io_delay + (m.out_delay - out_d)[:, None] for m in models])
    out.in_name = [n for m in models for n in m.in_name]
    out.in_unit = [n for m in models for n in m.in_unit]
    names = models[0].out_name
    for m in models[1:]:
        names = _names_merge(names, m.out_name)
    out.out_name = names
    out.time_unit = first.time_unit
    return out


def lti_vcat(items: list) -> LTI:
    models = _common_kind(items)
    if len(models) == 1:
        return models[0]
    return lti_transpose(lti_hcat([lti_transpose(m) for m in models]))


def lti_append(items: list) -> LTI:
    models = _common_kind(items)
    if len(models) == 1:
        return models[0]
    ts = _concat_ts(models)
    first = models[0]
    ny, nu = sum(m.ny for m in models), sum(m.nu for m in models)
    if isinstance(first, FRD):
        for m in models[1:]:
            _same_freq(first, m)
        resp = np.zeros((ny, nu, first.freq.size), dtype=complex)
        i = j = 0
        for m in models:
            resp[i:i + m.ny, j:j + m.nu] = m.resp
            i, j = i + m.ny, j + m.nu
        out: LTI = first.with_resp(resp)
        out.ts = ts
    elif isinstance(first, SS):
        out = SS(*cn.ss_append(*[m.abcd for m in models]), ts)
        out.state_name = [n for m in models for n in m.state_name]
    else:
        owner = [(k, i) for k, m in enumerate(models) for i in range(m.ny)]
        cols_ = [(k, j) for k, m in enumerate(models) for j in range(m.nu)]
        if isinstance(first, TF):
            out = TF([[models[a].num[i][j] if a == b else np.zeros(1) for (b, j) in cols_] for (a, i) in owner],
                     [[models[a].den[i][j] if a == b else np.ones(1) for (b, j) in cols_] for (a, i) in owner], ts,
                     _var_of(first, models[1], ts))
            out._nu = nu
        else:
            k = cn.blkdiag(*[m.k for m in models])
            out = ZPK([[models[a].z[i][j] if a == b else np.zeros(0) for (b, j) in cols_] for (a, i) in owner],
                      [[models[a].p[i][j] if a == b else np.zeros(0) for (b, j) in cols_] for (a, i) in owner], k,
                      ts, _var_of(first, models[1], ts))
            out.display_format = first.display_format
        out.io_delay = cn.blkdiag(*[m.io_delay for m in models]).real
    out.in_delay = np.concatenate([m.in_delay for m in models])
    out.out_delay = np.concatenate([m.out_delay for m in models])
    out.in_name = [n for m in models for n in m.in_name]
    out.out_name = [n for m in models for n in m.out_name]
    out.in_unit = [n for m in models for n in m.in_unit]
    out.out_unit = [n for m in models for n in m.out_unit]
    out.time_unit = first.time_unit
    return out


# -- feedback and other interconnections ------------------------------------------------------

def lti_feedback(g: Any, h: Any, sign: float = -1.0, feedin=None, feedout=None) -> LTI:
    """Closed loop of ``g`` with ``h`` in the return path (negative feedback when sign = -1)."""
    a, b = promote_pair(g, h)
    ts = common_ts(a, b)
    fin = np.arange(a.nu) if feedin is None else np.asarray(feedin, dtype=int)
    fout = np.arange(a.ny) if feedout is None else np.asarray(feedout, dtype=int)
    if b.nu != fout.size or b.ny != fin.size:
        raise MatlabError(f"The feedback model must have {fout.size} input(s) and {fin.size} output(s) to "
                          f"close the loop; it has {b.nu} and {b.ny}.")
    if isinstance(a, FRD):
        _same_freq(a, b)
        resp = np.empty_like(a.resp)
        ein = np.zeros((a.nu, b.ny), dtype=complex)
        ein[fin, np.arange(fin.size)] = 1.0
        for f in range(a.freq.size):
            ga, hb = a.resp[:, :, f], b.resp[:, :, f]
            loop = np.eye(a.nu) - sign * ein @ hb @ ga[fout, :]
            resp[:, :, f] = ga @ np.linalg.inv(loop)
        out: LTI = a.with_resp(resp)
        out.ts = ts
    else:
        a, b = _need_no_delay("Closing a feedback loop", a, b)
        full = feedin is None and feedout is None
        if full and a.is_siso and isinstance(a, TF):
            n1, d1, n2, d2 = a.num[0][0], a.den[0][0], b.num[0][0], b.den[0][0]
            den = cn.padd(cn.pmul(d1, d2), -sign * cn.pmul(n1, n2))
            if cn.piszero(den):
                raise MatlabError("The feedback loop is singular: 1 - sign*G*H is identically zero.")
            out = TF.siso(cn.pmul(n1, d2), den, ts, _var_of(a, b, ts))
        elif full and a.is_siso and isinstance(a, ZPK):
            k1, k2 = a.k[0, 0], b.k[0, 0]
            z1, p1, z2, p2 = (np.asarray(v) for v in (a.z[0][0], a.p[0][0], b.z[0][0], b.p[0][0]))
            den = cn.padd(cn.pmul(cn.rpoly(p1), cn.rpoly(p2)),
                          -sign * k1 * k2 * cn.pmul(cn.rpoly(z1), cn.rpoly(z2)))
            if cn.piszero(den):
                raise MatlabError("The feedback loop is singular: 1 - sign*G*H is identically zero.")
            poles = cn.proots(den)
            poles = poles if np.iscomplexobj(den) else cn.pair_conjugates(poles)
            out = ZPK.siso(np.concatenate([z1, p2]), poles, k1 / den[0], ts, _var_of(a, b, ts))
            out.display_format = a.display_format
        else:
            sa, sb = to_ss(a), to_ss(b)
            if full:
                res = cn.ss_feedback(sa.abcd, sb.abcd, sign)
            else:
                appended = cn.ss_append(sa.abcd, sb.abcd)
                nyt, nut = sa.ny + sb.ny, sa.nu + sb.nu
                k = np.zeros((nut, nyt))
                k[fin, sa.ny + np.arange(sb.ny)] = sign
                k[sa.nu + np.arange(sb.nu), fout] = 1.0
                ein2 = np.zeros((nut, sa.nu))
                ein2[:sa.nu, :] = np.eye(sa.nu)
                eout = np.zeros((sa.ny, nyt))
                eout[:, :sa.ny] = np.eye(sa.ny)
                res = cn.ss_interconnect(appended, k, ein2, eout)
            ss_out = SS(*res, ts)
            ss_out.state_name = list(sa.state_name) + list(sb.state_name)
            out = back_to(ss_out, a)
    out.in_name, out.out_name = list(a.in_name), list(a.out_name)
    out.in_unit, out.out_unit = list(a.in_unit), list(a.out_unit)
    out.in_group, out.out_group = dict(a.in_group), dict(a.out_group)
    out.time_unit = a.time_unit
    return out


def lti_lft(x: Any, y: Any, nu: int | None = None, ny: int | None = None) -> LTI:
    a, b = promote_pair(x, y)
    ts = common_ts(a, b)
    if isinstance(a, FRD):
        raise MatlabError("lft is not available for frequency-response data.")
    lower = True
    if nu is None or ny is None:
        if a.nu >= b.ny and a.ny >= b.nu:    # b closes the last channels of a
            nu, ny = b.ny, b.nu
        elif b.nu >= a.ny and b.ny >= a.nu:  # a closes the first channels of b
            lower = False
        else:
            raise MatlabError("lft: the dimensions of the two models do not match.")
    a, b = _need_no_delay("lft", a, b)
    sa, sb = to_ss(a), to_ss(b)
    if lower:
        if sb.nu != ny or sb.ny != nu:
            # only part of the second model takes part: its first ny inputs and first nu outputs
            if nu > min(sa.nu, sb.ny) or ny > min(sa.ny, sb.nu):  # type: ignore[operator]
                raise MatlabError("lft: nu and ny exceed the number of inputs or outputs of the models.")
            return _lft_partial(sa, sb, int(nu), int(ny), ts, a)  # type: ignore[arg-type]
        out_ss = SS(*cn.ss_lft(sa.abcd, sb.abcd, int(nu), int(ny)), ts)  # type: ignore[arg-type]
        out_ss.in_name, out_ss.out_name = sa.in_name[:sa.nu - int(nu)], sa.out_name[:sa.ny - int(ny)]  # type: ignore
    else:
        pin = np.concatenate([np.arange(sa.ny, sb.nu), np.arange(sa.ny)]).astype(int)
        pout = np.concatenate([np.arange(sa.nu, sb.ny), np.arange(sa.nu)]).astype(int)
        sbp = (sb.a, sb.b[:, pin], sb.c[pout, :], sb.d[np.ix_(pout, pin)])
        out_ss = SS(*cn.ss_lft(sbp, sa.abcd, sa.ny, sa.nu), ts)
        out_ss.in_name, out_ss.out_name = sb.in_name[sa.ny:], sb.out_name[sa.nu:]
    return back_to(out_ss, a)


def _lft_partial(sa: SS, sb: SS, nu: int, ny: int, ts: float, like: LTI) -> LTI:
    """lft(sys1, sys2, nu, ny): the last nu inputs / ny outputs of sys1 meet the first ny inputs /
    nu outputs of sys2; the remaining channels of both models stay open."""
    appended = cn.ss_append(sa.abcd, sb.abcd)
    nyt, nut = sa.ny + sb.ny, sa.nu + sb.nu
    k = np.zeros((nut, nyt))
    k[sa.nu - nu + np.arange(nu), sa.ny + np.arange(nu)] = 1.0      # u1 (last nu) = y2 (first nu)
    k[sa.nu + np.arange(ny), sa.ny - ny + np.arange(ny)] = 1.0      # u2 (first ny) = y1 (last ny)
    ext_in = list(range(sa.nu - nu)) + list(range(sa.nu + ny, nut))
    ext_out = list(range(sa.ny - ny)) + list(range(sa.ny + nu, nyt))
    ein = np.zeros((nut, len(ext_in)))
    ein[ext_in, np.arange(len(ext_in))] = 1.0
    eout = np.zeros((len(ext_out), nyt))
    eout[np.arange(len(ext_out)), ext_out] = 1.0
    out = SS(*cn.ss_interconnect(appended, k, ein, eout), ts)
    names_in = sa.in_name + sb.in_name
    names_out = sa.out_name + sb.out_name
    out.in_name = [names_in[i] for i in ext_in]
    out.out_name = [names_out[i] for i in ext_out]
    return back_to(out, like)


# -- display ----------------------------------------------------------------------------------

def poly_str(c: np.ndarray, var: str) -> str:
    """A polynomial as text, e.g. 's^2 + 3 s + 2' (or '1 + 0.5 z^-1' for inverse-power variables)."""
    c = np.atleast_1d(c)
    inverse = var.endswith("^-1")
    base = var[:-3] if inverse else var
    n = c.size - 1
    parts: list[str] = []
    for k, coef in enumerate(c):
        if coef == 0:
            continue
        power = k if inverse else n - k
        if np.iscomplexobj(coef) and np.imag(coef) != 0:
            mag, neg = fmt_num(complex(coef)), False
        else:
            v = float(np.real(coef))
            mag, neg = fmt_num(abs(v)), v < 0
        if power == 0:
            term = mag
        else:
            pw = (f"{base}^-{power}" if inverse else (base if power == 1 else f"{base}^{power}"))
            term = pw if mag == "1" else f"{mag} {pw}"
        if not parts:
            parts.append(("-" if neg else "") + term)
        else:
            parts.append((" - " if neg else " + ") + term)
    return "".join(parts) if parts else "0"


def _delay_prefix(sys: LTI, d: float) -> str:
    if d == 0:
        return ""
    if sys.is_dt:
        var = getattr(sys, "var", "z")
        base = var[:-3] if var.endswith("^-1") else var
        return f"{base}^(-{fmt_num(d)}) * "
    return f"exp(-{fmt_num(d)}*{getattr(sys, 'var', 's')}) * "


def _fraction_lines(num: str, den: str | None, prefix: str = "") -> list[str]:
    if den is None:
        return [prefix + num]
    width = max(len(num), len(den))
    pad = " " * len(prefix)
    return [pad + " " * ((width - len(num)) // 2) + num, prefix + "-" * width,
            pad + " " * ((width - len(den)) // 2) + den]


def _chan_name(names: list[str], k: int) -> str:
    return f'"{names[k]}"' if names[k] else str(k + 1)


def _io_display(sys: LTI, entry: Callable) -> list[str]:
    """Lay out one block of text lines per input/output pair, the way transfer functions are shown."""
    lines: list[str] = []
    if sys.ny == 0 or sys.nu == 0:
        return [f"  Empty {_kind_words(sys)}."]
    if sys.is_siso and not sys.in_name[0] and not sys.out_name[0]:
        return ["  " + ln for ln in entry(0, 0)] + [" "]
    for j in range(sys.nu):
        head = f"  From input {_chan_name(sys.in_name, j)} to output"
        if sys.ny == 1:
            lines.append(f"{head} {_chan_name(sys.out_name, 0)}:" if sys.out_name[0] else f"{head}:")
            lines += ["  " + ln for ln in entry(0, j)] + [" "]
            continue
        lines.append(head + "...")
        labels = [(sys.out_name[i] or str(i + 1)) + ":  " for i in range(sys.ny)]
        w = max(len(s) for s in labels)
        for i in range(sys.ny):
            block = entry(i, j)
            mid = len(block) // 2
            for r, ln in enumerate(block):
                lab = labels[i].rjust(w) if r == mid else " " * w
                lines.append("   " + lab + ln)
            lines.append(" ")
    return lines


def _kind_words(sys: LTI) -> str:
    return {"tf": "transfer function", "zpk": "zero/pole/gain model", "ss": "state-space model",
            "frd": "frequency response"}.get(sys.kind, "model")


def _footer(sys: LTI, static: bool) -> list[str]:
    if static:
        return ["Static gain."]
    return sys.ts_line() + [f"{sys.time_label()} {_kind_words(sys)}."]


def _tf_display(g: TF) -> list[str]:
    total = g.total_delay()

    def entry(i: int, j: int) -> list[str]:
        prefix = _delay_prefix(g, total[i, j])
        if g.var.endswith("^-1"):
            pn, pd = g.padded(i, j)
            if cn.piszero(pn):
                return ["0"]
            num, den = poly_str(pn, g.var), poly_str(pd, g.var)
        else:
            if cn.piszero(g.num[i][j]):
                return ["0"]
            num, den = poly_str(g.num[i][j], g.var), poly_str(g.den[i][j], g.var)
        return _fraction_lines(num, None if den == "1" else den, prefix)

    return _io_display(g, entry) + _footer(g, g.is_static())


def _group_roots(r) -> list:
    """Roots grouped as (value, multiplicity, is_pair): conjugate pairs become one quadratic factor."""
    r = cn.pair_conjugates(np.asarray(r))
    out: list = []
    used = np.zeros(len(r), dtype=bool)
    for i, v in enumerate(r):
        if used[i]:
            continue
        used[i] = True
        pair = False
        if np.imag(v) != 0:
            for j in range(len(r)):
                if not used[j] and r[j] == np.conj(v):
                    used[j] = True
                    pair = True
                    break
        key = complex(v) if not pair else complex(np.real(v), abs(np.imag(v)))
        for item in out:
            if item[2] == pair and abs(item[0] - key) <= 1e-10 * max(1.0, abs(key)):
                item[1] += 1
                break
        else:
            out.append([key, 1, pair])
    return out


def _factor_str(v: complex, pair: bool, var: str, fmt: str) -> str:
    inverse = var.endswith("^-1")
    base = var[:-3] if inverse else var
    if pair:
        a1, a0 = -2 * v.real, abs(v) ** 2
        if fmt == "frequency" or fmt == "time constant":
            w0 = np.sqrt(a0)
            zeta = a1 / (2 * w0)
            if fmt == "frequency":
                mid = f" + {fmt_num(2 * zeta)}({base}/{fmt_num(w0)})" if zeta > 0 else (
                    f" - {fmt_num(-2 * zeta)}({base}/{fmt_num(w0)})" if zeta < 0 else "")
                return f"(1{mid} + ({base}/{fmt_num(w0)})^2)"
            t = 1 / w0
            mid = f" + {fmt_num(2 * zeta)}({fmt_num(t)}{base})" if zeta > 0 else (
                f" - {fmt_num(-2 * zeta)}({fmt_num(t)}{base})" if zeta < 0 else "")
            return f"(1{mid} + ({fmt_num(t)}{base})^2)"
        if inverse:
            mid = "" if a1 == 0 else (f" + {fmt_num(a1)}{base}^-1" if a1 > 0 else f" - {fmt_num(-a1)}{base}^-1")
            return f"(1{mid} + {fmt_num(a0)}{base}^-2)"
        mid = "" if a1 == 0 else (f" + {_coef(a1)}{base}" if a1 > 0 else f" - {_coef(-a1)}{base}")
        return f"({base}^2{mid} + {fmt_num(a0)})"
    if v.imag != 0:
        return f"({base}-{fmt_num(v)})"
    x = v.real
    if fmt in ("frequency", "time constant") and x != 0:
        if fmt == "frequency":
            return f"(1{'+' if x < 0 else '-'}{base}/{fmt_num(abs(x))})"
        return f"(1{'+' if x < 0 else '-'}{_coef(1 / abs(x))}{base})"
    if inverse:
        return f"(1{'+' if x < 0 else '-'}{_coef(abs(x))}{base}^-1)"
    if x == 0:
        return base
    return f"({base}{'+' if x < 0 else '-'}{fmt_num(abs(x))})"


def _coef(x: float) -> str:
    s = fmt_num(x)
    return "" if s == "1" else s


def _zpk_strings(z, p, k, var: str, fmt: str):
    inverse = var.endswith("^-1")
    base = var[:-3] if inverse else var

    def side(roots):
        out = []
        for v, mult, pair in _group_roots(roots):
            if inverse and v == 0:
                continue
            s = _factor_str(v, pair, var, fmt)
            out.append(s + (f"^{mult}" if mult > 1 else ""))
        return out

    gain = complex(k) if np.iscomplexobj(k) and np.imag(k) != 0 else float(np.real(k))
    if fmt in ("frequency", "time constant") and not inverse:
        # gain of the factored form with unit constant terms
        for v in z:
            if v != 0:
                gain = gain * (-v)
        for v in p:
            if v != 0:
                gain = gain / (-v)
        gain = float(np.real(gain)) if abs(np.imag(gain)) < 1e-12 * max(abs(gain), 1e-300) else gain
    zf, pf = side(z), side(p)
    if inverse:
        e = len(p) - len(z)
        if e > 0:
            zf.append(f"{base}^-{e}" if e > 1 else f"{base}^-1")
        elif e < 0:
            pf.append(f"{base}^-{-e}" if e < -1 else f"{base}^-1")
    gs = fmt_num(gain)
    if zf:
        num = " ".join(zf) if gs == "1" else ("- " + " ".join(zf) if gs == "-1" else gs + " " + " ".join(zf))
    else:
        num = gs
    return num, (" ".join(pf) if pf else None)


def _zpk_display(g: ZPK) -> list[str]:
    total = g.total_delay()

    def entry(i: int, j: int) -> list[str]:
        if g.k[i, j] == 0:
            return ["0"]
        num, den = _zpk_strings(g.z[i][j], g.p[i][j], g.k[i, j], g.var, g.display_format)
        return _fraction_lines(num, den, _delay_prefix(g, total[i, j]))

    return _io_display(g, entry) + _footer(g, g.is_static())


def _matrix_block(title: str, m: np.ndarray, rows: list[str], cols_: list[str]) -> list[str]:
    lines = [f"  {title} = "]
    if m.shape[0] == 0 or m.shape[1] == 0:
        lines.append(f"     Empty matrix: {m.shape[0]}-by-{m.shape[1]}")
        return lines + [" "]
    cells = [[fmt_num(x, 4) if not (np.iscomplexobj(x) and np.imag(x) != 0) else fmt_num(complex(x), 4)
              for x in r] for r in m]
    lw = max(len(r) for r in rows)
    widths = [max(len(cols_[j]), max(len(cells[i][j]) for i in range(m.shape[0]))) for j in range(m.shape[1])]
    lines.append("   " + " " * lw + "".join("  " + cols_[j].rjust(widths[j]) for j in range(m.shape[1])))
    for i in range(m.shape[0]):
        lines.append("   " + rows[i].ljust(lw) + "".join("  " + cells[i][j].rjust(widths[j])
                                                           for j in range(m.shape[1])))
    return lines + [" "]


def _ss_display(g: SS) -> list[str]:
    xs = [g.state_name[k] or f"x{k + 1}" for k in range(g.nx)]
    us = [g.in_name[k] or f"u{k + 1}" for k in range(g.nu)]
    ys = [g.out_name[k] or f"y{k + 1}" for k in range(g.ny)]
    lines: list[str] = []
    if g.nx:
        lines += _matrix_block("A", g.a, xs, xs)
        lines += _matrix_block("B", g.b, xs, us)
        lines += _matrix_block("C", g.c, ys, xs)
    lines += _matrix_block("D", g.d, ys, us)
    unit = "sampling periods" if g.is_dt else g.time_unit
    if np.any(g.in_delay):
        lines += [f"  Input delays ({unit}): " + "  ".join(fmt_num(d) for d in g.in_delay), " "]
    if np.any(g.out_delay):
        lines += [f"  Output delays ({unit}): " + "  ".join(fmt_num(d) for d in g.out_delay), " "]
    return lines + _footer(g, g.nx == 0 and not g.has_delay())


def _frd_display(g: FRD) -> list[str]:
    unit = "rad/s" if g.freq_unit.lower() == "rad/timeunit" and g.time_unit == "seconds" else g.freq_unit
    head = f"Frequency({unit})"

    def entry(i: int, j: int) -> list[str]:
        vals = g.resp[i, j]
        if np.any(vals.imag):
            resp = [f"{fmt_num(v.real)} {'+' if v.imag >= 0 else '-'} {fmt_num(abs(v.imag))}i" for v in vals]
        else:
            resp = [fmt_num(v.real) for v in vals]
        freqs = [fmt_num(f) for f in g.freq]
        w1 = max([len(head)] + [len(s) for s in freqs])
        w2 = max([len("Response")] + [len(s) for s in resp])
        out = ["  " + head.rjust(w1) + "    " + "Response".rjust(w2),
               "  " + ("-" * len(head)).rjust(w1) + "    " + ("-" * len("Response")).rjust(w2)]
        out += ["  " + f.rjust(w1) + "    " + r.rjust(w2) for f, r in zip(freqs, resp)]
        return out

    return _io_display(g, entry) + _footer(g, False)


def _pid_display(c: PID) -> list[str]:
    kp, ki, kd, tf = c.gains()
    std = isinstance(c, PIDSTD)
    v = "z" if c.is_dt else "s"

    def frac(top: str, bottom: str) -> list[str]:
        w = max(len(top), len(bottom)) + 2
        return [top.center(w), "-" * w, bottom.center(w)]

    def plain(txt: str) -> list[str]:
        return [" " * len(txt), txt, " " * len(txt)]

    def cat(pieces: list, sep: str) -> list[str]:
        rows = ["", "", ""]
        for n, piece in enumerate(pieces):
            for r in range(3):
                rows[r] += ((sep if r == 1 else " " * len(sep)) if n else "") + piece[r]
        return rows

    ints = {"ForwardEuler": ("Ts", f"{v}-1"), "BackwardEuler": (f"Ts*{v}", f"{v}-1"),
            "Trapezoidal": (f"Ts*({v}+1)", f"2*({v}-1)")}
    if std:
        has_p, has_i, has_d = True, bool(np.isfinite(c.ti)), c.td != 0  # type: ignore[attr-defined]
        filt = has_d and bool(np.isfinite(c.n))  # type: ignore[attr-defined]
        tname = "(Td/N)"
    else:
        has_p, has_i, has_d = kp != 0, ki != 0, kd != 0
        filt = has_d and tf != 0
        tname = "Tf"
    if not (has_i or has_d):
        return [f"  Kp = {fmt_num(kp, 3)}", " ", "P-only controller."]
    if c.is_dt:
        integ = frac(*ints[c.iformula])
        dn, dd = ints[c.dformula]
        deriv = frac("1", f"{tname}+{dn}/({dd})") if filt else frac(dd, dn)
    else:
        integ = frac("1", v)
        deriv = frac(v, f"{tname}*{v}+1") if filt else plain(v)
    if std:
        inner = [plain("1")]
        if has_i:
            inner.append(cat([frac("1", "Ti"), integ], " * "))
        if has_d:
            inner.append(cat([plain("Td"), deriv], " * "))
        block = cat([plain("Kp * ("), cat(inner, " + "), plain(")")], "")
        names = [f"Kp = {fmt_num(c.kp, 3)}"]
        if has_i:
            names.append(f"Ti = {fmt_num(c.ti, 3)}")  # type: ignore[attr-defined]
        if has_d:
            names.append(f"Td = {fmt_num(c.td, 3)}")  # type: ignore[attr-defined]
        if filt:
            names.append(f"N = {fmt_num(c.n, 3)}")  # type: ignore[attr-defined]
    else:
        pieces = []
        names = []
        if has_p:
            pieces.append(plain("Kp"))
            names.append(f"Kp = {fmt_num(kp, 3)}")
        if has_i:
            pieces.append(cat([plain("Ki"), integ], " * "))
            names.append(f"Ki = {fmt_num(ki, 3)}")
        if has_d:
            pieces.append(cat([plain("Kd"), deriv], " * "))
            names.append(f"Kd = {fmt_num(kd, 3)}")
        if filt:
            names.append(f"Tf = {fmt_num(tf, 3)}")
        block = cat(pieces, " + ")
    if c.is_dt:
        names.append(f"Ts = {fmt_num(c.ts, 3)}" if c.ts > 0 else "Ts unspecified")
    kind = ("P" if has_p else "") + ("I" if has_i else "") + ("D" if has_d else "") + ("F" if filt else "")
    if not block[0].strip() and not block[2].strip():
        lines = ["  " + block[1].strip()]
    else:
        lines = ["  " + r.rstrip() for r in block]
    lines += ["", "  with " + ", ".join(names), " "]
    if c.is_dt and c.ts > 0:
        lines.append(f"Sample time: {fmt_num(c.ts)} {c.time_unit}")
    form = "standard" if std else "parallel"
    suffix = "-only controller." if len(kind) == 1 else f" controller in {form} form."
    lines.append(f"{c.time_label()} {kind}{suffix}")
    return lines


# -- constructors -----------------------------------------------------------------------------

def split_options(args: list, start: int = 0):
    """Positional arguments up to the first property name, and the name/value pairs after it."""
    pos = list(args[:start])
    k = start
    while k < len(args) and not is_text(args[k]):
        pos.append(args[k])
        k += 1
    rest = args[k:]
    if len(rest) % 2:
        raise MatlabError("Properties must be given as name/value pairs; the value of "
                          f"'{text(rest[-1]) if is_text(rest[-1]) else '?'}' is missing.")
    pairs = []
    for i in range(0, len(rest), 2):
        if not is_text(rest[i]):
            raise MatlabError("Property names must be text, e.g. 'InputDelay', 0.5.")
        pairs.append((text(rest[i]), rest[i + 1]))
    return pos, pairs


def apply_pairs(sys: LTI, pairs: list, skip: tuple = ()) -> LTI:
    for name, value in pairs:
        prop = sys.resolve_prop(name)
        if prop not in skip:
            sys.set_prop(prop, value)
    return sys


def _pair_value(pairs: list, sys_props: list[str], want: str):
    for name, value in pairs:
        low = name.lower()
        if low == want.lower() or (len(low) >= 2 and want.lower().startswith(low)
                                   and sum(p.lower().startswith(low) for p in sys_props) == 1):
            return value
    return None


def _ts_arg(v: Any) -> float:
    ts = as_scalar(v, "sample time Ts")
    if ts < 0 and ts != -1:
        raise MatlabError("The sample time Ts must be 0 (continuous), positive, or -1 (unspecified).")
    return ts


_CT_VARS, _DT_VARS = ("s", "p"), ("z", "q", "z^-1", "q^-1")


def _variable_model(args: list, what: str) -> TF | None:
    """tf('s'), tf('z', Ts): the model of the Laplace or shift variable itself."""
    if not args or not is_text(args[0]):
        return None
    v = text(args[0])
    if v in _CT_VARS:
        if len(args) > 1:
            raise MatlabError(f"{what}('{v}') takes no other argument; use {what}('z', Ts) for the "
                              "discrete-time variable.")
        return TF.siso([1.0, 0.0], [1.0], 0.0, v)
    if v in _DT_VARS:
        ts = _ts_arg(args[1]) if len(args) > 1 else -1.0
        if len(args) > 2:
            raise MatlabError(f"Too many arguments for {what}('{v}', Ts).")
        if ts == 0:
            raise MatlabError(f"{what}('{v}', Ts) needs a nonzero sample time.")
        if v.endswith("^-1"):
            return TF.siso([1.0], [1.0, 0.0], ts, v)
        return TF.siso([1.0, 0.0], [1.0], ts, v)
    raise MatlabError(f"Unknown variable '{v}': use 's' or 'p' (continuous time) or 'z', 'q', 'z^-1', 'q^-1' "
                      "(discrete time).")


def make_tf(num: Any, den: Any, ts: float = 0.0, var: str | None = None) -> TF:
    nums = poly_grid(num, "numerator")
    dens = poly_grid(den, "denominator")
    ny, nu = len(nums), len(nums[0]) if nums else 0
    if len(dens) == 1 and len(dens[0]) == 1 and (ny, nu) != (1, 1):
        dens = [[dens[0][0] for _ in range(nu)] for _ in range(ny)]   # one denominator for all channels
    if len(dens) != ny or (ny and len(dens[0]) != nu):
        raise MatlabError("The numerator and denominator cell arrays must have the same size (one entry per "
                          "output/input pair).")
    var = var or ("z" if ts != 0 else "s")
    if var.endswith("^-1"):
        # coefficients are in ascending powers of z^-1: pad on the right to a common length
        for i in range(ny):
            for j in range(nu):
                n, d = nums[i][j], dens[i][j]
                m = max(n.size, d.size)
                nums[i][j] = np.concatenate([n, np.zeros(m - n.size)])
                dens[i][j] = np.concatenate([d, np.zeros(m - d.size)])
    return TF(nums, dens, ts, var)


def _check_var(var: str | None, ts: float) -> None:
    if var is None:
        return
    allowed = _DT_VARS if ts != 0 else _CT_VARS
    if var not in allowed:
        raise MatlabError(f"Variable '{var}' cannot be used for a "
                          f"{'discrete' if ts != 0 else 'continuous'}-time model; use one of "
                          f"{', '.join(allowed)}.")


@ltifn("tf")
def _tf(interp, args, nargout):
    if not args:
        out = TF([], [], 0.0)
        return out
    v = _variable_model(args, "tf")
    if v is not None:
        return v
    if isinstance(args[0], LTI):
        pos, pairs = split_options(args, 1)
        if len(pos) > 1:
            raise MatlabError("tf(sys) converts a model; it takes no further positional arguments.")
        out = to_tf(args[0])
        out = out.clone()
        return apply_pairs(out, pairs)
    pos, pairs = split_options(args, 1)
    props = TF.PROPS + _COMMON_PROPS
    if len(pos) == 1:
        out = TF.gain(as_matrix(pos[0], "gain"))
        return apply_pairs(out, pairs)
    if len(pos) > 3:
        raise MatlabError("Too many arguments: use tf(num, den), tf(num, den, Ts) or name/value pairs.")
    ts = _ts_arg(pos[2]) if len(pos) == 3 else 0.0
    tsv = _pair_value(pairs, props, "Ts")
    if tsv is not None:
        ts = _ts_arg(tsv)
    varv = _pair_value(pairs, props, "Variable")
    var = text(varv) if varv is not None else None
    _check_var(var, ts)
    out = make_tf(pos[0], pos[1], ts, var)
    return apply_pairs(out, pairs, skip=("Ts", "Variable"))


@ltifn("filt")
def _filt(interp, args, nargout):
    pos, pairs = split_options(args)
    if len(pos) == 1:
        pos = [pos[0], np.ones((1, 1))]
    if len(pos) < 2 or len(pos) > 3:
        raise MatlabError("Use filt(num, den) or filt(num, den, Ts) with coefficients in ascending powers "
                          "of z^-1.")
    ts = _ts_arg(pos[2]) if len(pos) == 3 else -1.0
    if ts == 0:
        raise MatlabError("filt creates discrete-time filters; the sample time must not be 0.")
    out = make_tf(pos[0], pos[1], ts, "z^-1")
    return apply_pairs(out, pairs)


@ltifn("zpk")
def _zpk(interp, args, nargout):
    if not args:
        return ZPK([], [], np.zeros((0, 0)))
    v = _variable_model(args, "zpk")
    if v is not None:
        return to_zpk(v)
    if isinstance(args[0], LTI):
        pos, pairs = split_options(args, 1)
        if len(pos) > 1:
            raise MatlabError("zpk(sys) converts a model; it takes no further positional arguments.")
        return apply_pairs(to_zpk(args[0]).clone(), pairs)
    pos, pairs = split_options(args, 1)
    props = ZPK.PROPS + _COMMON_PROPS
    if len(pos) == 1:
        return apply_pairs(to_zpk(TF.gain(as_matrix(pos[0], "gain"))), pairs)
    if len(pos) < 3 or len(pos) > 4:
        raise MatlabError("Use zpk(z, p, k) or zpk(z, p, k, Ts): zeros, poles and gain.")
    k = as_matrix(pos[2], "gain k")
    ny, nu = k.shape
    ts = _ts_arg(pos[3]) if len(pos) == 4 else 0.0
    tsv = _pair_value(pairs, props, "Ts")
    if tsv is not None:
        ts = _ts_arg(tsv)
    varv = _pair_value(pairs, props, "Variable")
    var = text(varv) if varv is not None else None
    _check_var(var, ts)
    out = ZPK(root_grid(pos[0], ny, nu, "zeros"), root_grid(pos[1], ny, nu, "poles"), k, ts, var)
    return apply_pairs(out, pairs, skip=("Ts", "Variable"))


def make_ss(a, b, c, d, ts: float = 0.0) -> SS:
    a, b, c = (as_matrix(m, f"{n} matrix") for m, n in ((a, "A"), (b, "B"), (c, "C")))
    d = as_matrix(d, "D matrix")
    n = a.shape[0]
    if b.size == 0 and n == 0:
        b = np.zeros((0, d.shape[1] if d.size else 0))
    if c.size == 0 and n == 0:
        c = np.zeros((d.shape[0] if d.size else 0, 0))
    if d.size == 1 and d.flat[0] == 0 and (c.shape[0], b.shape[1]) != (1, 1):
        d = np.zeros((c.shape[0], b.shape[1]))     # D = 0 stands for a zero matrix of the right size
    if d.size == 0 and (c.shape[0] or b.shape[1]):
        d = np.zeros((c.shape[0], b.shape[1]))
    return SS(a, b, c, d, ts)


@ltifn("ss")
def _ss(interp, args, nargout):
    if not args:
        return SS(np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)))
    if isinstance(args[0], LTI):
        flags = [text(a).lower() for a in args[1:] if is_text(a)]
        if len(args) > 1 and (len(flags) != len(args) - 1 or any(f not in ("minimal", "min", "explicit")
                                                                 for f in flags)):
            raise MatlabError("ss(sys) accepts only the options 'minimal' and 'explicit'.")
        out = to_ss(args[0]).clone()
        if "minimal" in flags or "min" in flags:
            a, b, c, d = cn.ss_minreal(*out.abcd)
            res = SS(a, b, c, d, out.ts)
            _carry(res, out)
            return res
        return out
    pos, pairs = split_options(args, 1)
    if len(pos) == 1:
        d = as_matrix(pos[0], "gain")
        out = SS(np.zeros((0, 0)), np.zeros((0, d.shape[1])), np.zeros((d.shape[0], 0)), d)
        return apply_pairs(out, pairs)
    if len(pos) < 4 or len(pos) > 5:
        raise MatlabError("Use ss(A, B, C, D) or ss(A, B, C, D, Ts).")
    ts = _ts_arg(pos[4]) if len(pos) == 5 else 0.0
    out = make_ss(pos[0], pos[1], pos[2], pos[3], ts)
    return apply_pairs(out, pairs)


@ltifn("dss")
def _dss(interp, args, nargout):
    pos, pairs = split_options(args)
    if len(pos) < 5 or len(pos) > 6:
        raise MatlabError("Use dss(A, B, C, D, E) or dss(A, B, C, D, E, Ts).")
    e = as_matrix(pos[4], "E matrix")
    a, b = as_matrix(pos[0], "A matrix"), as_matrix(pos[1], "B matrix")
    if e.shape != a.shape:
        raise MatlabError("E must be square and of the same size as A.")
    if e.shape[0] and np.linalg.matrix_rank(e) < e.shape[0]:
        raise MatlabError("dss: a singular E matrix (a system with algebraic constraints) is not supported.")
    if e.shape[0]:
        a, b = np.linalg.solve(e, a), np.linalg.solve(e, b)
    ts = _ts_arg(pos[5]) if len(pos) == 6 else 0.0
    return apply_pairs(make_ss(a, b, pos[2], pos[3], ts), pairs)


@ltifn("frd")
def _frd(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Use frd(response, frequencies) or frd(sys, frequencies).")
    pos, pairs = split_options(args, 2)
    unit = "rad/TimeUnit"
    uv = _pair_value(pairs, FRD.PROPS + _COMMON_PROPS, "FrequencyUnit")
    if uv is None:
        for name, value in pairs:
            if name.lower() == "units":
                uv = value
    if uv is not None:
        unit = text(uv)
        freq_unit_factor(unit)
    pairs = [(n, v) for n, v in pairs if n.lower() not in ("units",)]
    freq = as_vector(pos[1], "frequency vector").real.astype(float)
    if isinstance(pos[0], LTI):
        src = pos[0]
        if len(pos) > 2:
            raise MatlabError("frd(sys, frequencies) takes no sample time: it is taken from the model.")
        if isinstance(src, FRD):
            w = freq * freq_unit_factor(unit)
            out = FRD(frd_interp(src, w), freq, src.ts, unit)
        else:
            out = FRD(lti_freqresp(src, freq * freq_unit_factor(unit)), freq, src.ts, unit)
        out.copy_meta(src)
        return apply_pairs(out, pairs, skip=("FrequencyUnit",))
    resp = np.asarray(pos[0])
    if not is_num(resp):
        raise MatlabError("The response data must be numeric.")
    if resp.ndim <= 2:
        if resp.size != freq.size or (resp.ndim == 2 and min(resp.shape) != 1 and resp.size):
            if resp.ndim == 2 and freq.size == 1:
                resp = resp.reshape(resp.shape + (1,))
            else:
                raise MatlabError("For a SISO model the response must be a vector with one value per "
                                  "frequency; for MIMO give an ny-by-nu-by-nf array.")
        else:
            resp = resp.reshape(1, 1, -1, order="F")
    ts = _ts_arg(pos[2]) if len(pos) > 2 else 0.0
    out = FRD(resp, freq, ts, unit)
    return apply_pairs(out, pairs, skip=("FrequencyUnit",))


def _pid_args(args: list, name: str, defaults: list[float]):
    pos, pairs = split_options(args)
    if len(pos) > 5:
        raise MatlabError(f"Too many arguments for {name}.")
    vals = list(defaults)
    for k, v in enumerate(pos[:4]):
        vals[k] = as_scalar(v, f"argument {k + 1} of {name}")
    ts = _ts_arg(pos[4]) if len(pos) == 5 else 0.0
    return vals, ts, pairs


@ltifn("pid")
def _pid(interp, args, nargout):
    if args and isinstance(args[0], LTI):
        pos, pairs = split_options(args, 1)
        src = args[0]
        if isinstance(src, PIDSTD):
            kp, ki, kd, tf = src.gains()
            out = PID(kp, ki, kd, tf, src.ts)
            out.iformula, out.dformula = src.iformula, src.dformula
            _carry(out, src)
        elif isinstance(src, PID):
            out = src.clone()
        else:
            like = PID(0, 0, 0, 0, src.ts)
            for nm, val in pairs:
                if nm.lower() in ("iformula", "dformula"):
                    like.set_prop("IFormula" if nm.lower() == "iformula" else "DFormula", val)
            res = tf_to_pid(src, like)
            if res is None:
                raise MatlabError("This model cannot be written as a PID controller in parallel form: a PID "
                                  "has at most one integrator, one filter pole and two zeros.")
            out = res
        return apply_pairs(out, pairs)
    vals, ts, pairs = _pid_args(args, "pid", [1.0, 0.0, 0.0, 0.0])
    out = PID(vals[0], vals[1], vals[2], vals[3], ts)
    apply_pairs(out, pairs)
    return out


@ltifn("pidstd")
def _pidstd(interp, args, nargout):
    if args and isinstance(args[0], LTI):
        pos, pairs = split_options(args, 1)
        src = args[0]
        if isinstance(src, PIDSTD):
            out: PID = src.clone()
        else:
            if not isinstance(src, PID):
                res = tf_to_pid(src, None)
                if res is None:
                    raise MatlabError("This model cannot be written as a PID controller in standard form.")
                src = res
            kp, ki, kd, tf = src.gains()
            out = pid_to_std(kp, ki, kd, tf, src.ts)
            out.iformula, out.dformula = src.iformula, src.dformula
            _carry(out, src)
        return apply_pairs(out, pairs)
    vals, ts, pairs = _pid_args(args, "pidstd", [1.0, np.inf, 0.0, np.inf])
    out = PIDSTD(vals[0], vals[1], vals[2], vals[3], ts)
    apply_pairs(out, pairs)
    return out


# -- data access ------------------------------------------------------------------------------

def need_lti(args: list, name: str, k: int = 0) -> LTI:
    if len(args) <= k:
        raise MatlabError(f"Not enough input arguments for {name}.")
    v = args[k]
    if isinstance(v, LTI):
        return v
    if is_num(v):
        return as_lti(v)
    raise MatlabError(f"{name} expects a control-system model (tf, zpk, ss, frd or pid).")


def _vec_flag(args: list, k: int, name: str) -> bool:
    if len(args) <= k:
        return False
    if is_text(args[k]) and text(args[k]).lower() in ("v", "vector"):
        return True
    raise MatlabError(f"The only option of {name} is 'v', which returns vectors instead of cell arrays for "
                      "a SISO model.")


@ltifn("tfdata")
def _tfdata(interp, args, nargout):
    g = to_tf(need_lti(args, "tfdata"))
    vec = _vec_flag(args, 1, "tfdata")
    if vec and g.is_siso:
        n, d = g.padded(0, 0)
        return row(n), row(d), float(g.ts)
    return g.get_prop("Numerator"), g.get_prop("Denominator"), float(g.ts)


@ltifn("zpkdata")
def _zpkdata(interp, args, nargout):
    g = to_zpk(need_lti(args, "zpkdata"))
    vec = _vec_flag(args, 1, "zpkdata")
    if vec and g.is_siso:
        z, p = g.get_prop("Z").data[0, 0], g.get_prop("P").data[0, 0]
        return z, p, g.k[0, 0], float(g.ts)
    return g.get_prop("Z"), g.get_prop("P"), g.k, float(g.ts)


@ltifn("ssdata")
def _ssdata(interp, args, nargout):
    g = to_ss(need_lti(args, "ssdata"))
    return g.a, g.b, g.c, g.d, float(g.ts)


@ltifn("dssdata")
def _dssdata(interp, args, nargout):
    g = to_ss(need_lti(args, "dssdata"))
    return g.a, g.b, g.c, g.d, np.eye(g.nx), float(g.ts)


@ltifn("frdata")
def _frdata(interp, args, nargout):
    g = need_lti(args, "frdata")
    if not isinstance(g, FRD):
        raise MatlabError("frdata needs a frequency-response (frd) model; create one with frd(sys, w).")
    vec = _vec_flag(args, 1, "frdata")
    resp = g.resp if np.any(g.resp.imag) else g.resp.real
    if vec and g.is_siso:
        return col(resp[0, 0]), col(g.freq), float(g.ts)
    return resp, col(g.freq), float(g.ts)


@ltifn("piddata")
def _piddata(interp, args, nargout):
    g = need_lti(args, "piddata")
    if not isinstance(g, PID):
        c = tf_to_pid(g, None)
        if c is None:
            raise MatlabError("piddata: this model cannot be written as a PID controller in parallel form.")
        g = c
    kp, ki, kd, tf = g.gains()
    return kp, ki, kd, tf, float(g.ts)


@ltifn("pidstddata")
def _pidstddata(interp, args, nargout):
    g = need_lti(args, "pidstddata")
    if not isinstance(g, PIDSTD):
        if not isinstance(g, PID):
            c = tf_to_pid(g, None)
            if c is None:
                raise MatlabError("pidstddata: this model cannot be written as a PID controller.")
            g = c
        kp, ki, kd, tf = g.gains()
        g = pid_to_std(kp, ki, kd, tf, g.ts)
    return g.kp, g.ti, g.td, g.n, float(g.ts)


@ltifn("get", builtin=False)
def _get(interp, args, nargout):
    g = args[0]
    if not isinstance(g, LTI):
        raise MatlabError("get: the first argument must be a model.")
    if len(args) == 1:
        s = Struct.scalar({p: g.get_prop(p) for p in g.prop_names()})
        if nargout:
            return s
        from ..display import display_lines
        interp.out("\n".join(display_lines(s)) + "\n\n")
        return None
    if len(args) > 2:
        raise MatlabError("get(sys, name) returns one property.")
    if isinstance(args[1], Cell):
        return Cell.from_list([g.get_prop(g.resolve_prop(n)) for n in names_from(args[1])])
    return g.get_prop(g.resolve_prop(text(args[1])))


@ltifn("set", builtin=False)
def _set(interp, args, nargout):
    g = args[0]
    if not isinstance(g, LTI):
        raise MatlabError("set: the first argument must be a model.")
    if len(args) == 1:
        names = g.prop_names()
        if nargout:
            return Struct.scalar({p: Cell(np.empty((0, 0), dtype=object)) for p in names})
        interp.out("".join(f"    {p}\n" for p in names))
        return None
    pos, pairs = split_options(args, 1)
    if len(pos) != 1 or not pairs:
        raise MatlabError("Use sys = set(sys, 'Property', value, ...).")
    out = g.clone()
    apply_pairs(out, pairs)
    return out


@ltifn("properties", "fieldnames", builtin=False)
def _props(interp, args, nargout):
    g = args[0]
    names = g.prop_names()
    if nargout:
        return names_cell(names)
    interp.out(f"\nProperties for class {g.kind}:\n\n" + "".join(f"    {p}\n" for p in names) + "\n")
    return None


@ltifn("isprop", builtin=False)
def _isprop(interp, args, nargout):
    g = args[0]
    if not isinstance(g, LTI) or len(args) < 2:
        return bool_scalar(False)
    try:
        g.resolve_prop(text(args[1]))
    except MatlabError:
        return bool_scalar(False)
    return bool_scalar(True)


# -- queries ----------------------------------------------------------------------------------

def lti_order(g: LTI) -> int:
    if isinstance(g, FRD):
        raise MatlabError("The order of frequency-response data is not defined.")
    if isinstance(g, SS):
        return g.nx
    if isinstance(g, PID):
        g = pid_to_tf(g)
    if g.is_siso:
        if isinstance(g, TF):
            return 0 if cn.piszero(g.num[0][0]) else max(g.num[0][0].size, g.den[0][0].size) - 1
        return 0 if g.k[0, 0] == 0 else max(len(g.z[0][0]), len(g.p[0][0]))  # type: ignore[attr-defined]
    return to_ss(g).nx


@ltifn("size", builtin=False)
def _size(interp, args, nargout):
    g = args[0]
    if not isinstance(g, LTI):
        return BUILTINS["size"](interp, args, nargout)
    dims = [g.ny, g.nu]
    if len(args) > 1:
        if is_text(args[1]):
            what = text(args[1]).lower()
            if what == "order":
                return float(lti_order(g))
            if what in ("frequency", "freq") and isinstance(g, FRD):
                return float(g.freq.size)
            raise MatlabError("size(sys, ...) accepts a dimension number, 'order' or (for frd models) "
                              "'frequency'.")
        k = int(as_scalar(args[1], "dimension"))
        if k < 1:
            raise MatlabError("The dimension must be a positive integer.")
        return float(dims[k - 1]) if k <= 2 else 1.0
    if nargout <= 0 and interp is not None and nargout == 0:
        words = {"tf": "Transfer function", "zpk": "Zero-pole-gain model", "ss": "State-space model",
                 "frd": "FRD model", "pid": "PID controller", "pidstd": "PID controller"}[g.kind]
        msg = f"{words} with {g.ny} outputs"
        if isinstance(g, SS):
            msg += f", {g.nu} inputs, and {g.nx} states."
        elif isinstance(g, FRD):
            msg += f", {g.nu} inputs, and {g.freq.size} frequency points."
        else:
            msg += f" and {g.nu} inputs."
        interp.out(msg + "\n")
        return None
    if nargout <= 1:
        return row(np.array(dims, dtype=float))
    out = [float(d) for d in dims] + [1.0] * (nargout - 2)
    return tuple(out[:nargout])


@ltifn("order")
def _order(interp, args, nargout):
    return float(lti_order(need_lti(args, "order")))


@ltifn("isct")
def _isct(interp, args, nargout):
    return bool_scalar(need_lti(args, "isct").ts == 0)


@ltifn("isdt")
def _isdt(interp, args, nargout):
    g = need_lti(args, "isdt")
    return bool_scalar(g.ts != 0 or g.is_static())


@ltifn("issiso")
def _issiso(interp, args, nargout):
    return bool_scalar(need_lti(args, "issiso").is_siso)


@ltifn("isstatic")
def _isstatic(interp, args, nargout):
    return bool_scalar(need_lti(args, "isstatic").is_static())


@ltifn("hasdelay")
def _hasdelay(interp, args, nargout):
    return bool_scalar(need_lti(args, "hasdelay").has_delay())


@ltifn("totaldelay")
def _totaldelay(interp, args, nargout):
    return need_lti(args, "totaldelay").total_delay()


@ltifn("absorbDelay")
def _absorb_delay(interp, args, nargout):
    return absorb_delay(need_lti(args, "absorbDelay"))


def lti_isproper(g: LTI, elementwise: bool = False):
    if isinstance(g, (SS, FRD)):
        ok = np.ones((g.ny, g.nu), dtype=bool)
    else:
        if isinstance(g, PID):
            g = pid_to_tf(g)
        ok = np.ones((g.ny, g.nu), dtype=bool)
        for i in range(g.ny):
            for j in range(g.nu):
                if isinstance(g, TF):
                    ok[i, j] = cn.piszero(g.num[i][j]) or g.num[i][j].size <= g.den[i][j].size
                else:
                    ok[i, j] = g.k[i, j] == 0 or len(g.z[i][j]) <= len(g.p[i][j])  # type: ignore[attr-defined]
    return ok if elementwise else bool(np.all(ok))


@ltifn("isproper")
def _isproper(interp, args, nargout):
    g = need_lti(args, "isproper")
    elem = False
    if len(args) > 1:
        if not (is_text(args[1]) and text(args[1]).lower() in ("elem", "element")):
            raise MatlabError("The only option of isproper is 'elem'.")
        elem = True
    res = lti_isproper(g, elem)
    val = res if elem else bool_scalar(res)
    if nargout > 1:
        return val, g
    return val


def lti_poles(g: LTI) -> np.ndarray:
    """Poles of a model (delays are not counted)."""
    if isinstance(g, FRD):
        raise MatlabError("Poles are not defined for frequency-response data.")
    if isinstance(g, PID):
        g = pid_to_tf(g)
    if isinstance(g, SS):
        p = cn.eigvals(g.a)
        return p if np.iscomplexobj(g.a) else cn.pair_conjugates(p, 1e-12)
    if g.is_siso:
        if isinstance(g, TF):
            return cn.proots(g.den[0][0]) if not cn.piszero(g.num[0][0]) else np.zeros(0)
        return np.asarray(g.p[0][0]) if g.k[0, 0] != 0 else np.zeros(0)  # type: ignore[attr-defined]
    if g.ny == 0 or g.nu == 0:
        return np.zeros(0)
    return lti_poles(to_ss(_no_delay_copy(g)))


def _no_delay_copy(g: LTI) -> LTI:
    if not g.has_delay():
        return g
    out = g.clone()
    out.in_delay, out.out_delay, out.io_delay = np.zeros(g.nu), np.zeros(g.ny), np.zeros((g.ny, g.nu))
    return out


@ltifn("isstable")
def _isstable(interp, args, nargout):
    g = need_lti(args, "isstable")
    return bool_scalar(cn.is_stable(lti_poles(g), g.is_dt))


@ltifn("exp", builtin=False)
def _exp(interp, args, nargout):
    g = args[0]
    if not isinstance(g, LTI):
        return BUILTINS["exp"](interp, args, nargout)
    if isinstance(g, FRD):
        return g.with_resp(np.exp(g.resp))
    t = to_tf(g)
    if t.is_dt:
        raise MatlabError("exp(-tau*s) creates a delay for continuous-time models only; in discrete time "
                          "use z^-n or the delay properties.")
    io = np.zeros((t.ny, t.nu))
    for i in range(t.ny):
        for j in range(t.nu):
            n, d = t.num[i][j], t.den[i][j]
            if cn.piszero(n):
                continue
            ok = d.size == 1 and n.size == 2 and n[1] == 0 and np.isreal(n[0] / d[0]) and np.real(n[0] / d[0]) < 0
            if not ok:
                raise MatlabError("exp of a model is only defined for exp(-tau*s) with tau >= 0, which "
                                  "creates a time delay of tau.")
            io[i, j] = -float(np.real(n[0] / d[0]))
    out = TF.gain(np.ones((t.ny, t.nu)), 0.0)
    out.var = t.var
    out.io_delay = io
    return out


# -- interconnection functions ----------------------------------------------------------------

def _index_arg(v: Any, n: int, what: str) -> np.ndarray:
    idx = as_vector(v, what).real
    if np.any(idx != np.round(idx)) or (idx.size and (idx.min() < 1 or idx.max() > n)):
        raise MatlabError(f"The {what} must be whole numbers between 1 and {n}.")
    return idx.astype(int) - 1


def _sign_arg(v: Any) -> float:
    s = as_scalar(v, "feedback sign")
    if s not in (1.0, -1.0):
        raise MatlabError("The feedback sign must be +1 (positive feedback) or -1 (negative feedback).")
    return s


@ltifn("feedback")
def _feedback(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("feedback needs two models: feedback(sys1, sys2). Use feedback(sys, 1) for a "
                          "unity-feedback loop.")
    g, h = args[0], args[1]
    if not isinstance(g, LTI) and not isinstance(h, LTI):
        g = as_lti(g)
    if len(args) == 2:
        return lti_feedback(g, h)
    if len(args) == 3:
        return lti_feedback(g, h, _sign_arg(args[2]))
    if len(args) in (4, 5):
        gl = as_lti(g)
        fin = _index_arg(args[2], gl.nu, "feedin indices")
        fout = _index_arg(args[3], gl.ny, "feedout indices")
        return lti_feedback(g, h, _sign_arg(args[4]) if len(args) == 5 else -1.0, fin, fout)
    raise MatlabError("Too many arguments for feedback.")


@ltifn("series")
def _series(interp, args, nargout):
    if len(args) == 2:
        return lti_mul(args[1], args[0])
    if len(args) == 4:
        s1, s2 = as_lti(args[0]), as_lti(args[1])
        o1 = _index_arg(args[2], s1.ny, "outputs1 indices")
        i2 = _index_arg(args[3], s2.nu, "inputs2 indices")
        if o1.size != i2.size:
            raise MatlabError("series: outputs1 and inputs2 must have the same length.")
        return lti_mul(s2.sub(np.arange(s2.ny), i2), s1.sub(o1, np.arange(s1.nu)))
    raise MatlabError("Use series(sys1, sys2) or series(sys1, sys2, outputs1, inputs2).")


@ltifn("parallel")
def _parallel(interp, args, nargout):
    if len(args) == 2:
        return lti_add(args[0], args[1])
    if len(args) == 6:
        s1, s2 = as_lti(args[0]), as_lti(args[1])
        i1 = _index_arg(args[2], s1.nu, "inp1 indices")
        i2 = _index_arg(args[3], s2.nu, "inp2 indices")
        o1 = _index_arg(args[4], s1.ny, "out1 indices")
        o2 = _index_arg(args[5], s2.ny, "out2 indices")
        if i1.size != i2.size or o1.size != o2.size:
            raise MatlabError("parallel: the index vectors of the two models must have the same lengths.")
        v1 = [k for k in range(s1.nu) if k not in set(i1.tolist())]
        v2 = [k for k in range(s2.nu) if k not in set(i2.tolist())]
        z1 = [k for k in range(s1.ny) if k not in set(o1.tolist())]
        z2 = [k for k in range(s2.ny) if k not in set(o2.tolist())]
        nin = len(v1) + i1.size + len(v2)
        nout = len(z1) + o1.size + len(z2)
        ei = np.zeros((s1.nu + s2.nu, nin))
        for c_, k in enumerate(v1):
            ei[k, c_] = 1
        for c_, (k1, k2) in enumerate(zip(i1, i2)):
            ei[k1, len(v1) + c_] = 1
            ei[s1.nu + k2, len(v1) + c_] = 1
        for c_, k in enumerate(v2):
            ei[s1.nu + k, len(v1) + i1.size + c_] = 1
        eo = np.zeros((nout, s1.ny + s2.ny))
        for r, k in enumerate(z1):
            eo[r, k] = 1
        for r, (k1, k2) in enumerate(zip(o1, o2)):
            eo[len(z1) + r, k1] = 1
            eo[len(z1) + r, s1.ny + k2] = 1
        for r, k in enumerate(z2):
            eo[len(z1) + o1.size + r, s1.ny + k] = 1
        return lti_mul(lti_mul(eo, lti_append([s1, s2])), ei)
    raise MatlabError("Use parallel(sys1, sys2) or parallel(sys1, sys2, inp1, inp2, out1, out2).")


@ltifn("append")
def _append(interp, args, nargout):
    if not args:
        raise MatlabError("append needs at least one model.")
    if not any(isinstance(a, LTI) for a in args):
        if all(is_num(a) for a in args):
            return cn.blkdiag(*[as_matrix(a) for a in args])
        raise MatlabError("append for control-system models needs at least one model argument.")
    return lti_append(list(args))


@ltifn("blkdiag", builtin=False)
def _blkdiag(interp, args, nargout):
    return lti_append(list(args))


@ltifn("lft")
def _lft(interp, args, nargout):
    if len(args) == 2:
        return lti_lft(args[0], args[1])
    if len(args) == 4:
        return lti_lft(args[0], args[1], int(as_scalar(args[2], "nu")), int(as_scalar(args[3], "ny")))
    raise MatlabError("Use lft(sys1, sys2) or lft(sys1, sys2, nu, ny).")


@ltifn("inv", builtin=False)
def _inv(interp, args, nargout):
    return lti_inv(args[0])


def _binary_fn(fn: Callable) -> Callable:
    def f(interp, args, nargout):
        if len(args) != 2:
            raise MatlabError("Two operands are expected.")
        return fn(args[0], args[1])
    return f


for _name, _fn in (("plus", lti_add), ("minus", lambda a, b: lti_add(a, b, -1.0)), ("mtimes", lti_mul),
                   ("times", lti_times), ("mrdivide", lambda a, b: lti_mul(a, lti_inv(b))),
                   ("mldivide", lambda a, b: lti_mul(lti_inv(a), b)), ("mpower", lti_power),
                   ("rdivide", lambda a, b: lti_times(a, b, True)),
                   ("power", lambda a, b: lti_power(a, b, True))):
    LTI_FUNCS[_name] = _binary_fn(_fn)
LTI_FUNCS["uminus"] = lambda interp, args, nargout: lti_neg(args[0])
LTI_FUNCS["uplus"] = lambda interp, args, nargout: args[0]
LTI_FUNCS["transpose"] = lambda interp, args, nargout: lti_transpose(args[0])
LTI_FUNCS["ctranspose"] = lambda interp, args, nargout: lti_ctranspose(args[0])
LTI_FUNCS["horzcat"] = lambda interp, args, nargout: lti_hcat(list(args))
LTI_FUNCS["vertcat"] = lambda interp, args, nargout: lti_vcat(list(args))


@ltifn("sumblk")
def _sumblk(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("sumblk needs a formula such as 'e = r - y'.")
    formula = text(args[0])
    width = 1
    subs: list[str] = []
    for a in args[1:]:
        if is_text(a):
            subs.append(text(a))
        elif isinstance(a, (Cell, StringArray)):
            raise MatlabError("sumblk: give each replacement name as its own text argument.")
        else:
            width = int(as_scalar(a, "signal size"))
    if width < 1:
        raise MatlabError("sumblk: the signal size must be a positive whole number.")
    if "=" not in formula:
        raise MatlabError("sumblk: the formula must look like 'output = input1 + input2 - input3'.")
    left, right = formula.split("=", 1)
    import re
    toks = re.findall(r"[+-]|%?[A-Za-z_][A-Za-z_0-9.]*|\S", right)
    names, signs = [], []
    sign = 1.0
    expect_name = True
    for t in toks:
        if t in "+-" and len(t) == 1:
            sign = sign * (-1.0 if t == "-" else 1.0) if expect_name else (-1.0 if t == "-" else 1.0)
            expect_name = True
        elif re.fullmatch(r"%?[A-Za-z_][A-Za-z_0-9.]*", t) and expect_name:
            names.append(t)
            signs.append(sign)
            sign = 1.0
            expect_name = False
        else:
            raise MatlabError(f"sumblk: cannot read the formula '{formula}' near '{t}'.")
    out_name = left.strip()
    if not names or expect_name or not re.fullmatch(r"%?[A-Za-z_][A-Za-z_0-9.]*", out_name):
        raise MatlabError(f"sumblk: cannot read the formula '{formula}'.")
    allnames = [out_name] + names
    it = iter(subs)
    for k, nm in enumerate(allnames):
        if nm.startswith("%"):
            try:
                allnames[k] = next(it)
            except StopIteration:
                raise MatlabError("sumblk: not enough names for the % placeholders of the formula.") from None
    out_name, names = allnames[0], allnames[1:]
    d = np.hstack([s * np.eye(width) for s in signs])
    out = TF.gain(d)

    def expand(nm):
        return [nm] if width == 1 else [f"{nm}({k + 1})" for k in range(width)]

    out.out_name = expand(out_name)
    out.in_name = [x for nm in names for x in expand(nm)]
    return out


def _signal_names(v: Any, what: str) -> list[str]:
    if isinstance(v, (Cell, StringArray)) or is_text(v):
        return names_from(v)
    raise MatlabError(f"connect: the {what} must be given as names (text or a cell array of names).")


def _match_names(wanted: list[str], available: list[str]) -> list[list[int]]:
    """For each wanted name the matching channels; 'u' also matches the vector channels u(1), u(2)..."""
    out = []
    for nm in wanted:
        hits = [k for k, x in enumerate(available) if x == nm]
        if not hits:
            hits = [k for k, x in enumerate(available) if x.startswith(nm + "(") and x.endswith(")")]
        out.append(hits)
    return out


@ltifn("connect")
def _connect(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("Use connect(sys1, ..., sysN, inputs, outputs) with named signals, or "
                          "connect(sys, Q, inputs, outputs) with an index-based connection matrix.")
    if isinstance(args[0], LTI) and is_num(args[1]) and len(args) == 4 and is_num(args[2]):
        return _connect_index(args)
    k = 0
    blocks = []
    while k < len(args) and (isinstance(args[k], LTI) or is_num(args[k])):
        if is_num(args[k]):
            raise MatlabError("connect: numeric gains need signal names; wrap them as tf(gain) and set "
                              "InputName and OutputName.")
        blocks.append(args[k])
        k += 1
    rest = args[k:]
    if len(rest) != 2 or not blocks:
        raise MatlabError("connect: after the models give the external input names and the output names.")
    inputs, outputs = _signal_names(rest[0], "inputs"), _signal_names(rest[1], "outputs")
    if any(isinstance(b, FRD) for b in blocks):
        raise MatlabError("connect does not support frequency-response data.")
    blocks = _need_no_delay("connect", *blocks)
    ss_blocks = [to_ss(b) for b in blocks]
    ts = _concat_ts(ss_blocks)
    in_names = [n for b in ss_blocks for n in b.in_name]
    out_names = [n for b in ss_blocks for n in b.out_name]
    if any(not n for n in in_names + out_names):
        raise MatlabError("connect: every input and output of every block needs a name (set InputName and "
                          "OutputName, e.g. G.InputName = 'u'; G.OutputName = 'y').")
    dup = sorted({n for n in out_names if out_names.count(n) > 1})
    if dup:
        raise MatlabError(f"connect: the signal '{dup[0]}' is the output of more than one block.")
    appended = cn.ss_append(*[b.abcd for b in ss_blocks])
    nut, nyt = len(in_names), len(out_names)
    kmat = np.zeros((nut, nyt))
    for i, nm in enumerate(in_names):
        if nm in out_names:
            kmat[i, out_names.index(nm)] = 1.0
    ext_in: list[str] = []
    in_hits = _match_names(inputs, in_names)
    for nm, hits in zip(inputs, in_hits):
        if not hits:
            raise MatlabError(f"connect: no block has an input named '{nm}'.")
        ext_in.extend(sorted(set(in_names[h] for h in hits), key=lambda s: in_names.index(s)))
    ein = np.zeros((nut, len(ext_in)))
    for c_, nm in enumerate(ext_in):
        for i, x in enumerate(in_names):
            if x == nm:
                ein[i, c_] = 1.0
    ext_out: list[tuple] = []
    for nm, hits in zip(outputs, _match_names(outputs, out_names)):
        if hits:
            ext_out.extend(("y", h) for h in hits)
            continue
        hits_in = [c_ for c_, x in enumerate(ext_in) if x == nm or (x.startswith(nm + "(") and x.endswith(")"))]
        if not hits_in:
            raise MatlabError(f"connect: no block has an output named '{nm}'.")
        ext_out.extend(("u", h) for h in hits_in)
    eout = np.zeros((len(ext_out), nyt))
    direct = np.zeros((len(ext_out), len(ext_in)))
    names_out = []
    for r, (kind, h) in enumerate(ext_out):
        if kind == "y":
            eout[r, h] = 1.0
            names_out.append(out_names[h])
        else:
            direct[r, h] = 1.0
            names_out.append(ext_in[h])
    a, b, c, d = cn.ss_interconnect(appended, kmat, ein, eout)
    out = SS(a, b, c, d + direct, ts)
    out.in_name, out.out_name = ext_in, names_out
    out.state_name = [n for blk in ss_blocks for n in blk.state_name]
    idx = cn.sminreal_states(out.a, out.b, out.c)
    if idx.size < out.nx:
        res = SS(out.a[np.ix_(idx, idx)], out.b[idx, :], out.c[:, idx], out.d, ts)
        res.in_name, res.out_name = out.in_name, out.out_name
        res.state_name = [out.state_name[i] for i in idx]
        out = res
    return out


def _connect_index(args: list) -> LTI:
    sys = args[0]
    q = as_matrix(args[1], "connection matrix Q").real
    sys = _need_no_delay("connect", sys)[0]
    s = to_ss(sys)
    inputs = _index_arg(args[2], s.nu, "external input indices")
    outputs = _index_arg(args[3], s.ny, "external output indices")
    kmat = np.zeros((s.nu, s.ny))
    for r in q:
        if r.size == 0 or r[0] == 0:
            continue
        i = int(r[0])
        if i < 1 or i > s.nu or r[0] != i:
            raise MatlabError("connect: the first column of Q holds input numbers of the model.")
        for v in r[1:]:
            if v == 0:
                continue
            j = int(abs(v))
            if j < 1 or j > s.ny or abs(v) != j:
                raise MatlabError("connect: the entries of Q after the first column are output numbers "
                                  "(negative for a subtracted signal).")
            kmat[i - 1, j - 1] += 1.0 if v > 0 else -1.0
    ein = np.zeros((s.nu, inputs.size))
    ein[inputs, np.arange(inputs.size)] = 1.0
    eout = np.zeros((outputs.size, s.ny))
    eout[np.arange(outputs.size), outputs] = 1.0
    out = SS(*cn.ss_interconnect(s.abcd, kmat, ein, eout), s.ts)
    out.in_name = [s.in_name[i] for i in inputs]
    out.out_name = [s.out_name[i] for i in outputs]
    out.state_name = list(s.state_name)
    return back_to(out, sys)
