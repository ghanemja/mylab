"""Unit-testing framework.

Three ways to write tests are supported:

- function-based: a file whose main function returns ``functiontests(localfunctions)``; local
  functions named ``test...`` / ``...test`` are tests, ``setupOnce``, ``teardownOnce``, ``setup`` and
  ``teardown`` are fixtures;
- class-based: ``classdef T < matlab.unittest.TestCase`` with ``methods (Test)`` blocks, setup and
  teardown method blocks, ``properties (TestParameter)`` and ``TestTags``;
- script-based: a script split into tests by ``%%`` section titles.

Tests are collected into arrays of ``matlab.unittest.Test`` objects (``testsuite``,
``matlab.unittest.TestSuite.from...``), run by ``runtests`` / ``run`` / ``matlab.unittest.TestRunner``
and reported as ``matlab.unittest.TestResult`` arrays. Test cases offer the ``verify...``,
``assert...``, ``assume...`` and ``fatalAssert...`` qualifications.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable

import numpy as np

from ..classdef import BUILTIN_BASES, ClassInfo, NativeArray, NativeItem, ObjectArray, UserObject
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char, shape_of, to_str,
    values_equal,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, text

LEVELS = {"none": 0, "terse": 1, "concise": 2, "detailed": 3, "verbose": 4}
_LEVEL_NAMES = ["None", "Terse", "Concise", "Detailed", "Verbose"]
_RULE = "=" * 80
_KIND_ID = {"assert": "MATLAB:unittest:AssertionFailed", "fatal": "MATLAB:unittest:FatalAssertionFailed",
            "assume": "MATLAB:unittest:AssumptionFailed"}
_KIND_LABEL = {"verify": "Verification failed", "assert": "Assertion failed",
               "fatal": "Fatal assertion failed", "assume": "Assumption failed"}


class QualificationStop(MatlabError):
    """Raised by a failed assertion, fatal assertion or assumption: the current test stops."""

    def __init__(self, kind: str, message: str):
        super().__init__(message, _KIND_ID[kind])
        self.kind = kind


class TestState:
    """What the framework knows about one running test (or about a class-level setup)."""

    __test__ = False  # not a pytest class

    def __init__(self, name: str = "", interactive: bool = False):
        self.name = name
        self.failed = False
        self.incomplete = False
        self.records: list[dict] = []
        self.teardowns: list[Callable[[], None]] = []
        self.failure_hooks: list[Any] = []
        self.interactive = interactive
        self.emit: Callable[[str], None] | None = None  # prints a diagnostic block
        self.log_level = 1
        self.reasons: list[str] = []


def _indent(lines: list[str], by: str = "    ") -> list[str]:
    return [by + ln if ln else ln for ln in lines]


def _value_lines(v: Any) -> list[str]:
    from ..display import display_lines
    try:
        lines = display_lines(v)
    except MatlabError:
        lines = []
    if not lines:
        shape = "x".join(str(d) for d in shape_of(v))
        lines = [f"     {shape} empty {class_name(v)}"]
    if len(lines) > 30:
        lines = lines[:30] + ["     ..."]
    return lines


def _section(title: str, body: list[str]) -> list[str]:
    bar = "-" * (len(title) + 1)
    return [bar, title + ":", bar] + body


def _diag_lines(interp, diag: Any) -> list[str]:
    """Text of a test diagnostic: a message, a function to run, or a diagnostic object."""
    if diag is None:
        return []
    if isinstance(diag, Diagnostic):
        return diag.lines(interp)
    if isinstance(diag, FunctionHandle):
        captured: list[str] = []
        old = interp.out
        interp.out = captured.append
        try:
            interp.call_function_value(diag, [], 0)
        finally:
            interp.out = old
        return "".join(captured).rstrip("\n").split("\n") if captured else []
    if isinstance(diag, StringArray):
        return [x or "" for x in diag.values_f()]
    if isinstance(diag, Cell):
        out: list[str] = []
        for item in diag.values_f():
            out += _diag_lines(interp, item)
        return out
    if is_text(diag):
        return text(diag).split("\n")
    return _value_lines(diag)


class Diagnostic(NativeItem):
    """matlab.unittest.diagnostics: StringDiagnostic, FunctionHandleDiagnostic, DisplayDiagnostic."""

    kind = "matlab.unittest.diagnostics.Diagnostic"

    def __init__(self, kind: str, payload: Any):
        self.kind = "matlab.unittest.diagnostics." + kind
        self.payload = payload

    def lines(self, interp) -> list[str]:
        if self.kind.endswith("DisplayDiagnostic"):
            return _value_lines(self.payload)
        return _diag_lines(interp, self.payload)

    def get_field(self, interp, name: str) -> Any:
        if name == "DiagnosticText":
            return char("\n".join(self.lines(interp)))
        if name in ("Value", "Fcn", "DiagnosticString"):
            return self.payload
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")


for _dname in ("StringDiagnostic", "FunctionHandleDiagnostic", "DisplayDiagnostic"):
    def _make_diag(interp, args, nargout, _k=_dname):
        if len(args) != 1:
            raise MatlabError(f"{_k} takes one argument.")
        return Diagnostic(_k, args[0])
    builtin("matlab.unittest.diagnostics." + _dname)(_make_diag)


def _where(interp) -> list[str]:
    frame = interp.frame
    if frame.file:
        return _section("Stack Information", [f"In {frame.file} ({frame.name})"])
    return []


def _report(interp, st: TestState, kind: str, qname: str, framework: list[str], diag: Any) -> None:
    """Record a failed qualification, print its diagnostics, and stop the test unless it is a verify."""
    body: list[str] = []
    user = _diag_lines(interp, diag)
    for hook in st.failure_hooks:
        user += _diag_lines(interp, hook)
    if user:
        body += _section("Test Diagnostic", user)
    body += _section("Framework Diagnostic", framework)
    body += _where(interp)
    if st.interactive:
        interp.out(f"{_KIND_LABEL[kind]}.\n" + "\n".join(_indent(body)) + "\n")
    else:
        head = f"{st.name} was filtered." if kind == "assume" else f"{_KIND_LABEL[kind]} in {st.name}."
        block = f"\n{_RULE}\n{head}\n" + "\n".join(_indent(body)) + f"\n{_RULE}\n"
        if st.emit is not None:
            st.emit(block)
    st.records.append({"kind": kind, "name": qname, "text": "\n".join(body)})
    if kind == "assume":
        st.incomplete = True
        st.reasons.append("Filtered by assumption.")
    else:
        st.failed = True
        st.reasons.append({"verify": "Failed by verification.", "assert": "Failed by assertion.",
                           "fatal": "Failed by fatal assertion."}[kind])
        if kind != "verify":
            st.incomplete = True
    if kind != "verify":
        raise QualificationStop(kind, f"{_KIND_LABEL[kind]}." + (" " + user[0] if user else ""))


# -- comparison ---------------------------------------------------------------------------------------

class Tolerance(NativeItem):
    """AbsoluteTolerance / RelativeTolerance, and their combinations with | and &."""

    kind = "matlab.unittest.constraints.Tolerance"

    def __init__(self, test: Callable[[np.ndarray, np.ndarray], np.ndarray], label: str):
        self.test = test
        self.label = label

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("|", "&") and isinstance(left, Tolerance) and isinstance(right, Tolerance):
            if op == "|":
                return Tolerance(lambda a, e: left.test(a, e) | right.test(a, e),
                                 f"{left.label} | {right.label}")
            return Tolerance(lambda a, e: left.test(a, e) & right.test(a, e), f"{left.label} & {right.label}")
        return NotImplemented

    def display_lines(self) -> list[str]:
        return [f"  {self.label}"]


def _tol_number(v: Any, what: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "fiu" or np.any(np.asarray(v, dtype=float) < 0):
        raise MatlabError(f"{what} must be a nonnegative number.")
    return np.asarray(v, dtype=float)


def _abs_tol(v: Any) -> Tolerance:
    t = _tol_number(v, "AbsTol")
    return Tolerance(lambda a, e: np.abs(a - e) <= t, f"AbsoluteTolerance({float(t.flat[0]):g})")


def _rel_tol(v: Any) -> Tolerance:
    t = _tol_number(v, "RelTol")
    return Tolerance(lambda a, e: np.abs(a - e) <= t * np.abs(e), f"RelativeTolerance({float(t.flat[0]):g})")


@builtin("matlab.unittest.constraints.AbsoluteTolerance")
def _absolute_tolerance(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("AbsoluteTolerance takes one value.")
    return _abs_tol(args[0])


@builtin("matlab.unittest.constraints.RelativeTolerance")
def _relative_tolerance(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("RelativeTolerance takes one value.")
    return _rel_tol(args[0])


def _size_text(v: Any) -> str:
    return "[" + " ".join(str(d) for d in shape_of(v)) + "]"


def compare(actual: Any, expected: Any, tol: Tolerance | None = None, path: str = "",
            ignore_case: bool = False, ignore_space: bool = False, ignore_fields: tuple = ()) -> list[str] | None:
    """Why ``actual`` differs from ``expected`` (lines of text), or None when they are equal.

    Class, size and values must agree; NaN equals NaN; ``tol`` loosens the comparison of numbers.
    """
    here = f" at {path}" if path else ""
    ca, ce = class_name(actual), class_name(expected)
    if ca != ce:
        return [f"--> Classes do not match{here}.", f"      Actual Class:   {ca}", f"      Expected Class: {ce}"]
    textual = (is_char(actual) or isinstance(actual, MString)) and (ignore_case or ignore_space)
    if textual and (is_char(expected) or isinstance(expected, MString)):
        a, e = to_str(actual), to_str(expected)
        if ignore_case:
            a, e = a.lower(), e.lower()
        if ignore_space:
            a, e = "".join(a.split()), "".join(e.split())
        return None if a == e else [f"--> The text values are not equal{here}."]
    if shape_of(actual) != shape_of(expected):
        return [f"--> Sizes do not match{here}.", f"      Actual size:   {_size_text(actual)}",
                f"      Expected size: {_size_text(expected)}"]
    if isinstance(actual, Cell):
        for k, (x, y) in enumerate(zip(actual.data.flatten(order="F"), expected.data.flatten(order="F"))):
            why = compare(x, y, tol, f"{path}{{{k + 1}}}", ignore_case, ignore_space, ignore_fields)
            if why is not None:
                return why
        return None
    if isinstance(actual, Struct):
        fa = [f for f in actual.fields if f not in ignore_fields]
        fe = [f for f in expected.fields if f not in ignore_fields]
        if set(fa) != set(fe):
            extra = sorted(set(fa) - set(fe))
            missing = sorted(set(fe) - set(fa))
            lines = [f"--> The structs have different fields{here}."]
            if extra:
                lines.append("      Only in the actual value:   " + ", ".join(extra))
            if missing:
                lines.append("      Only in the expected value: " + ", ".join(missing))
            return lines
        many = actual.data.size > 1
        for k, (x, y) in enumerate(zip(actual.data.flatten(order="F"), expected.data.flatten(order="F"))):
            for f in fa:
                where = f"{path}({k + 1}).{f}" if many else f"{path}.{f}"
                why = compare(x[f], y[f], tol, where, ignore_case, ignore_space, ignore_fields)
                if why is not None:
                    return why
        return None
    if isinstance(actual, np.ndarray) and actual.dtype.kind in "fiuc":
        a = np.asarray(actual)
        e = np.asarray(expected)
        if a.dtype.kind != "c" and e.dtype.kind != "c":
            a = a.astype(float)
            e = e.astype(float)
        with np.errstate(invalid="ignore", over="ignore"):
            same = (a == e) | (np.isnan(a) & np.isnan(e))
            if tol is not None and not same.all():
                within = np.asarray(tol.test(a, e), dtype=bool) & np.isfinite(a) & np.isfinite(e)
                same = same | within
        if same.all():
            return None
        bad = np.flatnonzero(~same.flatten(order="F"))
        lines = [f"--> The numeric values are not equal{here}" +
                 (f" within {tol.label}." if tol is not None else " (NaN values are treated as equal).")]
        if a.size == 1:
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                err = a.flat[0] - e.flat[0]
                rel = err / e.flat[0] if e.flat[0] != 0 else None
            lines.append(f"      Error: {_num(err)}" + (f"   RelativeError: {_num(rel)}" if rel is not None else ""))
        else:
            lines.append(f"--> {bad.size} of {a.size} elements differ. First differences:")
            lines.append("        Index    Actual    Expected    Error")
            af, ef = a.flatten(order="F"), e.flatten(order="F")
            for i in bad[:8]:
                lines.append(f"        {i + 1:<8} {_num(af[i]):<9} {_num(ef[i]):<11} {_num(af[i] - ef[i])}")
        return lines
    if isinstance(actual, MObject) and not isinstance(actual, (UserObject, ObjectArray)):
        from ..interpreter import current_interpreter
        interp = current_interpreter()  # tables, datetimes, ...: the type's own isequal decides
        if interp is not None and actual.has_method("isequal"):
            res = actual.call_method(interp, "isequal", [actual, expected], 1)
            if res and bool(np.all(np.asarray(res[0]))):
                return None
            return [f"--> The values are not equal{here}."]
    if values_equal(actual, expected, nan_equal=True):
        return None
    kind = {"char": "character arrays", "string": "strings", "logical": "logical values"}.get(ca, "values")
    return [f"--> The {kind} are not equal{here}."]


def _num(x: Any) -> str:
    if isinstance(x, complex) or np.iscomplexobj(x):
        return f"{complex(x):.6g}"
    return f"{float(x):.15g}"


# -- test case objects ----------------------------------------------------------------------------------

TESTCASE_METHODS: dict[str, Callable] = {}


def _method(*names: str):
    def deco(fn: Callable) -> Callable:
        for n in names:
            TESTCASE_METHODS[n] = fn
            BUILTIN_BASES["matlab.unittest.TestCase"].methods[n] = fn
        return fn
    return deco


class FunctionTestCase(NativeItem):
    """The ``testCase`` a function-based test receives (also what forInteractiveUse returns)."""

    kind = "matlab.unittest.FunctionTestCase"
    is_handle = True

    def __init__(self, state: TestState):
        self.state = state
        self.test_data: Any = Struct()

    def isa_class(self, name: str) -> bool:
        return name in (self.kind, "matlab.unittest.TestCase", "handle",
                        "matlab.unittest.qualifications.Verifiable")

    def has_method(self, name: str) -> bool:
        return name in TESTCASE_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return interp.call_builtin(name, TESTCASE_METHODS[name], list(args), nargout)

    def get_field(self, interp, name: str) -> Any:
        if name == "TestData":
            return self.test_data
        if name in TESTCASE_METHODS:
            res = self.call_method(interp, name, [self], 1)
            return res[0] if res else np.zeros((0, 0))
        raise MatlabError(f"Unrecognized property or method '{name}' for a test case.")

    def set_field(self, interp, name: str, value: Any) -> "FunctionTestCase":
        if name != "TestData":
            raise MatlabError(f"Cannot set property '{name}' of a test case; store test data in "
                              "testCase.TestData.")
        self.test_data = value
        return self

    def display_lines(self) -> list[str]:
        from ..display import summary
        return ["  FunctionTestCase with properties:", "", f"    TestData: {summary(self.test_data, False)}"]


def _state(tc: Any) -> TestState:
    if isinstance(tc, FunctionTestCase):
        return tc.state
    if isinstance(tc, UserObject) and ("matlab.unittest.TestCase" in tc.cls.bases
                                       or "matlab.unittest.fixtures.Fixture" in tc.cls.bases):
        st = tc.__dict__.get("_ut")
        if st is None:
            st = tc._ut = TestState(tc.cls.name, interactive=True)  # used outside a test run
        return st
    if isinstance(tc, Fixture):
        return tc.state
    raise MatlabError("The first argument must be the test case object (testCase).")


def _split_options(rest: list, names: tuple) -> tuple[dict, Any]:
    """Name-value options (given in any letter case) and the one remaining argument: the diagnostic."""
    opts: dict[str, Any] = {}
    diag = None
    lowered = {n.lower(): n for n in names}
    k = 0
    while k < len(rest):
        a = rest[k]
        if is_text(a) and text(a).lower() in lowered and k + 1 < len(rest):
            opts[lowered[text(a).lower()]] = rest[k + 1]
            k += 2
            continue
        if diag is not None:
            raise MatlabError("Too many input arguments.", "MATLAB:maxrhs")
        diag = a
        k += 1
    return opts, diag


def _need(args: list, n: int, usage: str) -> None:
    if len(args) < n:
        raise MatlabError(f"Not enough input arguments. Usage: {usage}", "MATLAB:minrhs")


def _actual_block(actual: Any) -> list[str]:
    return ["Actual Value:"] + _value_lines(actual)


def _logical_scalar(v: Any, want: bool) -> bool:
    return isinstance(v, np.ndarray) and v.dtype == bool and v.size == 1 and bool(v.flat[0]) is want


def _count(v: Any) -> int:
    return int(np.prod(shape_of(v)))


def _class_arg(v: Any) -> str:
    if hasattr(v, "kind") and getattr(v, "kind", "") == "meta.class":
        return v.name
    return text(v, "class name")


def _numbers(v: Any, what: str) -> np.ndarray:
    if isinstance(v, UserObject) and v._data is not None:
        v = v._data
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "fiub":
        raise MatlabError(f"{what} must be numeric.")
    return np.asarray(v, dtype=float)


def _run_capturing_warnings(interp, fh: Any, nargout: int) -> tuple[list, list[tuple[str, str]]]:
    issued: list[tuple[str, str]] = []

    def spy(msg: str, identifier: str = "") -> None:
        interp.last_warning, interp.last_warning_id = msg, identifier
        issued.append((identifier, msg))

    outer = interp.__dict__.get("warn")
    interp.warn = spy  # type: ignore[method-assign]
    try:
        res = interp.call_function_value(fh, [], nargout)
    finally:
        if outer is None:
            del interp.__dict__["warn"]
        else:
            interp.warn = outer  # type: ignore[method-assign]
    return res, issued


# Each check takes the arguments after the test case and returns
# (passed, framework diagnostic lines, test diagnostic, values to return).

def _check_equal(interp, args: list, negate: bool = False):
    _need(args, 2, "verifyEqual(testCase, actual, expected)")
    actual, expected = args[0], args[1]
    opts, diag = _split_options(args[2:], ("AbsTol", "RelTol"))
    tol = None
    if "AbsTol" in opts:
        tol = _abs_tol(opts["AbsTol"])
    if "RelTol" in opts:
        rel = _rel_tol(opts["RelTol"])
        tol = rel if tol is None else Tolerance(lambda a, e, t1=tol, t2=rel: t1.test(a, e) | t2.test(a, e),
                                                f"{tol.label} | {rel.label}")
    why = compare(actual, expected, tol)
    if negate:
        lines = ["--> The values are equal."] + _actual_block(actual)
        return why is not None, lines, diag, []
    lines = (why or []) + _actual_block(actual) + ["Expected Value:"] + _value_lines(expected)
    return why is None, lines, diag, []


def _check_true(interp, args: list, want: bool = True):
    _need(args, 1, "verifyTrue(testCase, actual)")
    _, diag = _split_options(args[1:], ())
    word = "true" if want else "false"
    return _logical_scalar(args[0], want), [f"--> The value must be a logical scalar that is {word}."] \
        + _actual_block(args[0]), diag, []


def _check_empty(interp, args: list, want: bool = True):
    _need(args, 1, "verifyEmpty(testCase, actual)")
    _, diag = _split_options(args[1:], ())
    empty = _count(args[0]) == 0
    msg = "--> The value must be empty." if want else "--> The value must not be empty."
    return empty is want, [msg, f"      Actual size: {_size_text(args[0])}"] + _actual_block(args[0]), diag, []


def _check_size(interp, args: list):
    _need(args, 2, "verifySize(testCase, actual, expectedSize)")
    _, diag = _split_options(args[2:], ())
    want = [int(x) for x in _numbers(args[1], "The expected size").flatten()]
    got = list(shape_of(args[0]))
    while len(want) > 2 and want[-1] == 1:
        want.pop()
    return got == want, ["--> The value has an incorrect size.", f"      Actual size:   {_size_text(args[0])}",
                         "      Expected size: [" + " ".join(str(d) for d in want) + "]"], diag, []


def _check_length(interp, args: list, count: bool = False):
    _need(args, 2, "verifyLength(testCase, actual, expectedLength)")
    _, diag = _split_options(args[2:], ())
    want = int(_numbers(args[1], "The expected length").flat[0])
    dims = shape_of(args[0])
    got = _count(args[0]) if count else (0 if 0 in dims else max(dims))
    what = "number of elements" if count else "length"
    return got == want, [f"--> The value has an incorrect {what}.", f"      Actual {what}:   {got}",
                         f"      Expected {what}: {want}"], diag, []


def _check_class(interp, args: list, exact: bool = True):
    _need(args, 2, "verifyClass(testCase, actual, className)")
    _, diag = _split_options(args[2:], ())
    want = _class_arg(args[1])
    actual = args[0]
    got = class_name(actual)
    if exact:
        ok = got == want
    elif hasattr(actual, "isa_class"):
        ok = bool(actual.isa_class(want))
    else:
        ok = bool(interp.call_builtin("isa", interp.builtins["isa"], [actual, char(want)], 1)[0].flat[0])
    msg = "--> The value's class is incorrect." if exact else "--> The value must be an instance of the expected type."
    return ok, [msg, f"      Actual Class:   {got}", f"      Expected {'Class' if exact else 'Type'}:  {want}"], \
        diag, []


def _same_handle(a: Any, b: Any) -> bool:
    if isinstance(a, ObjectArray) and isinstance(b, ObjectArray):
        return a.data.shape == b.data.shape and all(x is y for x, y in zip(a.data.flat, b.data.flat))
    return isinstance(a, MObject) and a.is_handle and a is b


def _check_same_handle(interp, args: list, want: bool = True):
    _need(args, 2, "verifySameHandle(testCase, actual, expectedHandle)")
    _, diag = _split_options(args[2:], ())
    msg = "--> The two values must be the same handle." if want else "--> The two values must be different handles."
    return _same_handle(args[0], args[1]) is want, [msg], diag, []


def _check_substring(interp, args: list):
    _need(args, 2, "verifySubstring(testCase, actual, substring)")
    _, diag = _split_options(args[2:], ())
    if not is_text(args[0]) or not is_text(args[1]):
        return False, ["--> Both the actual value and the substring must be text."] + _actual_block(args[0]), diag, []
    a, sub = text(args[0]), text(args[1])
    return sub in a, ["--> The text does not contain the substring.", f"      Actual text:        {a}",
                      f"      Expected substring: {sub}"], diag, []


def _check_matches(interp, args: list):
    _need(args, 2, "verifyMatches(testCase, actual, expression)")
    _, diag = _split_options(args[2:], ())
    if not is_text(args[0]) or not is_text(args[1]):
        return False, ["--> Both the actual value and the expression must be text."] + _actual_block(args[0]), diag, []
    res = interp.call_builtin("regexp", interp.builtins["regexp"],
                              [char(text(args[0])), char(text(args[1])), char("once"), char("match")], 1)
    found = bool(res) and _count(res[0]) > 0
    return found, ["--> The text does not match the regular expression.", f"      Actual text: {text(args[0])}",
                   f"      Expression:  {text(args[1])}"], diag, []


def _check_relation(op: str, words: str):
    def check(interp, args: list):
        _need(args, 2, "verifyGreaterThan(testCase, actual, bound)")
        _, diag = _split_options(args[2:], ())
        a, b = _numbers(args[0], "The actual value"), _numbers(args[1], "The bound")
        with np.errstate(invalid="ignore"):
            ok = {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
        return bool(np.all(ok)) and a.size > 0, [f"--> The value must be {words} the bound."] + _actual_block(args[0]) \
            + ["Bound:"] + _value_lines(args[1]), diag, []
    return check


def _check_returns_true(interp, args: list):
    _need(args, 1, "verifyReturnsTrue(testCase, functionHandle)")
    _, diag = _split_options(args[1:], ())
    if not isinstance(args[0], FunctionHandle):
        raise MatlabError("verifyReturnsTrue needs a function handle.")
    res = interp.call_function_value(args[0], [], 1)
    value = res[0] if res else np.zeros((0, 0))
    return _logical_scalar(value, True), ["--> The function must return a logical scalar that is true.",
                                          "Returned Value:"] + _value_lines(value), diag, []


def _check_error(interp, args: list, nargout: int = 0):
    _need(args, 2, "verifyError(testCase, functionHandle, identifier)")
    _, diag = _split_options(args[2:], ())
    fh = args[0]
    if not isinstance(fh, FunctionHandle):
        raise MatlabError("verifyError needs a function handle as the code to run, e.g. @() myFunction(1).")
    any_error = hasattr(args[1], "kind") and getattr(args[1], "kind", "") == "meta.class"
    want = _class_arg(args[1])
    try:
        interp.call_function_value(fh, [], nargout)
    except QualificationStop:
        raise
    except MatlabError as e:
        got = e.identifier or ""
        ok = True if any_error else got == want
        return ok, ["--> The function threw the wrong exception.", f"      Actual Exception:   '{got}'",
                    f"      Expected Exception: '{want}'", f"      Message: {e.message}"], diag, []
    return False, ["--> The function did not throw any exception.", f"      Expected Exception: '{want}'"], diag, []


def _check_warning(interp, args: list, nargout: int = 0, free: bool = False):
    _need(args, 1 if free else 2, "verifyWarning(testCase, functionHandle, identifier)")
    _, diag = _split_options(args[(1 if free else 2):], ())
    fh = args[0]
    if not isinstance(fh, FunctionHandle):
        raise MatlabError("The code to run must be a function handle, e.g. @() myFunction(1).")
    res, issued = _run_capturing_warnings(interp, fh, nargout)
    listed = [f"      '{i}'  {m}" for i, m in issued] or ["      (none)"]
    if free:
        return not issued, ["--> The function issued warnings.", "    Warnings issued:"] + listed, diag, res
    want = text(args[1], "warning identifier")
    return any(i == want for i, _ in issued), ["--> The function did not issue the expected warning.",
                                               f"      Expected Warning: '{want}'",
                                               "    Warnings issued:"] + listed, diag, res


def _check_fail(interp, args: list):
    if len(args) > 1:
        raise MatlabError("Too many input arguments.", "MATLAB:maxrhs")
    return False, ["--> The test was failed on purpose."], args[0] if args else None, []


def _check_that(interp, args: list):
    _need(args, 2, "verifyThat(testCase, actual, constraint)")
    _, diag = _split_options(args[2:], ())
    constraint = args[1]
    if not isinstance(constraint, Constraint):
        raise MatlabError("verifyThat needs a constraint, e.g. matlab.unittest.constraints.IsEqualTo(5).")
    ok, lines = constraint.check(interp, args[0])
    return ok, [f"--> {constraint.label} failed."] + lines, diag, []


_CHECKS: dict[str, Callable] = {
    "Equal": _check_equal,
    "NotEqual": lambda interp, args: _check_equal(interp, args, negate=True),
    "True": _check_true,
    "False": lambda interp, args: _check_true(interp, args, want=False),
    "Empty": _check_empty,
    "NotEmpty": lambda interp, args: _check_empty(interp, args, want=False),
    "Size": _check_size,
    "Length": _check_length,
    "NumElements": lambda interp, args: _check_length(interp, args, count=True),
    "Class": _check_class,
    "InstanceOf": lambda interp, args: _check_class(interp, args, exact=False),
    "SameHandle": _check_same_handle,
    "NotSameHandle": lambda interp, args: _check_same_handle(interp, args, want=False),
    "Substring": _check_substring,
    "Matches": _check_matches,
    "GreaterThan": _check_relation(">", "greater than"),
    "GreaterThanOrEqual": _check_relation(">=", "greater than or equal to"),
    "LessThan": _check_relation("<", "less than"),
    "LessThanOrEqual": _check_relation("<=", "less than or equal to"),
    "ReturnsTrue": _check_returns_true,
    "Fail": _check_fail,
    "That": _check_that,
}
_CALL_CHECKS = {"Error": _check_error, "Warning": _check_warning,
                "WarningFree": lambda interp, args, nargout: _check_warning(interp, args, nargout, free=True)}


def _qualification(prefix: str, kind: str, suffix: str) -> Callable:
    name = prefix + suffix
    calls = suffix in _CALL_CHECKS

    def run(interp, args, nargout):
        if not args:
            raise MatlabError(f"{name} needs the test case as its first argument.")
        st = _state(args[0])
        if calls:
            ok, lines, diag, outs = _CALL_CHECKS[suffix](interp, list(args[1:]), nargout)
        else:
            ok, lines, diag, outs = _CHECKS[suffix](interp, list(args[1:]))
        if not ok:
            _report(interp, st, kind, name, [f"{name} failed."] + lines, diag)
        elif st.interactive:
            interp.out({"verify": "Verification", "assert": "Assertion", "fatal": "Fatal assertion",
                        "assume": "Assumption"}[kind] + " passed.\n")
        if not outs:
            return None
        return tuple(outs) if len(outs) > 1 else outs[0]

    return run


for _prefix, _kind in (("verify", "verify"), ("assert", "assert"), ("assume", "assume"),
                       ("fatalAssert", "fatal")):
    for _suffix in list(_CHECKS) + list(_CALL_CHECKS):
        _method(_prefix + _suffix)(_qualification(_prefix, _kind, _suffix))


@_method("addTeardown")
def _add_teardown(interp, args, nargout):
    if len(args) < 2 or not isinstance(args[1], FunctionHandle):
        raise MatlabError("addTeardown needs the test case and a function handle: "
                          "addTeardown(testCase, @cleanup, arg1, ...).")
    st = _state(args[0])
    fh, extra = args[1], list(args[2:])
    st.teardowns.append(lambda: interp.call_function_value(fh, extra, 0))
    return None


@_method("onFailure")
def _on_failure(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("onFailure needs the test case and a diagnostic.")
    _state(args[0]).failure_hooks.append(args[1])
    return None


def _level_of(v: Any) -> int:
    if isinstance(v, VerbosityValue):
        return v.level
    if is_text(v):
        key = text(v).lower()
        if key not in LEVELS:
            raise MatlabError(f"Unknown verbosity '{text(v)}'; use None, Terse, Concise, Detailed or Verbose.")
        return LEVELS[key]
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "fiu":
        level = int(v.flat[0])
        if 0 <= level <= 4:
            return level
    raise MatlabError("A verbosity is a number from 0 to 4 or one of None, Terse, Concise, Detailed, Verbose.")


@_method("log")
def _log(interp, args, nargout):
    if len(args) not in (2, 3):
        raise MatlabError("log needs the test case and a message: log(testCase, 'text') or "
                          "log(testCase, level, 'text').")
    st = _state(args[0])
    level = _level_of(args[1]) if len(args) == 3 else LEVELS["concise"]
    if level == 0:
        raise MatlabError("A message cannot be logged at verbosity None.")
    if level <= st.log_level or st.interactive:
        lines = _diag_lines(interp, args[-1])
        label = f"[{_LEVEL_NAMES[level]}] Diagnostic logged ({time.strftime('%Y-%m-%dT%H:%M:%S')}): "
        interp.out("   " + label + ("\n" + "\n".join(lines) if len(lines) > 1 else (lines[0] if lines else ""))
                   + "\n")
    return None


@_method("applyFixture")
def _apply_fixture(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("applyFixture needs the test case and a fixture, e.g. "
                          "testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).")
    st = _state(args[0])
    fixture = args[1]
    if isinstance(fixture, Fixture):
        fixture.setup(interp)
        st.teardowns.append(lambda: fixture.teardown(interp))
        return fixture
    if isinstance(fixture, UserObject) and "matlab.unittest.fixtures.Fixture" in fixture.cls.bases:
        inner = _state(fixture)
        inner.interactive = False
        inner.name = st.name
        inner.emit = st.emit
        if "setup" in fixture.cls.methods:
            fixture.cls.call(interp, "setup", [fixture], 0)

        def undo() -> None:
            if "teardown" in fixture.cls.methods:
                fixture.cls.call(interp, "teardown", [fixture], 0)
            while inner.teardowns:
                inner.teardowns.pop()()

        st.teardowns.append(undo)
        return fixture
    raise MatlabError("applyFixture needs a fixture object (a matlab.unittest.fixtures.Fixture).")


@_method("createTemporaryFolder")
def _create_temp_folder(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("createTemporaryFolder takes the test case only.")
    fixture = TemporaryFolderFixture({})
    fixture.setup(interp)
    _state(args[0]).teardowns.append(lambda: fixture.teardown(interp))
    return char(fixture.folder)


for _fname in ("addTeardown", "applyFixture", "log", "onFailure"):
    BUILTIN_BASES["matlab.unittest.fixtures.Fixture"].methods[_fname] = TESTCASE_METHODS[_fname]


@builtin("matlab.unittest.TestCase.forInteractiveUse")
def _for_interactive_use(interp, args, nargout):
    if args:
        cls = interp.find_class(_class_arg(args[0]))
        if cls is None or "matlab.unittest.TestCase" not in cls.bases:
            raise MatlabError("forInteractiveUse takes the meta.class of a test class, e.g. ?MyTest.")
        obj = cls.construct(interp, [], 1)[0]
        obj._ut = TestState(cls.name, interactive=True)
        return obj
    return FunctionTestCase(TestState("interactive", interactive=True))


_for_interactive_use.static = True  # type: ignore[attr-defined]
BUILTIN_BASES["matlab.unittest.TestCase"].methods["forInteractiveUse"] = _for_interactive_use


# -- verbosity ------------------------------------------------------------------------------------------

class VerbosityValue(NativeItem):
    kind = "matlab.unittest.Verbosity"

    def __init__(self, level: int):
        self.level = level

    def same_as(self, other: Any) -> bool:
        return isinstance(other, VerbosityValue) and other.level == self.level

    def has_method(self, name: str) -> bool:
        return name in ("double", "char", "string")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "double":
            return [np.array([[float(self.level)]])]
        label = _LEVEL_NAMES[self.level]
        return [char(label) if name == "char" else MString(label)]

    def summary_text(self, in_cell: bool) -> str:
        return _LEVEL_NAMES[self.level]

    def display_lines(self) -> list[str]:
        return ["  Verbosity enumeration", "", f"    {_LEVEL_NAMES[self.level]}"]


for _i, _label in enumerate(_LEVEL_NAMES):
    _v = VerbosityValue(_i)
    builtin(f"matlab.unittest.Verbosity.{_label}", f"matlab.automation.Verbosity.{_label}")(
        lambda interp, args, nargout, _v=_v: _v)


# -- constraints --------------------------------------------------------------------------------------------

class Constraint(NativeItem):
    """A matlab.unittest.constraints object: ``check(actual)`` says whether it is satisfied and why not."""

    kind = "matlab.unittest.constraints.Constraint"

    def __init__(self, label: str, check: Callable[[Any, Any], tuple[bool, list[str]]]):
        self.label = label
        self.check = check
        self.kind = "matlab.unittest.constraints." + label.split("(")[0].lstrip("~ ")

    def isa_class(self, name: str) -> bool:
        return name in (self.kind, "matlab.unittest.constraints.Constraint")

    def unary_op(self, interp, op: str) -> Any:
        if op != "~":
            return NotImplemented
        inner = self

        def check(interp, actual):
            ok, _ = inner.check(interp, actual)
            return not ok, [f"--> The value must not satisfy {inner.label}."] + _actual_block(actual)

        return Constraint(f"~{self.label}", check)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("&", "|") or not (isinstance(left, Constraint) and isinstance(right, Constraint)):
            return NotImplemented

        def check(interp, actual):
            ok1, l1 = left.check(interp, actual)
            ok2, l2 = right.check(interp, actual)
            ok = (ok1 and ok2) if op == "&" else (ok1 or ok2)
            return ok, ([] if ok1 else l1) + ([] if ok2 else l2)

        return Constraint(f"{left.label} {op} {right.label}", check)

    def has_method(self, name: str) -> bool:
        return name == "satisfiedBy"

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if len(args) != 2:
            raise MatlabError("satisfiedBy needs the constraint and a value.")
        return [bool_scalar(self.check(interp, args[1])[0])]

    def display_lines(self) -> list[str]:
        return [f"  {self.label} constraint"]


def _constraint(name: str):
    def deco(make: Callable) -> Callable:
        builtin("matlab.unittest.constraints." + name)(lambda interp, args, nargout: make(interp, list(args)))
        return make
    return deco


def _flag(v: Any, what: str) -> bool:
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "bfiu":
        return bool(v.flat[0])
    raise MatlabError(f"{what} must be true or false.")


def _simple(label: str, test: Callable[[Any], bool], msg: str) -> Constraint:
    return Constraint(label, lambda interp, actual: (bool(test(actual)), [f"--> {msg}"] + _actual_block(actual)))


@_constraint("IsEqualTo")
def _is_equal_to(interp, args: list) -> Constraint:
    if not args:
        raise MatlabError("IsEqualTo needs the expected value.")
    expected = args[0]
    opts, extra = _split_options(args[1:], ("Within", "IgnoringCase", "IgnoringWhitespace", "IgnoringFields",
                                            "Using"))
    if extra is not None:
        raise MatlabError("IsEqualTo: unexpected argument; options are Within, IgnoringCase, "
                          "IgnoringWhitespace and IgnoringFields.")
    if "Using" in opts:
        raise MatlabError("IsEqualTo: the 'Using' option (custom comparators) is not supported.")
    tol = opts.get("Within")
    if tol is not None and not isinstance(tol, Tolerance):
        raise MatlabError("IsEqualTo: 'Within' needs a tolerance, e.g. AbsoluteTolerance(1e-6).")
    ic = _flag(opts["IgnoringCase"], "IgnoringCase") if "IgnoringCase" in opts else False
    iw = _flag(opts["IgnoringWhitespace"], "IgnoringWhitespace") if "IgnoringWhitespace" in opts else False
    fields: tuple = ()
    if "IgnoringFields" in opts:
        f = opts["IgnoringFields"]
        fields = tuple(to_str(x) for x in f.values_f()) if isinstance(f, Cell) else \
            (tuple(x or "" for x in f.values_f()) if isinstance(f, StringArray) else (text(f),))

    def check(interp, actual):
        why = compare(actual, expected, tol, "", ic, iw, fields)
        return why is None, (why or []) + _actual_block(actual) + ["Expected Value:"] + _value_lines(expected)

    return Constraint("IsEqualTo", check)


@_constraint("IsTrue")
def _is_true(interp, args: list) -> Constraint:
    return _simple("IsTrue", lambda a: _logical_scalar(a, True), "The value must be a logical scalar that is true.")


@_constraint("IsFalse")
def _is_false(interp, args: list) -> Constraint:
    return _simple("IsFalse", lambda a: _logical_scalar(a, False),
                   "The value must be a logical scalar that is false.")


@_constraint("IsEmpty")
def _is_empty(interp, args: list) -> Constraint:
    return _simple("IsEmpty", lambda a: _count(a) == 0, "The value must be empty.")


@_constraint("IsScalar")
def _is_scalar(interp, args: list) -> Constraint:
    return _simple("IsScalar", lambda a: _count(a) == 1, "The value must be a scalar.")


@_constraint("IsAnything")
def _is_anything(interp, args: list) -> Constraint:
    return _simple("IsAnything", lambda a: True, "Any value is accepted.")


@_constraint("IsReal")
def _is_real(interp, args: list) -> Constraint:
    return _simple("IsReal", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "fiubU",
                   "The value must be real.")


@_constraint("IsFinite")
def _is_finite(interp, args: list) -> Constraint:
    return _simple("IsFinite", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "fiubc"
                   and bool(np.all(np.isfinite(a))), "Every element must be finite.")


@_constraint("HasNaN")
def _has_nan(interp, args: list) -> Constraint:
    return _simple("HasNaN", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "fc"
                   and bool(np.any(np.isnan(a))), "The value must contain a NaN.")


@_constraint("HasInf")
def _has_inf(interp, args: list) -> Constraint:
    return _simple("HasInf", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "fc"
                   and bool(np.any(np.isinf(a))), "The value must contain an Inf.")


@_constraint("IsSparse")
def _is_sparse(interp, args: list) -> Constraint:
    return _simple("IsSparse", lambda a: bool(getattr(a, "is_sparse", False)), "The value must be sparse.")


def _relation(name: str, op: str, words: str) -> None:
    @_constraint(name)
    def make(interp, args: list) -> Constraint:
        if len(args) != 1:
            raise MatlabError(f"{name} needs the bound to compare with.")
        bound = _numbers(args[0], "The bound")

        def check(interp, actual):
            try:
                a = _numbers(actual, "The actual value")
            except MatlabError:
                return False, ["--> The value must be numeric."] + _actual_block(actual)
            with np.errstate(invalid="ignore"):
                ok = {">": a > bound, ">=": a >= bound, "<": a < bound, "<=": a <= bound}[op]
            return bool(np.all(ok)) and a.size > 0, [f"--> The value must be {words} the bound."] \
                + _actual_block(actual) + ["Bound:"] + _value_lines(args[0])

        return Constraint(name, check)


_relation("IsGreaterThan", ">", "greater than")
_relation("IsGreaterThanOrEqualTo", ">=", "greater than or equal to")
_relation("IsLessThan", "<", "less than")
_relation("IsLessThanOrEqualTo", "<=", "less than or equal to")


@_constraint("HasSize")
def _has_size(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("HasSize needs the expected size, e.g. HasSize([2 3]).")
    want = [int(x) for x in _numbers(args[0], "The expected size").flatten()]
    return Constraint("HasSize", lambda interp, a: (
        list(shape_of(a)) == want, ["--> The value has an incorrect size.", f"      Actual size:   {_size_text(a)}",
                                    "      Expected size: [" + " ".join(str(d) for d in want) + "]"]))


@_constraint("HasLength")
def _has_length(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("HasLength needs the expected length.")
    want = int(_numbers(args[0], "The expected length").flat[0])
    return Constraint("HasLength", lambda interp, a: (
        (0 if 0 in shape_of(a) else max(shape_of(a))) == want,
        ["--> The value has an incorrect length.", f"      Actual size: {_size_text(a)}",
         f"      Expected length: {want}"]))


@_constraint("HasElementCount")
def _has_count(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("HasElementCount needs the expected number of elements.")
    want = int(_numbers(args[0], "The expected count").flat[0])
    return Constraint("HasElementCount", lambda interp, a: (
        _count(a) == want, ["--> The value has an incorrect number of elements.",
                            f"      Actual count:   {_count(a)}", f"      Expected count: {want}"]))


@_constraint("IsOfClass")
def _is_of_class(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("IsOfClass needs a class name.")
    want = _class_arg(args[0])
    return Constraint("IsOfClass", lambda interp, a: (
        class_name(a) == want, ["--> The value's class is incorrect.", f"      Actual Class:   {class_name(a)}",
                                f"      Expected Class: {want}"]))


@_constraint("IsInstanceOf")
def _is_instance_of(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("IsInstanceOf needs a class name.")
    want = _class_arg(args[0])

    def check(interp, a):
        ok = bool(interp.call_builtin("isa", interp.builtins["isa"], [a, char(want)], 1)[0].flat[0])
        return ok, ["--> The value must be an instance of the expected type.",
                    f"      Actual Class:  {class_name(a)}", f"      Expected Type: {want}"]

    return Constraint("IsInstanceOf", check)


def _text_constraint(name: str, test: Callable[[str, str], bool], words: str) -> None:
    @_constraint(name)
    def make(interp, args: list) -> Constraint:
        if not args or not is_text(args[0]):
            raise MatlabError(f"{name} needs the text to look for.")
        want = text(args[0])
        opts, extra = _split_options(args[1:], ("IgnoringCase", "IgnoringWhitespace", "WithCount"))
        if extra is not None or "WithCount" in opts:
            raise MatlabError(f"{name}: only the IgnoringCase and IgnoringWhitespace options are supported.")
        ic = _flag(opts["IgnoringCase"], "IgnoringCase") if "IgnoringCase" in opts else False
        iw = _flag(opts["IgnoringWhitespace"], "IgnoringWhitespace") if "IgnoringWhitespace" in opts else False

        def norm(s: str) -> str:
            s = s.lower() if ic else s
            return "".join(s.split()) if iw else s

        def check(interp, actual):
            if not is_text(actual):
                return False, ["--> The value must be text."] + _actual_block(actual)
            return test(norm(text(actual)), norm(want)), [f"--> The text must {words}.",
                                                          f"      Actual text:   {text(actual)}",
                                                          f"      Expected text: {want}"]

        return Constraint(name, check)


_text_constraint("ContainsSubstring", lambda a, w: w in a, "contain the substring")
_text_constraint("StartsWithSubstring", lambda a, w: a.startswith(w), "start with the substring")
_text_constraint("EndsWithSubstring", lambda a, w: a.endswith(w), "end with the substring")
_text_constraint("IsSubstringOf", lambda a, w: a in w, "be a substring of the expected text")


@_constraint("Matches")
def _matches(interp, args: list) -> Constraint:
    if len(args) != 1 or not is_text(args[0]):
        raise MatlabError("Matches needs a regular expression.")
    expr = text(args[0])

    def check(interp, actual):
        ok, lines, _, _ = _check_matches(interp, [actual, char(expr)])
        return ok, lines

    return Constraint("Matches", check)


@_constraint("IsSameHandleAs")
def _is_same_handle_as(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("IsSameHandleAs needs the expected handle.")
    return Constraint("IsSameHandleAs", lambda interp, a: (_same_handle(a, args[0]),
                                                            ["--> The two values must be the same handle."]))


@_constraint("HasField")
def _has_field(interp, args: list) -> Constraint:
    if len(args) != 1 or not is_text(args[0]):
        raise MatlabError("HasField needs a field name.")
    name = text(args[0])
    return Constraint("HasField", lambda interp, a: (
        isinstance(a, Struct) and name in a.fields, [f"--> The value must be a struct with the field '{name}'."]
        + _actual_block(a)))


def _members(v: Any) -> list:
    if isinstance(v, Cell):
        return v.values_f()
    if isinstance(v, StringArray):
        return [MString(x or "") for x in v.values_f()]
    if isinstance(v, np.ndarray):
        return [v.flatten(order="F")[k:k + 1].reshape(1, 1) for k in range(v.size)]
    return [v]


def _set_constraint(name: str, mode: str) -> None:
    @_constraint(name)
    def make(interp, args: list) -> Constraint:
        if len(args) != 1:
            raise MatlabError(f"{name} needs the set to compare with.")
        other = _members(args[0])

        def inside(x: Any, pool: list) -> bool:
            return any(values_equal(x, y) for y in pool)

        def check(interp, actual):
            mine = _members(actual)
            sub = all(inside(x, other) for x in mine)
            sup = all(inside(y, mine) for y in other)
            ok = {"subset": sub, "superset": sup, "same": sub and sup}[mode]
            return ok, [f"--> The value must be a {mode} set of the expected set."
                        if mode != "same" else "--> The value must hold the same elements as the expected set."] \
                + _actual_block(actual)

        return Constraint(name, check)


_set_constraint("IsSubsetOf", "subset")
_set_constraint("IsSupersetOf", "superset")
_set_constraint("IsSameSetAs", "same")


@_constraint("Throws")
def _throws(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("Throws needs the expected error identifier.")

    def check(interp, actual):
        ok, lines, _, _ = _check_error(interp, [actual, args[0]])
        return ok, lines

    return Constraint("Throws", check)


@_constraint("IssuesNoWarnings")
def _issues_no_warnings(interp, args: list) -> Constraint:
    def check(interp, actual):
        ok, lines, _, _ = _check_warning(interp, [actual], 0, free=True)
        return ok, lines

    return Constraint("IssuesNoWarnings", check)


@_constraint("IssuesWarnings")
def _issues_warnings(interp, args: list) -> Constraint:
    if len(args) != 1:
        raise MatlabError("IssuesWarnings needs a cell array of warning identifiers.")
    wanted = [to_str(x) for x in args[0].values_f()] if isinstance(args[0], Cell) else [text(args[0])]

    def check(interp, actual):
        if not isinstance(actual, FunctionHandle):
            return False, ["--> The value must be a function handle."]
        _, issued = _run_capturing_warnings(interp, actual, 0)
        got = [i for i, _ in issued]
        return all(w in got for w in wanted), ["--> The function did not issue every expected warning.",
                                               "      Expected: " + ", ".join(wanted),
                                               "      Issued:   " + (", ".join(got) or "(none)")]

    return Constraint("IssuesWarnings", check)


def _path_constraint(name: str, test: Callable[[str], bool], words: str) -> None:
    @_constraint(name)
    def make(interp, args: list) -> Constraint:
        return Constraint(name, lambda interp, a: (is_text(a) and test(text(a)),
                                                    [f"--> The value must be the name of an existing {words}."]
                                                    + _actual_block(a)))


_path_constraint("IsFile", os.path.isfile, "file")
_path_constraint("IsFolder", os.path.isdir, "folder")


# -- fixtures ------------------------------------------------------------------------------------------------

class Fixture(NativeItem):
    kind = "matlab.unittest.fixtures.Fixture"
    is_handle = True
    FIELDS: tuple = ()

    def __init__(self, opts: dict):
        self.opts = opts
        self.state = TestState("fixture")
        self.undo: list[Callable[[], None]] = []

    def isa_class(self, name: str) -> bool:
        return name in (self.kind, "matlab.unittest.fixtures.Fixture", "handle")

    def setup(self, interp) -> None:
        raise NotImplementedError

    def teardown(self, interp) -> None:
        while self.undo:
            self.undo.pop()()
        while self.state.teardowns:
            self.state.teardowns.pop()()

    def has_method(self, name: str) -> bool:
        return name in ("setup", "teardown", "addTeardown")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "setup":
            self.setup(interp)
        elif name == "teardown":
            self.teardown(interp)
        else:
            _add_teardown(interp, args, 0)
        return []

    def display_lines(self) -> list[str]:
        from ..display import summary
        short = self.kind.rsplit(".", 1)[-1]
        if not self.FIELDS:
            return [f"  {short} with no properties."]
        w = max(len(n) for n in self.FIELDS)
        return [f"  {short} with properties:", ""] + \
            [f"    {n:>{w}}: {summary(self.get_field(None, n), False)}" for n in self.FIELDS]


def _fixture_options(args: list, names: tuple, what: str) -> dict:
    opts, extra = _split_options(args, names)
    if extra is not None:
        raise MatlabError(f"{what}: unexpected argument. Options: " + (", ".join(names) or "(none)") + ".")
    return opts


class TemporaryFolderFixture(Fixture):
    kind = "matlab.unittest.fixtures.TemporaryFolderFixture"
    FIELDS = ("Folder", "PreserveOnFailure", "Suffix")
    folder = ""

    def setup(self, interp) -> None:
        import shutil
        import tempfile
        suffix = text(self.opts["WithSuffix"]) if "WithSuffix" in self.opts else ""
        self.folder = os.path.realpath(tempfile.mkdtemp(prefix="tp", suffix=suffix))
        keep = _flag(self.opts["PreservingOnFailure"], "PreservingOnFailure") \
            if "PreservingOnFailure" in self.opts else False
        folder = self.folder
        if not keep:
            self.undo.append(lambda: shutil.rmtree(folder, ignore_errors=True))

    def get_field(self, interp, name: str) -> Any:
        if name == "Folder":
            return char(self.folder)
        if name == "PreserveOnFailure":
            return bool_scalar("PreservingOnFailure" in self.opts and bool(
                np.asarray(self.opts["PreservingOnFailure"]).flat[0]))
        if name == "Suffix":
            return char(text(self.opts["WithSuffix"]) if "WithSuffix" in self.opts else "")
        raise MatlabError(f"Unrecognized property '{name}' for class 'TemporaryFolderFixture'.")


class WorkingFolderFixture(TemporaryFolderFixture):
    kind = "matlab.unittest.fixtures.WorkingFolderFixture"

    def setup(self, interp) -> None:
        super().setup(interp)
        old = os.getcwd()
        os.chdir(self.folder)
        self.undo.append(lambda: os.chdir(old))  # runs before the folder is removed


class CurrentFolderFixture(Fixture):
    kind = "matlab.unittest.fixtures.CurrentFolderFixture"
    FIELDS = ("Folder",)

    def setup(self, interp) -> None:
        folder = os.path.abspath(text(self.opts["Folder"], "folder"))
        if not os.path.isdir(folder):
            raise MatlabError(f"CurrentFolderFixture: the folder '{folder}' does not exist.")
        old = os.getcwd()
        os.chdir(folder)
        self.undo.append(lambda: os.chdir(old))

    def get_field(self, interp, name: str) -> Any:
        if name == "Folder":
            return char(os.path.abspath(text(self.opts["Folder"])))
        raise MatlabError(f"Unrecognized property '{name}' for class 'CurrentFolderFixture'.")


class PathFixture(Fixture):
    kind = "matlab.unittest.fixtures.PathFixture"
    FIELDS = ("Folders", "IncludeSubfolders", "Position")

    def folders(self) -> list[str]:
        v = self.opts["Folders"]
        names = [to_str(x) for x in v.values_f()] if isinstance(v, Cell) else \
            ([x or "" for x in v.values_f()] if isinstance(v, StringArray) else [text(v, "folder")])
        out = []
        deep = _flag(self.opts["IncludeSubfolders"], "IncludeSubfolders") \
            if "IncludeSubfolders" in self.opts else False
        for n in names:
            folder = os.path.abspath(n)
            if not os.path.isdir(folder):
                raise MatlabError(f"PathFixture: the folder '{folder}' does not exist.")
            out.append(folder)
            if deep:
                for root, dirs, _ in os.walk(folder):
                    dirs[:] = sorted(d for d in dirs if d[:1] not in "+@." and d != "private")
                    out += [os.path.join(root, d) for d in dirs]
        return out

    def setup(self, interp) -> None:
        saved = list(interp.path)
        position = text(self.opts["Position"]).lower() if "Position" in self.opts else "begin"
        if position not in ("begin", "end"):
            raise MatlabError("PathFixture: Position must be 'begin' or 'end'.")
        new = [f for f in self.folders() if f not in interp.path]
        interp.path[:] = new + interp.path if position == "begin" else interp.path + new

        def restore() -> None:
            interp.path[:] = saved

        self.undo.append(restore)

    def get_field(self, interp, name: str) -> Any:
        if name in ("Folders", "Folder"):
            return Cell.from_list([char(f) for f in self.folders()])
        if name == "IncludeSubfolders":
            return bool_scalar("IncludeSubfolders" in self.opts and bool(
                np.asarray(self.opts["IncludeSubfolders"]).flat[0]))
        if name == "Position":
            return char(text(self.opts["Position"]) if "Position" in self.opts else "begin")
        raise MatlabError(f"Unrecognized property '{name}' for class 'PathFixture'.")


class SuppressedWarningsFixture(Fixture):
    kind = "matlab.unittest.fixtures.SuppressedWarningsFixture"
    FIELDS = ("Warnings",)

    def ids(self) -> list[str]:
        v = self.opts["Warnings"]
        return [to_str(x) for x in v.values_f()] if isinstance(v, Cell) else \
            ([x or "" for x in v.values_f()] if isinstance(v, StringArray) else [text(v, "warning identifier")])

    def setup(self, interp) -> None:
        saved = set(interp.disabled_warnings)
        interp.disabled_warnings |= set(self.ids())

        def restore() -> None:
            interp.disabled_warnings.clear()
            interp.disabled_warnings.update(saved)

        self.undo.append(restore)

    def get_field(self, interp, name: str) -> Any:
        if name == "Warnings":
            return Cell.from_list([char(i) for i in self.ids()])
        raise MatlabError(f"Unrecognized property '{name}' for class 'SuppressedWarningsFixture'.")


class EnvironmentVariableFixture(Fixture):
    kind = "matlab.unittest.fixtures.EnvironmentVariableFixture"
    FIELDS = ("Name", "Value")

    def setup(self, interp) -> None:
        name, value = text(self.opts["Name"], "name"), text(self.opts["Value"], "value")
        old = os.environ.get(name)
        os.environ[name] = value

        def restore() -> None:
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

        self.undo.append(restore)

    def get_field(self, interp, name: str) -> Any:
        if name in ("Name", "Value"):
            return char(text(self.opts[name]))
        raise MatlabError(f"Unrecognized property '{name}' for class 'EnvironmentVariableFixture'.")


@builtin("matlab.unittest.fixtures.TemporaryFolderFixture")
def _temp_folder_fixture(interp, args, nargout):
    return TemporaryFolderFixture(_fixture_options(list(args), ("WithSuffix", "PreservingOnFailure"),
                                                   "TemporaryFolderFixture"))


@builtin("matlab.unittest.fixtures.WorkingFolderFixture")
def _working_folder_fixture(interp, args, nargout):
    return WorkingFolderFixture(_fixture_options(list(args), ("WithSuffix", "PreservingOnFailure"),
                                                 "WorkingFolderFixture"))


@builtin("matlab.unittest.fixtures.CurrentFolderFixture")
def _current_folder_fixture(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("CurrentFolderFixture needs the folder to change to.")
    return CurrentFolderFixture({"Folder": args[0]})


@builtin("matlab.unittest.fixtures.PathFixture")
def _path_fixture(interp, args, nargout):
    if not args:
        raise MatlabError("PathFixture needs the folder to add to the path.")
    opts = _fixture_options(list(args[1:]), ("IncludeSubfolders", "Position", "IncludingSubfolders"),
                            "PathFixture")
    if "IncludingSubfolders" in opts:
        opts["IncludeSubfolders"] = opts.pop("IncludingSubfolders")
    opts["Folders"] = args[0]
    return PathFixture(opts)


@builtin("matlab.unittest.fixtures.SuppressedWarningsFixture")
def _suppressed_warnings_fixture(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("SuppressedWarningsFixture needs the warning identifiers to turn off.")
    return SuppressedWarningsFixture({"Warnings": args[0]})


@builtin("matlab.unittest.fixtures.EnvironmentVariableFixture")
def _env_fixture(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("EnvironmentVariableFixture needs a variable name and a value.")
    return EnvironmentVariableFixture({"Name": args[0], "Value": args[1]})


# -- tests, suites and results -----------------------------------------------------------------------------

class Group:
    """Tests that share fixtures: one test file or one test class."""

    def __init__(self, kind: str, name: str, folder: str = ""):
        self.kind = kind  # 'function', 'class' or 'script'
        self.name = name
        self.folder = folder
        self.handles: dict[str, FunctionHandle] = {}  # function-based: every local function
        self.cls: ClassInfo | None = None
        self.file = ""
        self.shared = ""  # script-based: code that runs before each test
        self.functions: dict = {}


class TestElement(NativeItem):
    """One test of a suite (class matlab.unittest.Test)."""

    __test__ = False
    kind = "matlab.unittest.Test"
    FIELDS = ("Name", "ProcedureName", "TestClass", "BaseFolder", "Parameterization", "SharedTestFixtures",
              "Tags")

    def __init__(self, group: Group, proc: str, name: str, params: list | None = None, tags: tuple = (),
                 code: str = ""):
        self.group = group
        self.proc = proc
        self.name = name
        self.params = params or []  # (property, value name, value)
        self.tags = tuple(tags)
        self.code = code

    def get_field(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name == "ProcedureName":
            return char(self.proc)
        if name == "TestClass":
            return MString(self.group.name) if self.group.kind == "class" else \
                StringArray(np.empty((0, 0), dtype=object))
        if name == "BaseFolder":
            return char(self.group.folder)
        if name == "Parameterization":
            data = np.empty((1, len(self.params)), dtype=object)
            for k, (prop, pname, value) in enumerate(self.params):
                data[0, k] = {"Property": char(prop), "Name": char(pname), "Value": value}
            return Struct(data, ["Property", "Name", "Value"])
        if name == "SharedTestFixtures":
            return np.zeros((0, 0))
        if name == "Tags":
            return Cell.from_list([char(t) for t in self.tags])
        if self.array_method(name):
            return self.call_array_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized property '{name}' for class 'matlab.unittest.Test'.")

    def array_props(self) -> list[str]:
        return list(self.FIELDS)

    def has_method(self, name: str) -> bool:
        return self.array_method(name)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return self.call_array_method(interp, name, args, nargout)

    def array_method(self, name: str) -> bool:
        return name in ("run", "selectIf", "sortByFixtures")

    def call_array_method(self, interp, name: str, args: list, nargout: int) -> list:
        tests = _tests_of(args[0])
        if name == "run":
            if len(args) != 1:
                raise MatlabError("run(suite) takes the suite only; use a TestRunner for options.")
            return [_results(run_tests(interp, tests, RunOptions()))]
        if name == "sortByFixtures":
            return [args[0]]
        return [_suite(select_tests(interp, tests, list(args[1:])))]

    def display_lines(self) -> list[str]:
        from ..display import summary
        w = max(len(n) for n in self.FIELDS)
        return ["  Test with properties:", ""] + \
            [f"    {n:>{w}}: {summary(self.get_field(None, n), False)}" for n in self.FIELDS]


class TestResult(NativeItem):
    __test__ = False
    kind = "matlab.unittest.TestResult"
    FIELDS = ("Name", "Passed", "Failed", "Incomplete", "Duration", "Details")

    def __init__(self, name: str, failed: bool, incomplete: bool, duration: float, records: list,
                 reasons: list):
        self.name = name
        self.failed = failed
        self.incomplete = incomplete
        self.duration = duration
        self.records = records
        self.reasons = reasons

    @property
    def passed(self) -> bool:
        return not self.failed and not self.incomplete

    def get_field(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name == "Passed":
            return bool_scalar(self.passed)
        if name == "Failed":
            return bool_scalar(self.failed)
        if name == "Incomplete":
            return bool_scalar(self.incomplete)
        if name == "Duration":
            return np.array([[self.duration]])
        if name == "Details":
            data = np.empty((1, len(self.records)), dtype=object)
            for k, r in enumerate(self.records):
                data[0, k] = {"Event": char(r["kind"]), "Report": char(r["text"])}
            return Struct.scalar({"DiagnosticRecord": Struct(data, ["Event", "Report"])})
        if self.array_method(name):
            return self.call_array_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized property '{name}' for class 'matlab.unittest.TestResult'.")

    def array_props(self) -> list[str]:
        return list(self.FIELDS)

    def has_method(self, name: str) -> bool:
        return self.array_method(name)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return self.call_array_method(interp, name, args, nargout)

    def array_method(self, name: str) -> bool:
        return name in ("table", "assertSuccess")

    def call_array_method(self, interp, name: str, args: list, nargout: int) -> list:
        results = _results_of(args[0])
        if name == "assertSuccess":
            bad = [r.name for r in results if not r.passed]
            if bad:
                raise MatlabError(f"{len(bad)} of {len(results)} tests did not pass: " + ", ".join(bad) + ".",
                                  "MATLAB:unittest:TestResult:assertSuccess")
            return []
        n = len(results)

        def column(values: list) -> np.ndarray:
            return np.array(values).reshape(n, 1)

        names = np.empty((n, 1), dtype=object)
        details = np.empty((n, 1), dtype=object)
        for k, r in enumerate(results):
            names[k, 0] = char(r.name)
            details[k, 0] = r.get_field(interp, "Details")
        cols = [Cell(names), column([r.passed for r in results]).astype(bool),
                column([r.failed for r in results]).astype(bool),
                column([r.incomplete for r in results]).astype(bool),
                column([r.duration for r in results]).astype(float), Cell(details)]
        labels = Cell.from_list([char(x) for x in self.FIELDS])
        return interp.call_builtin("table", interp.builtins["table"], [*cols, char("VariableNames"), labels], 1)

    def array_footer(self, arr: Any) -> list[str]:
        return _totals_lines(_results_of(arr))

    def display_lines(self) -> list[str]:
        from ..display import summary
        w = max(len(n) for n in self.FIELDS)
        return ["  TestResult with properties:", ""] + \
            [f"    {n:>{w}}: {summary(self.get_field(None, n), False)}" for n in self.FIELDS] + \
            _totals_lines([self])


NativeArray.prototypes["matlab.unittest.Test"] = TestElement(Group("function", ""), "", "")
NativeArray.prototypes["matlab.unittest.TestResult"] = TestResult("", False, False, 0.0, [], [])


def _totals_lines(results: list) -> list[str]:
    passed = sum(r.passed for r in results)
    failed = sum(r.failed for r in results)
    incomplete = sum(r.incomplete for r in results)
    return ["", "Totals:", f"   {passed} Passed, {failed} Failed, {incomplete} Incomplete.",
            f"   {sum(r.duration for r in results):.5g} seconds testing time."]


def _row(items: list, kind: str) -> Any:
    if len(items) == 1:
        return items[0]
    data = np.empty((1, len(items)), dtype=object)
    for k, item in enumerate(items):
        data[0, k] = item
    return NativeArray(data, kind)  # 1-by-0 when there are no items


def _suite(tests: list) -> Any:
    return _row(tests, "matlab.unittest.Test")


def _results(results: list) -> Any:
    return _row(results, "matlab.unittest.TestResult")


def _tests_of(v: Any) -> list:
    if isinstance(v, TestElement):
        return [v]
    if isinstance(v, NativeArray) and v.kind == "matlab.unittest.Test":
        return v.elements()
    if isinstance(v, Cell):
        out: list = []
        for item in v.values_f():
            out += _tests_of(item)
        return out
    raise MatlabError("Expected a test suite (an array of matlab.unittest.Test objects).")


def _results_of(v: Any) -> list:
    if isinstance(v, TestResult):
        return [v]
    if isinstance(v, NativeArray) and v.kind == "matlab.unittest.TestResult":
        return v.elements()
    raise MatlabError("Expected an array of test results.")


def _valid_name(title: str) -> str:
    """A section title as a test name: words joined in camel case, other characters as underscores."""
    words = title.split()
    joined = "".join(w if k == 0 else w[:1].upper() + w[1:] for k, w in enumerate(words))
    name = re.sub(r"[^A-Za-z0-9_]", "_", joined)
    if name and not name[0].isalpha():
        name = "x" + name
    return name


def _is_test_name(name: str) -> bool:
    low = name.lower()
    return low.startswith("test") or low.endswith("test")


def _tag_list(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [p.strip().strip("'\"") for p in value.strip("{}[] ").split(",") if p.strip().strip("'\"")]


# -- building suites ---------------------------------------------------------------------------------------

def _function_suite(interp, handles: list, file: str | None) -> list:
    name = os.path.splitext(os.path.basename(file))[0] if file else "tests"
    group = Group("function", name, os.path.dirname(file) if file else os.getcwd())
    group.file = file or ""
    tests = []
    for h in handles:
        if not isinstance(h, FunctionHandle) or h.is_anonymous:
            raise MatlabError("functiontests needs a cell array of handles to local functions; "
                              "write tests = functiontests(localfunctions).")
        group.handles[h.name] = h
    for fname in group.handles:
        if fname.lower() in ("setuponce", "teardownonce", "setup", "teardown"):
            continue
        if _is_test_name(fname):
            tests.append(TestElement(group, fname, f"{name}/{fname}"))
    return tests


@builtin("functiontests")
def _functiontests(interp, args, nargout):
    if len(args) != 1 or not isinstance(args[0], Cell):
        raise MatlabError("functiontests needs a cell array of function handles: "
                          "tests = functiontests(localfunctions).")
    return _suite(_function_suite(interp, args[0].values_f(), interp.frame.file))


def _param_values(interp, cls: ClassInfo, prop: str) -> list[tuple[str, Any]]:
    entry = cls.prop_defs.get(prop)
    if entry is None or not (entry[0].attrs.get("TestParameter") or entry[0].attrs.get("MethodSetupParameter")
                             or entry[0].attrs.get("ClassSetupParameter")):
        raise MatlabError(f"The test method argument '{prop}' of class '{cls.name}' is not a property of a "
                          "properties (TestParameter) block.", "MATLAB:unittest:Parameter:NotFound")
    value = cls.defaults(interp).get(prop)
    if isinstance(value, Struct) and value.data.size == 1:
        return [(f, value.data.flat[0][f]) for f in value.fields]
    if isinstance(value, Cell):
        out = []
        for k, item in enumerate(value.values_f()):
            label = f"value{k + 1}"
            if is_text(item) and re.fullmatch(r"[A-Za-z]\w*", text(item)):
                label = text(item)
            elif isinstance(item, np.ndarray) and item.size == 1 and item.dtype.kind in "fiu" \
                    and float(item.flat[0]) == int(item.flat[0]) and item.flat[0] >= 0:
                label = str(int(item.flat[0]))
            out.append((label, item))
        return out
    if isinstance(value, StringArray):
        return [(x or f"value{k + 1}", MString(x or "")) for k, x in enumerate(value.values_f())]
    raise MatlabError(f"The TestParameter property '{prop}' of class '{cls.name}' must be a nonempty cell "
                      "array or a struct of values.", "MATLAB:unittest:Parameter:InvalidValue")


def _combinations(lists: list, how: str) -> list:
    if not lists:
        return [[]]
    how = how.lower()
    if how == "sequential":
        n = max(len(x) for x in lists)
        if any(len(x) not in (1, n) for x in lists):
            raise MatlabError("With ParameterCombination = 'sequential' every parameter needs the same number "
                              "of values.", "MATLAB:unittest:Parameter:SequentialSizeMismatch")
        return [[x[k if len(x) > 1 else 0] for x in lists] for k in range(n)]
    import itertools
    full = [list(c) for c in itertools.product(*lists)]
    if how == "exhaustive" or len(lists) < 3:
        return full
    if how != "pairwise":
        raise MatlabError(f"Unknown ParameterCombination '{how}'; use exhaustive, sequential or pairwise.")
    # pairwise: greedily pick combinations until every pair of values of two parameters is covered
    need = set()
    for i, j in itertools.combinations(range(len(lists)), 2):
        for a in range(len(lists[i])):
            for b in range(len(lists[j])):
                need.add((i, a, j, b))
    index = [list(c) for c in itertools.product(*[range(len(x)) for x in lists])]
    chosen = []
    while need:
        best, gain = None, -1
        for combo in index:
            g = sum((i, combo[i], j, combo[j]) in need for i, j in itertools.combinations(range(len(lists)), 2))
            if g > gain:
                best, gain = combo, g
        for i, j in itertools.combinations(range(len(lists)), 2):
            need.discard((i, best[i], j, best[j]))
        chosen.append([lists[k][best[k]] for k in range(len(lists))])
    return chosen


def is_test_class(cls: ClassInfo | None) -> bool:
    return cls is not None and "matlab.unittest.TestCase" in cls.bases


def _class_suite(interp, cls: ClassInfo, only: str | None = None) -> list:
    if not is_test_class(cls):
        raise MatlabError(f"Class '{cls.name}' is not a test class; it must derive from "
                          "matlab.unittest.TestCase.", "MATLAB:unittest:TestSuite:NonTestCase")
    for c in cls.lineage():
        if "SharedTestFixtures" in c.cd.attrs:
            raise MatlabError("The SharedTestFixtures class attribute is not supported; apply the fixture "
                              "in a TestClassSetup method with testCase.applyFixture(...).")
    group = Group("class", cls.full_name, os.path.dirname(cls.path) if cls.path else os.getcwd())
    group.cls = cls
    group.file = cls.path or ""
    class_tags: list[str] = []
    for c in cls.lineage():
        class_tags += _tag_list(c.cd.attrs.get("TestTags"))
    tests = []
    for mname, (fn, _owner) in cls.methods.items():
        attrs = cls.method_attrs.get(mname, {})
        if not attrs.get("Test") or (only is not None and mname != only):
            continue
        props = [p for p in fn.params[1:] if p != "~"]
        value_lists = [[(p, label, v) for label, v in _param_values(interp, cls, p)] for p in props]
        tags = tuple(dict.fromkeys(class_tags + _tag_list(attrs.get("TestTags"))))
        for combo in _combinations(value_lists, str(attrs.get("ParameterCombination", "exhaustive"))):
            suffix = "(" + ",".join(f"{p}={label}" for p, label, _ in combo) + ")" if combo else ""
            tests.append(TestElement(group, mname, f"{cls.full_name}/{mname}{suffix}", list(combo), tags))
    if only is not None and not tests:
        raise MatlabError(f"'{only}' is not a Test method of class '{cls.name}'.",
                          "MATLAB:unittest:TestSuite:InvalidMethod")
    return tests


def _script_suite(interp, path: str, program) -> list:
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    stop = len(lines)
    if program.functions:
        stop = min(getattr(fn, "line", stop + 1) for fn in program.functions.values()) - 1
    name = os.path.splitext(os.path.basename(path))[0]
    group = Group("script", name, os.path.dirname(path))
    group.file = path
    group.functions = dict(program.functions)
    heads = [k for k in range(stop) if lines[k].strip().startswith("%%")]

    def chunk(a: int, b: int) -> str:
        return "\n" * a + "\n".join(lines[a:b])  # leading blank lines keep the file's line numbers

    if not heads:
        return [TestElement(group, name, f"{name}/{name}", code=chunk(0, stop))]
    group.shared = chunk(0, heads[0])
    tests = []
    used: set[str] = set()
    for k, start in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else stop
        title = lines[start].strip()[2:].strip()
        proc = _valid_name(title) or f"test_{k + 1}"
        while proc in used:
            proc += "_1"
        used.add(proc)
        tests.append(TestElement(group, proc, f"{name}/{proc}", code=chunk(start + 1, end)))
    return tests


def _calls(node: Any, name: str) -> bool:
    from .. import ast as A
    if isinstance(node, A.Ident) and node.name == name:
        return True
    if isinstance(node, A.Node):
        return any(_calls(v, name) for v in vars(node).values())
    if isinstance(node, (list, tuple)):
        return any(_calls(v, name) for v in node)
    return False


def suite_from_file(interp, path: str, strict: bool = True) -> list:
    path = os.path.abspath(path)
    if not path.endswith(".m"):
        path += ".m"
    program = interp._load_file(path)
    if program is None:
        raise MatlabError(f"The test file '{path}' was not found.", "MATLAB:unittest:TestSuite:FileNotFound")
    name = os.path.splitext(os.path.basename(path))[0]
    if program.classdefs:
        cls = _class_at(interp, path, name)
        if is_test_class(cls) and not cls.is_abstract:
            return _class_suite(interp, cls)
        if strict:
            raise MatlabError(f"'{name}' is not a test class (it must derive from matlab.unittest.TestCase "
                              "and not be abstract).", "MATLAB:unittest:TestSuite:NonTestCase")
        return []
    if program.functions and not program.body:
        main = next(iter(program.functions.values()))
        if not _calls(main.body, "functiontests"):
            if strict:
                raise MatlabError(f"'{name}' is not a test file: its main function must return "
                                  "functiontests(localfunctions).", "MATLAB:unittest:TestSuite:InvalidFile")
            return []
        folder = os.path.dirname(path)
        added = folder not in interp.path
        if added:
            interp.path.insert(0, folder)
        try:
            res = interp.call_user_function(main, [], 1, program.functions, path)
        finally:
            if added and folder in interp.path:
                interp.path.remove(folder)
        return _tests_of(res[0]) if res else []
    return _script_suite(interp, path, program)


def _class_at(interp, path: str, name: str) -> ClassInfo | None:
    """The class defined in ``path`` (found through the path, a package, or loaded directly)."""
    parts = []
    folder = os.path.dirname(path)
    if os.path.basename(folder) == "@" + name:
        folder = os.path.dirname(folder)
    while os.path.basename(folder).startswith("+"):
        parts.insert(0, os.path.basename(folder)[1:])
        folder = os.path.dirname(folder)
    added = folder not in interp.path and folder != os.getcwd()
    if added:
        interp.path.insert(0, folder)
    return interp.find_class(".".join(parts + [name]))


def suite_from_folder(interp, folder: str, deep: bool = False) -> list:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise MatlabError(f"The folder '{folder}' was not found.", "MATLAB:unittest:TestSuite:FolderNotFound")
    tests: list = []
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if entry.endswith(".m") and os.path.isfile(full):
            program = interp._load_file(full)
            if program is None:
                continue
            # test classes and function-based test files count whatever their name; a script is a
            # test only when its name starts or ends with "test"
            if program.classdefs or not program.body or _is_test_name(entry[:-2]):
                tests += suite_from_file(interp, full, strict=False)
        elif entry.startswith("@") and os.path.isfile(os.path.join(full, entry[1:] + ".m")):
            tests += suite_from_file(interp, os.path.join(full, entry[1:] + ".m"), strict=False)
        elif deep and os.path.isdir(full) and entry[:1] not in "+@." and entry != "private":
            tests += suite_from_folder(interp, full, True)
    return tests


def suite_from_package(interp, name: str, deep: bool = False) -> list:
    dirs = list(interp.path)
    if os.getcwd() not in dirs:
        dirs.append(os.getcwd())
    tests: list = []
    found = False
    for d in dirs:
        folder = os.path.join(d, *("+" + p for p in name.split(".")))
        if not os.path.isdir(folder):
            continue
        found = True
        for entry in sorted(os.listdir(folder)):
            full = os.path.join(folder, entry)
            if entry.endswith(".m"):
                program = interp._load_file(full)
                if program is not None and program.classdefs:
                    cls = interp.find_class(f"{name}.{entry[:-2]}")
                    if is_test_class(cls) and not cls.is_abstract:
                        tests += _class_suite(interp, cls)
                elif program is not None and _is_test_name(entry[:-2]):
                    tests += suite_from_file(interp, full, strict=False)
            elif deep and entry.startswith("+") and os.path.isdir(full):
                tests += suite_from_package(interp, f"{name}.{entry[1:]}", True)
    if not found:
        raise MatlabError(f"The package '{name}' was not found.", "MATLAB:unittest:TestSuite:InvalidPackage")
    return tests


def suite_from_name(interp, name: str, deep: bool = False) -> list:
    """Tests named by a file, folder, class, package, or 'File/procedure'."""
    if os.path.isdir(name):
        return suite_from_folder(interp, name, deep)
    if os.path.isfile(name) or os.path.isfile(name + ".m"):
        return suite_from_file(interp, name)
    if "/" in name and not os.path.exists(os.path.dirname(name) or "."):
        parent = name.split("/", 1)[0]
        tests = suite_from_name(interp, parent)
        picked = [t for t in tests if t.name == name or t.name.split("(")[0] == name]
        if not picked:
            raise MatlabError(f"No test named '{name}' was found.", "MATLAB:unittest:TestSuite:InvalidName")
        return picked
    if "/" in name:
        parent, proc = name.rsplit("/", 1)
        tests = suite_from_name(interp, parent)
        picked = [t for t in tests if t.proc == proc or t.name.endswith("/" + proc)]
        if not picked:
            raise MatlabError(f"No test named '{name}' was found.", "MATLAB:unittest:TestSuite:InvalidName")
        return picked
    cls = interp.find_class(name)
    if cls is not None:
        if not is_test_class(cls):
            raise MatlabError(f"Class '{name}' is not a test class; it must derive from "
                              "matlab.unittest.TestCase.", "MATLAB:unittest:TestSuite:NonTestCase")
        return _class_suite(interp, cls)
    found = interp.find_file_function(name)
    if found is not None:
        return suite_from_file(interp, found[1])
    try:
        return suite_from_package(interp, name, deep)
    except MatlabError:
        pass
    raise MatlabError(f"No test file, folder, class or package named '{name}' was found.",
                      "MATLAB:unittest:TestSuite:InvalidName")


# -- selection -----------------------------------------------------------------------------------------------

def _wild(pattern: str) -> re.Pattern:
    return re.compile("".join(".*" if c == "*" else ("." if c == "?" else re.escape(c)) for c in pattern) + r"\Z")


class Selector(NativeItem):
    kind = "matlab.unittest.selectors.Selector"

    def __init__(self, label: str, test: Callable[[Any, TestElement], bool]):
        self.label = label
        self.test = test
        self.kind = "matlab.unittest.selectors." + label.split("(")[0].lstrip("~ ")

    def isa_class(self, name: str) -> bool:
        return name in (self.kind, "matlab.unittest.selectors.Selector")

    def unary_op(self, interp, op: str) -> Any:
        if op != "~":
            return NotImplemented
        inner = self
        return Selector(f"~{self.label}", lambda interp, t: not inner.test(interp, t))

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("&", "|") or not (isinstance(left, Selector) and isinstance(right, Selector)):
            return NotImplemented
        if op == "&":
            return Selector(f"{left.label} & {right.label}",
                            lambda interp, t: left.test(interp, t) and right.test(interp, t))
        return Selector(f"{left.label} | {right.label}",
                        lambda interp, t: left.test(interp, t) or right.test(interp, t))

    def display_lines(self) -> list[str]:
        return [f"  {self.label} selector"]


def _text_test(interp, want: Any, what: str) -> Callable[[Any, str], bool]:
    """A test for text: a constraint, or a name where * and ? stand for any characters."""
    if isinstance(want, Constraint):
        return lambda interp, s: want.check(interp, char(s))[0]
    pattern = _wild(text(want, what))
    return lambda interp, s: bool(pattern.match(s))


def _selector(name: str, getter: Callable[[TestElement], list[str]]) -> Callable:
    def make(interp, args: list) -> Selector:
        if len(args) != 1:
            raise MatlabError(f"{name} needs one value to match.")
        test = _text_test(interp, args[0], name)
        return Selector(name, lambda interp, t: any(test(interp, s) for s in getter(t)))
    builtin("matlab.unittest.selectors." + name)(lambda interp, args, nargout: make(interp, list(args)))
    return make


def _superclasses(t: TestElement) -> list[str]:
    if t.group.cls is None:
        return []
    from .classes_more import superclass_names
    return superclass_names(t.group.cls)


_SELECTORS = {
    "Name": _selector("HasName", lambda t: [t.name]),
    "Tag": _selector("HasTag", lambda t: list(t.tags)),
    "ProcedureName": _selector("HasProcedureName", lambda t: [t.proc]),
    "BaseFolder": _selector("HasBaseFolder", lambda t: [t.group.folder]),
    "Superclass": _selector("HasSuperclass", _superclasses),
    "ParameterProperty": _selector("HasParameterProperty", lambda t: [p for p, _, _ in t.params]),
    "ParameterName": _selector("HasParameterName", lambda t: [n for _, n, _ in t.params]),
}


@builtin("matlab.unittest.selectors.HasParameter")
def _has_parameter(interp, args, nargout):
    opts, extra = _split_options(list(args), ("Property", "Name", "Value"))
    if extra is not None:
        raise MatlabError("HasParameter takes the options Property, Name and Value.")
    prop = _text_test(interp, opts["Property"], "Property") if "Property" in opts else None
    pname = _text_test(interp, opts["Name"], "Name") if "Name" in opts else None

    def test(interp, t: TestElement) -> bool:
        for p, n, v in t.params:
            if prop is not None and not prop(interp, p):
                continue
            if pname is not None and not pname(interp, n):
                continue
            if "Value" in opts:
                want = opts["Value"]
                ok = want.check(interp, v)[0] if isinstance(want, Constraint) else values_equal(v, want)
                if not ok:
                    continue
            return True
        return False

    return Selector("HasParameter", test)


def select_tests(interp, tests: list, args: list) -> list:
    """selectIf(suite, selector) or selectIf(suite, 'Name', pattern, 'Tag', tag, ...)."""
    if len(args) == 1 and isinstance(args[0], Selector):
        sel = args[0]
        return [t for t in tests if sel.test(interp, t)]
    if not args or len(args) % 2:
        raise MatlabError("selectIf needs a selector or name-value pairs such as 'Name', 'MyTest/*'.")
    picked = list(tests)
    for k in range(0, len(args), 2):
        key = text(args[k], "option name")
        match = next((n for n in _SELECTORS if n.lower() == key.lower()), None)
        if match is None:
            raise MatlabError(f"Unknown selection option '{key}'. Use one of: " + ", ".join(_SELECTORS) + ".")
        values = args[k + 1]
        if isinstance(values, Cell):
            options = values.values_f()
        elif isinstance(values, StringArray):
            options = [MString(x or "") for x in values.values_f()]
        else:
            options = [values]
        sels = [_SELECTORS[match](interp, [v]) for v in options]
        picked = [t for t in picked if any(s.test(interp, t) for s in sels)]
    return picked


# -- running -----------------------------------------------------------------------------------------------

class RunOptions:
    def __init__(self, detail: int = 2, logging: int = 1, text_output: bool = True, strict: bool = False):
        self.detail = detail          # 0 silent ... 4 verbose (what the runner prints)
        self.logging = logging        # highest level of log(testCase, ...) messages that are shown
        self.text_output = text_output
        self.strict = strict
        self.junit: str | None = None
        self.stop_on_failure = False


def _error_block(interp, name: str, e: MatlabError, what: str = "") -> str:
    body = _section("Error ID", [f"'{e.identifier or ''}'"])
    where = ""
    line = getattr(e, "line", None)
    func = getattr(e, "func", None)
    if func and line:
        where = f"Error using {func} (line {line})"
    elif line:
        where = f"Error in line {line}"
    body += _section("Error Details", ([where] if where else []) + e.message.split("\n"))
    head = f"Error occurred in {name}{what} and it did not run to completion."
    return f"\n{_RULE}\n{head}\n" + "\n".join(_indent(body)) + f"\n{_RULE}\n"


class _Run:
    """One run of a list of tests: prints progress and collects the results."""

    def __init__(self, interp, opts: RunOptions):
        self.interp = interp
        self.opts = opts
        self.results: list[TestResult] = []
        self.dots = 0

    def say(self, s: str, level: int = 2) -> None:
        if self.opts.text_output and self.opts.detail >= level:
            self.interp.out(s)

    def block(self, s: str) -> None:
        if self.opts.text_output and self.opts.detail >= 1:
            if self.dots:
                self.interp.out("\n")
                self.dots = 0
            self.interp.out(s)

    def new_state(self, name: str) -> TestState:
        st = TestState(name)
        st.emit = self.block
        st.log_level = self.opts.logging
        return st

    def guard(self, st: TestState, fn: Callable[[], Any], what: str = "") -> bool:
        """Run one step of a test; a failure is recorded in ``st``. False means the step did not finish."""
        try:
            fn()
            return True
        except QualificationStop as e:
            if e.kind == "fatal":
                raise
            return False
        except MatlabError as e:
            st.failed = True
            st.incomplete = True
            st.reasons.append("Errored.")
            block = _error_block(self.interp, st.name, e, what)
            st.records.append({"kind": "error", "name": "error", "text": block})
            self.block(block)
            return False

    def teardown(self, st: TestState, mark: int = 0) -> None:
        while len(st.teardowns) > mark:
            fn = st.teardowns.pop()
            self.guard(st, fn, " (teardown)")

    def finish(self, test: TestElement, st: TestState, started: float, shared: TestState | None) -> None:
        failed, incomplete = st.failed, st.incomplete
        reasons = list(st.reasons)
        if shared is not None and (shared.failed or shared.incomplete):
            failed = failed or shared.failed
            incomplete = True
            reasons += [r for r in shared.reasons if r not in reasons]
        if self.opts.strict and not failed and getattr(st, "warned", False):
            failed = True
        self.results.append(TestResult(test.name, failed, incomplete, time.perf_counter() - started,
                                       st.records, reasons))
        self.say(".")
        self.dots += 1
        if self.dots % 50 == 0:
            self.say("\n")

    # -- the three kinds of test files ------------------------------------------------------------------
    def run_group(self, group: Group, tests: list) -> None:
        self.say(f"Running {group.name}\n")
        self.dots = 0
        old_dir = os.getcwd()
        added = bool(group.folder) and group.folder not in self.interp.path and group.folder != old_dir
        if added:
            self.interp.path.insert(0, group.folder)
        try:
            {"function": self.run_functions, "class": self.run_class, "script": self.run_script}[group.kind](
                group, tests)
        finally:
            if added and group.folder in self.interp.path:
                self.interp.path.remove(group.folder)
        self.say(("\n" if self.dots % 50 else "") + f"Done {group.name}\n__________\n\n")
        self.dots = 0

    def run_functions(self, group: Group, tests: list) -> None:
        interp = self.interp
        hooks = {n.lower(): h for n, h in group.handles.items()}
        shared = self.new_state(group.name)
        shared_tc = FunctionTestCase(shared)
        ready = True
        if "setuponce" in hooks:
            ready = self.guard(shared, lambda: interp.call_function_value(hooks["setuponce"], [shared_tc], 0),
                               " (setupOnce)")
        for test in tests:
            st = self.new_state(test.name)
            tc = FunctionTestCase(st)
            tc.test_data = shared_tc.test_data.copy() if isinstance(shared_tc.test_data, Struct) \
                else shared_tc.test_data
            started = time.perf_counter()
            if ready:
                ok = True
                if "setup" in hooks:
                    ok = self.guard(st, lambda: interp.call_function_value(hooks["setup"], [tc], 0), " (setup)")
                mark = len(st.teardowns)
                if ok:
                    self.guard(st, lambda: interp.call_function_value(group.handles[test.proc], [tc], 0))
                self.teardown(st, mark)  # what the test registered with addTeardown, newest first
                if "teardown" in hooks:
                    self.guard(st, lambda: interp.call_function_value(hooks["teardown"], [tc], 0), " (teardown)")
                self.teardown(st)
            self.finish(test, st, started, shared)
        if "teardownonce" in hooks:
            self.guard(shared, lambda: interp.call_function_value(hooks["teardownonce"], [shared_tc], 0),
                       " (teardownOnce)")
        self.teardown(shared)

    def _hooks(self, cls: ClassInfo, attr: str) -> list[str]:
        return [m for m in cls.methods if cls.method_attrs.get(m, {}).get(attr)]

    def run_class(self, group: Group, tests: list) -> None:
        from .classes_more import shallow_copy
        interp = self.interp
        cls = group.cls
        shared = self.new_state(group.name)
        proto: Any = None

        def make() -> None:
            nonlocal proto
            proto = cls.construct(interp, [], 1)[0]
            proto._ut = shared
            for m in self._hooks(cls, "TestClassSetup"):
                cls.call(interp, m, [proto], 0)

        ready = self.guard(shared, make, " (TestClassSetup)")
        for test in tests:
            st = self.new_state(test.name)
            started = time.perf_counter()
            if ready:
                tc = shallow_copy(interp, proto)
                tc._ut = st
                ok = True
                for m in self._hooks(cls, "TestMethodSetup"):
                    ok = ok and self.guard(st, lambda m=m: cls.call(interp, m, [tc], 0), " (TestMethodSetup)")
                mark = len(st.teardowns)
                if ok:
                    values = [v for _, _, v in test.params]
                    self.guard(st, lambda: cls.call(interp, test.proc, [tc, *values], 0))
                self.teardown(st, mark)  # what the test method registered, newest first
                for m in reversed(self._hooks(cls, "TestMethodTeardown")):
                    self.guard(st, lambda m=m: cls.call(interp, m, [tc], 0), " (TestMethodTeardown)")
                self.teardown(st)
            self.finish(test, st, started, shared)
        if proto is not None:
            for m in reversed(self._hooks(cls, "TestClassTeardown")):
                self.guard(shared, lambda m=m: cls.call(interp, m, [proto], 0), " (TestClassTeardown)")
        self.teardown(shared)

    def run_script(self, group: Group, tests: list) -> None:
        from ..interpreter import Frame
        from ..parser import parse
        interp = self.interp
        for test in tests:
            st = self.new_state(test.name)
            started = time.perf_counter()
            frame = Frame(interp, dict(group.functions), name=test.proc, file=group.file)

            def body() -> None:
                interp.frames.append(frame)
                try:
                    if group.shared.strip():
                        interp.run_program(parse(group.shared), frame, group.file)
                    interp.run_program(parse(test.code), frame, group.file)
                finally:
                    interp.frames.pop()
                    if interp.cleanup_count:
                        interp.run_cleanups(frame)

            self.guard(st, body)
            self.finish(test, st, started, None)

    def run(self, tests: list) -> list[TestResult]:
        k = 0
        try:
            while k < len(tests):
                group = tests[k].group
                j = k
                while j < len(tests) and tests[j].group is group:
                    j += 1
                self.run_group(group, tests[k:j])
                k = j
                if self.opts.stop_on_failure and any(r.failed for r in self.results):
                    break
        except QualificationStop as e:
            self.block(f"\n{_RULE}\nA fatal assertion stopped the test run.\n{_RULE}\n")
            raise MatlabError("Fatal assertion failed: the test run was stopped. " + e.message,
                              "MATLAB:unittest:FatalAssertionFailed") from None
        self.summary()
        if self.opts.junit:
            _write_junit(self.opts.junit, self.results)
        return self.results

    def summary(self) -> None:
        bad = [r for r in self.results if not r.passed]
        if not bad or not self.opts.text_output or self.opts.detail < 1:
            return
        w = max(len(r.name) for r in bad) + 2
        lines = ["Failure Summary:", "", "     " + "Name".ljust(w) + "Failed  Incomplete  Reason(s)",
                 "    " + "=" * (w + 44)]
        for r in bad:
            reasons = list(dict.fromkeys(r.reasons)) or ["Did not pass."]
            lines.append("     " + r.name.ljust(w) + ("  X   " if r.failed else "      ") + "  "
                         + ("    X     " if r.incomplete else "          ") + "  " + reasons[0])
            lines += [" " * (w + 25) + extra for extra in reasons[1:]]
        self.interp.out("\n".join(lines) + "\n\n")


def _write_junit(path: str, results: list) -> None:
    from xml.sax.saxutils import escape, quoteattr
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r.name.split("/", 1)[0], []).append(r)
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>"]
    for name, items in groups.items():
        skipped = sum(i.incomplete and not i.failed for i in items)
        out.append(f'  <testsuite name={quoteattr(name)} tests="{len(items)}" '
                   f'failures="{sum(i.failed for i in items)}" skipped="{skipped}" '
                   f'errors="0" time="{sum(i.duration for i in items):.6f}">')
        for i in items:
            case = i.name.split("/", 1)[-1]
            head = f'    <testcase classname={quoteattr(name)} name={quoteattr(case)} time="{i.duration:.6f}"'
            if i.passed:
                out.append(head + "/>")
                continue
            out.append(head + ">")
            body = escape("\n".join(r["text"] for r in i.records))
            out.append(f"      <failure>{body}</failure>" if i.failed else f"      <skipped>{body}</skipped>")
            out.append("    </testcase>")
        out.append("  </testsuite>")
    out.append("</testsuites>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def run_tests(interp, tests: list, opts: RunOptions) -> list[TestResult]:
    return _Run(interp, opts).run(tests)


# -- runtests, testsuite, TestSuite, TestRunner ----------------------------------------------------------

_SUITE_OPTIONS = ("Name", "Tag", "ProcedureName", "BaseFolder", "Superclass", "ParameterProperty",
                  "ParameterName", "IncludeSubfolders", "IncludeSubpackages", "IncludingSubfolders",
                  "IncludingSubpackages", "IncludeInnerPackages", "IncludingInnerPackages")
_RUN_OPTIONS = ("OutputDetail", "LoggingLevel", "Strict", "UseParallel", "Debug", "Verbosity")


def _parse_suite_args(interp, args: list, extra: tuple = ()) -> tuple[list, dict]:
    """Split ``(tests, Name, Value, ...)`` into the test sources and the options."""
    names = _SUITE_OPTIONS + extra
    lowered = {n.lower(): n for n in names}
    sources: list = []
    opts: dict[str, Any] = {}
    k = 0
    if args and not (is_text(args[0]) and text(args[0]).lower() in lowered and len(args) % 2 == 0):
        sources.append(args[0])
        k = 1
    while k < len(args):
        if not is_text(args[k]) or text(args[k]).lower() not in lowered:
            shown = text(args[k]) if is_text(args[k]) else class_name(args[k])
            raise MatlabError(f"Unknown option '{shown}'. Options: " + ", ".join(names) + ".")
        if k + 1 >= len(args):
            raise MatlabError(f"No value was given for the option '{text(args[k])}'.")
        opts[lowered[text(args[k]).lower()]] = args[k + 1]
        k += 2
    return sources, opts


def _build_suite(interp, sources: list, opts: dict) -> list:
    deep = any(_flag(opts[n], n) for n in ("IncludeSubfolders", "IncludingSubfolders", "IncludeSubpackages",
                                           "IncludingSubpackages", "IncludeInnerPackages",
                                           "IncludingInnerPackages") if n in opts)
    tests: list = []
    if not sources:
        tests = suite_from_folder(interp, os.getcwd(), deep)
    for src in sources:
        if isinstance(src, (TestElement, NativeArray)):
            tests += _tests_of(src)
        elif isinstance(src, Cell):
            for item in src.values_f():
                tests += _tests_of(item) if isinstance(item, (TestElement, NativeArray)) \
                    else suite_from_name(interp, to_str(item, "test name"), deep)
        elif isinstance(src, StringArray):
            for item in src.values_f():
                tests += suite_from_name(interp, item or "", deep)
        elif is_text(src):
            tests += suite_from_name(interp, text(src), deep)
        else:
            raise MatlabError("Tests are given as a file, folder, class or package name, a cell array of "
                              "names, or a test suite.")
    pairs: list = []
    for key in ("Name", "Tag", "ProcedureName", "BaseFolder", "Superclass", "ParameterProperty",
                "ParameterName"):
        if key in opts:
            pairs += [char(key), opts[key]]
    return select_tests(interp, tests, pairs) if pairs else tests


def _run_options(opts: dict) -> RunOptions:
    ro = RunOptions()
    if "OutputDetail" in opts:
        ro.detail = _level_of(opts["OutputDetail"])
    if "Verbosity" in opts:
        ro.detail = _level_of(opts["Verbosity"])
    if "LoggingLevel" in opts:
        ro.logging = _level_of(opts["LoggingLevel"])
    if "Strict" in opts and _flag(opts["Strict"], "Strict"):
        raise MatlabError("runtests: 'Strict', true (warnings fail tests) is not supported; use "
                          "verifyWarningFree in the tests that must not warn.")
    if "UseParallel" in opts and _flag(opts["UseParallel"], "UseParallel"):
        raise MatlabError("runtests: 'UseParallel', true is not supported; tests run one after another.")
    if "Debug" in opts and _flag(opts["Debug"], "Debug"):
        raise MatlabError("runtests: 'Debug', true is not supported.")
    return ro


@builtin("runtests")
def _runtests(interp, args, nargout):
    sources, opts = _parse_suite_args(interp, list(args), _RUN_OPTIONS)
    tests = _build_suite(interp, sources, opts)
    results = run_tests(interp, tests, _run_options(opts))
    return _results(results)


@builtin("testsuite")
def _testsuite(interp, args, nargout):
    sources, opts = _parse_suite_args(interp, list(args))
    return _suite(_build_suite(interp, sources, opts))


def _suite_factory(name: str):
    def deco(fn: Callable) -> Callable:
        builtin("matlab.unittest.TestSuite." + name)(fn)
        return fn
    return deco


def _selected(interp, tests: list, rest: list) -> Any:
    """Apply the trailing selector or name-value selection arguments of a TestSuite.from... call."""
    rest = list(rest)
    deep_names = {"includingsubfolders", "includesubfolders", "includingsubpackages", "includesubpackages",
                  "includinginnerpackages", "includeinnerpackages"}
    kept = []
    k = 0
    while k < len(rest):
        if is_text(rest[k]) and text(rest[k]).lower() in deep_names and k + 1 < len(rest):
            k += 2
            continue
        kept.append(rest[k])
        k += 1
    return _suite(select_tests(interp, tests, kept) if kept else tests)


def _deep(rest: list) -> bool:
    for k in range(len(rest) - 1):
        if is_text(rest[k]) and text(rest[k]).lower() in ("includingsubfolders", "includesubfolders",
                                                          "includingsubpackages", "includesubpackages",
                                                          "includinginnerpackages", "includeinnerpackages"):
            return _flag(rest[k + 1], text(rest[k]))
    return False


@_suite_factory("fromFile")
def _from_file(interp, args, nargout):
    if not args:
        raise MatlabError("TestSuite.fromFile needs a file name.")
    return _selected(interp, suite_from_file(interp, text(args[0], "file name")), args[1:])


@_suite_factory("fromFolder")
def _from_folder(interp, args, nargout):
    if not args:
        raise MatlabError("TestSuite.fromFolder needs a folder name.")
    return _selected(interp, suite_from_folder(interp, text(args[0], "folder"), _deep(list(args[1:]))), args[1:])


@_suite_factory("fromClass")
def _from_class(interp, args, nargout):
    if not args:
        raise MatlabError("TestSuite.fromClass needs a class, e.g. fromClass(?MyTest).")
    cls = interp.find_class(_class_arg(args[0]))
    if cls is None:
        raise MatlabError(f"Class '{_class_arg(args[0])}' was not found.")
    return _selected(interp, _class_suite(interp, cls), args[1:])


@_suite_factory("fromMethod")
def _from_method(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("TestSuite.fromMethod needs a class and a method name: fromMethod(?MyTest, 'name').")
    cls = interp.find_class(_class_arg(args[0]))
    if cls is None:
        raise MatlabError(f"Class '{_class_arg(args[0])}' was not found.")
    method = args[1].name if getattr(args[1], "kind", "") == "meta.method" else text(args[1], "method name")
    return _selected(interp, _class_suite(interp, cls, method), args[2:])


@_suite_factory("fromPackage")
def _from_package(interp, args, nargout):
    if not args:
        raise MatlabError("TestSuite.fromPackage needs a package name.")
    return _selected(interp, suite_from_package(interp, text(args[0], "package"), _deep(list(args[1:]))),
                     args[1:])


@_suite_factory("fromName")
def _from_name(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("TestSuite.fromName needs the name of one test, e.g. 'MyTest/testA'.")
    names = [to_str(x) for x in args[0].values_f()] if isinstance(args[0], Cell) else [text(args[0])]
    tests: list = []
    for n in names:
        tests += suite_from_name(interp, n)
    return _suite(tests)


class TestRunner(NativeItem):
    __test__ = False
    kind = "matlab.unittest.TestRunner"
    is_handle = True

    def __init__(self, opts: RunOptions):
        self.opts = opts

    def has_method(self, name: str) -> bool:
        return name in ("run", "addPlugin", "runInParallel")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "run":
            if len(args) != 2:
                raise MatlabError("run needs the runner and a suite: results = run(runner, suite).")
            return [_results(run_tests(interp, _tests_of(args[1]), self.opts))]
        if name == "runInParallel":
            raise MatlabError("runInParallel is not supported; use run(runner, suite).")
        if len(args) != 2 or not isinstance(args[1], Plugin):
            raise MatlabError("addPlugin needs a plugin object, e.g. "
                              "matlab.unittest.plugins.XMLPlugin.producingJUnitFormat('results.xml').")
        args[1].apply(self.opts)
        return []

    def get_field(self, interp, name: str) -> Any:
        if name == "ArtifactsRootFolder":
            import tempfile
            return MString(tempfile.gettempdir())
        if name == "PrebuiltFixtures":
            return np.zeros((0, 0))
        raise MatlabError(f"Unrecognized property '{name}' for class 'matlab.unittest.TestRunner'.")

    def display_lines(self) -> list[str]:
        return ["  TestRunner with properties:", "", "    ArtifactsRootFolder: (temporary folder)"]


class Plugin(NativeItem):
    kind = "matlab.unittest.plugins.TestRunnerPlugin"

    def __init__(self, label: str, apply: Callable[[RunOptions], None]):
        self.kind = "matlab.unittest.plugins." + label
        self.apply = apply

    def display_lines(self) -> list[str]:
        return [f"  {self.kind.rsplit('.', 1)[-1]}"]


def _runner_options(args: list, what: str) -> RunOptions:
    opts, extra = _split_options(list(args), ("OutputDetail", "Verbosity", "LoggingLevel"))
    if extra is not None:
        raise MatlabError(f"{what}: unexpected argument; options are OutputDetail and LoggingLevel.")
    return _run_options(opts)


@builtin("matlab.unittest.TestRunner.withTextOutput", "matlab.unittest.TestRunner.withDefaultPlugins")
def _with_text_output(interp, args, nargout):
    return TestRunner(_runner_options(list(args), "TestRunner.withTextOutput"))


@builtin("matlab.unittest.TestRunner.withNoPlugins")
def _with_no_plugins(interp, args, nargout):
    if args:
        raise MatlabError("TestRunner.withNoPlugins takes no arguments.")
    ro = RunOptions()
    ro.text_output = False
    return TestRunner(ro)


@builtin("matlab.unittest.plugins.XMLPlugin.producingJUnitFormat")
def _junit_plugin(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("XMLPlugin.producingJUnitFormat needs the name of the file to write.")
    path = text(args[0], "file name")

    def apply(ro: RunOptions) -> None:
        ro.junit = path

    return Plugin("XMLPlugin", apply)


@builtin("matlab.unittest.plugins.StopOnFailuresPlugin")
def _stop_plugin(interp, args, nargout):
    if args:
        raise MatlabError("StopOnFailuresPlugin: options are not supported.")

    def apply(ro: RunOptions) -> None:
        ro.stop_on_failure = True

    return Plugin("StopOnFailuresPlugin", apply)


@builtin("matlab.unittest.plugins.TestRunProgressPlugin.withVerbosity")
def _progress_plugin(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("TestRunProgressPlugin.withVerbosity needs a verbosity level.")
    level = _level_of(args[0])

    def apply(ro: RunOptions) -> None:
        ro.text_output = True
        ro.detail = level

    return Plugin("TestRunProgressPlugin", apply)


@builtin("matlab.unittest.plugins.LoggingPlugin.withVerbosity")
def _logging_plugin(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("LoggingPlugin.withVerbosity needs a verbosity level.")
    level = _level_of(args[0])

    def apply(ro: RunOptions) -> None:
        ro.logging = level

    return Plugin("LoggingPlugin", apply)


@builtin("matlab.unittest.plugins.DiagnosticsOutputPlugin", "matlab.unittest.plugins.FailureDiagnosticsPlugin")
def _diagnostics_plugin(interp, args, nargout):
    if args:
        raise MatlabError("This plugin takes no arguments in mylab.")

    def apply(ro: RunOptions) -> None:
        ro.text_output = True
        ro.detail = max(ro.detail, 1)

    return Plugin("DiagnosticsOutputPlugin", apply)


@_method("run")
def _run_testcase(interp, args, nargout):
    """run(testCase) / run(testCase, 'method'): run the Test methods of a test class instance."""
    if not args or not isinstance(args[0], UserObject):
        raise MatlabError("run(testCase) needs an object of a test class.")
    cls = args[0].cls
    only = text(args[1], "method name") if len(args) > 1 else None
    if len(args) > 2:
        raise MatlabError("run(testCase, method) takes at most one method name.")
    return _results(run_tests(interp, _class_suite(interp, cls, only), RunOptions()))
