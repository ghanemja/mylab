"""Reference counts that mean "nothing else can see this array".

A few fast paths modify an array in place instead of copying it, when no other variable can
observe the change: ``x(i) = v`` on an unshared array, the loop compiler's element stores, and
``a.*b + c`` reusing the temporary ``a.*b``. They detect "unshared" with ``sys.getrefcount``.

What that function returns for the same situation depends on the Python version. 3.11 to 3.13
count the reference held by a called function's parameter and by getrefcount's own argument;
3.10 also counts the caller's copy on its evaluation stack; 3.14 *borrows* references when a
local variable is passed to a function, so neither the parameter nor the argument is counted.
Hard-coding the numbers for one version gives wrong answers on another (on 3.14 a shared array
looked unshared, so assigning into one variable changed another).

So each number is measured here, once, by a function with the same shape as the code that uses
it. Where the measurement is not trustworthy (a free-threaded build, where reference counts are
deferred, or a count below the number of references that must exist) the value is -1, which
turns the corresponding in-place path off: the code then always copies, which is slower but
always correct.
"""

from __future__ import annotations

import sys

import numpy as np


def _dict_and_local() -> int:
    # interpreter._assign_inplace and jit._prepare:  arr = frame.vars.get(name); getrefcount(arr)
    store = {"x": np.empty((1, 1))}
    arr = store.get("x")
    return sys.getrefcount(arr)


def _element() -> int:
    # jit._prepare: the fields dict of a 1x1 struct, read straight from its object array
    box = np.empty((1, 1), dtype=object)
    box[0, 0] = {}
    return sys.getrefcount(box[0, 0])


def _callee(a, i, x):
    return sys.getrefcount(a)


def _caller_local() -> int:
    # jit._set1/_set2, called from generated code with a local variable:  _set1(a, i, x)
    a = np.empty((1, 1))
    return _callee(a, 1.0, 2.0)


class _Fresh:
    def binop(self, op, left, right):
        return sys.getrefcount(left), sys.getrefcount(right)

    @staticmethod
    def make():
        return np.empty((1, 1))

    def measure(self):
        # Interpreter._binop_fresh:  self._binop_fresh(op, self.eval(l), self.eval(r))
        return self.binop("+", self.make(), self.make())


def _measure() -> tuple[int, int, int, int, int]:
    gil = getattr(sys, "_is_gil_enabled", None)
    if gil is not None and not gil():
        return -1, -1, -1, -1, -1
    dict_and_local = _dict_and_local()
    element = _element()
    caller_local = _caller_local()
    fresh_left, fresh_right = _Fresh().measure()
    # At least this many references must exist in each case (the dict and the local; the object
    # array; the caller's local; the argument itself). A smaller count means getrefcount is not
    # telling the whole story, so the in-place path is not used.
    dict_and_local = dict_and_local if dict_and_local >= 2 else -1
    element = element if element >= 2 else -1
    caller_local = caller_local if caller_local >= 1 else -1
    fresh_left = fresh_left if fresh_left >= 1 else -1
    fresh_right = fresh_right if fresh_right >= 1 else -1
    return dict_and_local, element, caller_local, fresh_left, fresh_right


class _Slot:
    __slots__ = ("data",)


def _slot_and_local() -> int:
    # interpreter._assign_field_inplace:  data = struct.data; getrefcount(data)  (Struct keeps it in a slot)
    holder = _Slot()
    holder.data = np.empty((1, 1), dtype=object)
    data = holder.data
    return sys.getrefcount(data)


def _measure_slot() -> int:
    gil = getattr(sys, "_is_gil_enabled", None)
    if gil is not None and not gil():
        return -1
    n = _slot_and_local()
    return n if n >= 2 else -1  # (the slot and the local must both be counted)


#: ``getrefcount(data)`` for an array held by one ``__slots__`` attribute and one local variable.
SLOT_AND_LOCAL = _measure_slot()

#: ``getrefcount(arr)`` for an array held by one dict entry and one local variable, nothing else.
#: Also the struct object itself in ``jit._prepare``.
#: ``getrefcount(box[0, 0])`` for an object-array element held by nothing but the array.
#: ``getrefcount(a)`` inside a function called with the caller's local ``a`` (its only holder).
#: ``getrefcount(left)`` / ``getrefcount(right)`` for call results passed straight to a method.
DICT_AND_LOCAL, ELEMENT, CALLER_LOCAL, FRESH_LEFT, FRESH_RIGHT = _measure()
