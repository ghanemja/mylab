"""Regression tests for MATLAB conformance of the numeric library.

Each case compares an expression with the value and class MATLAB gives. The list follows the
differences found by tests/conformance (integer and single classes, N-D reductions, set
functions with NaN and 'rows', empty-result sizes and a few missing functions).
"""

import numpy as np
import pytest

from mylab.errors import MatlabError

SAME = [
    # mod / rem / gcd / lcm keep the integer class and single
    ("mod(int8(-7), int8(3))", "int8(2)"),
    ("rem(int8(-7), int8(3))", "int8(-1)"),
    ("mod(int8(7), 3)", "int8(1)"),
    ("mod(7, int8(3))", "int8(1)"),
    ("mod(int8(7), int8(0))", "int8(7)"),
    ("rem(int8(7), int8(0))", "int8(0)"),
    ("mod(int32(10), 2.5)", "int32(1)"),
    ("mod(int16([10 -10]), int16(4))", "int16([2 2])"),
    ("mod(single(7.5), 2)", "single(1.5)"),
    ("mod(uint64(18446744073709551615), uint64(10))", "uint64(5)"),
    ("gcd(int8(12), int8(18))", "int8(6)"),
    ("lcm(int16(6), int16(15))", "int16(30)"),
    ("lcm(int8(100), int8(3))", "int8(127)"),
    ("gcd([12 9 0], 6)", "[6 3 6]"),
    # bit functions
    ("bitshift(uint8(255), 1)", "uint8(254)"),
    ("bitshift(uint8(1), 8)", "uint8(0)"),
    ("bitshift(int8(64), 1)", "int8(-128)"),
    ("bitshift(int8(-8), -1)", "int8(-4)"),
    ("bitshift([1 2 4], [1 2 3])", "[2 8 32]"),
    ("bitshift(uint8(16), -5)", "uint8(0)"),
    ("bitand(uint8(255), uint8(15))", "uint8(15)"),
    ("bitor(uint8(1), 2)", "uint8(3)"),
    ("bitxor(uint16(65535), uint16(255))", "uint16(65280)"),
    ("bitcmp(uint8(5))", "uint8(250)"),
    ("bitget(uint8(255), 8)", "uint8(1)"),
    ("bitset(uint8(0), 8)", "uint8(128)"),
    ("bitand(2^52 + 1, 1)", "1"),
    ("swapbytes(uint16(1))", "uint16(256)"),
    # set functions
    ("intersect([1 2; 3 4; 5 6], [3 4; 1 2; 7 8], 'rows')", "[1 2; 3 4]"),
    ("setdiff([1 2; 3 4; 5 6], [3 4], 'rows')", "[1 2; 5 6]"),
    ("union([1 2; 3 4], [3 4; 0 1], 'rows')", "[0 1; 1 2; 3 4]"),
    ("setxor([1 2; 3 4], [3 4; 5 6], 'rows')", "[1 2; 5 6]"),
    ("unique([1 2; 3 4; 1 2; 0 9], 'rows', 'stable')", "[1 2; 3 4; 0 9]"),
    ("union([1 2], [2; 3])", "[1; 2; 3]"),
    ("intersect([1 2 3], [3; 2])", "[2; 3]"),
    ("unique([NaN NaN 1])", "[1 NaN NaN]"),
    ("union([1 NaN], [NaN 2])", "[1 2 NaN NaN]"),
    ("union([3 NaN 1], [2 NaN])", "[1 2 3 NaN NaN]"),
    ("setdiff([1 NaN 2], NaN)", "[1 2 NaN]"),
    ("intersect([1 NaN 2], [NaN 2])", "2"),
    ("setdiff('hello', 'l')", "'eho'"),
    ("union('abc', 'bcd')", "'abcd'"),
    ("nchoosek('abc', 2)", "['ab'; 'ac'; 'bc']"),
    ("unique([true false true])", "[false true]"),
    ("unique(int8([3 1 1]))", "int8([1 3])"),
    ("intersect([1 2 3 4], [4 2], 'stable')", "[2 4]"),
    ("union([3 1], [2 1], 'stable')", "[3 1 2]"),
    ("size(setdiff([1 2 3], [1 2 3]))", "[1 0]"),
    # reductions on N-D input and along dimensions beyond ndims
    ("sum(ones(1, 1, 3))", "3"),
    ("prod(2 * ones(1, 1, 3))", "8"),
    ("mean(cat(3, 1, 2, 6))", "3"),
    ("max(cat(3, 1, 5, 2))", "5"),
    ("cumsum(ones(1, 1, 3))", "cat(3, 1, 2, 3)"),
    ("any([1 0; 0 0], 3)", "logical([1 0; 0 0])"),
    ("all([1 0; 2 3], 3)", "logical([1 0; 1 1])"),
    ("cumsum([1 2; 3 4], 3)", "[1 2; 3 4]"),
    ("sort([3 1; 2 4], 3)", "[3 1; 2 4]"),
    ("var([1 2; 3 4], 0, 3)", "[0 0; 0 0]"),
    ("std([1 2; 3 4], 0, 3)", "[0 0; 0 0]"),
    ("sum([1 2; 3 4], 3)", "[1 2; 3 4]"),
    ("max([1 2; 3 4], [], 3)", "[1 2; 3 4]"),
    ("median([1 2; 3 4], 3)", "[1 2; 3 4]"),
    ("mode([1 2; 3 4], 3)", "[1 2; 3 4]"),
    # output class options
    ("sum(int8([100 100]), 'double')", "200"),
    ("sum(int8([100 100]))", "int8(127)"),
    ("sum([true true], 'native')", "true"),
    ("sum(single([1 2]), 'double')", "3"),
    ("sum(int8([]))", "int8(0)"),
    ("mean(int8([1 2]), 'native')", "int8(2)"),
    ("mean(single([1 2]), 'double')", "1.5"),
    ("mean(int8([100 100 100]))", "100"),
    ("median(int8([1 2 4 5]))", "int8(3)"),
    ("median(uint8([255 254]))", "uint8(255)"),
    ("max(single(1), 2)", "single(2)"),
    ("max(1, single(2))", "single(2)"),
    ("max(int8(1), 5.7)", "int8(6)"),
    ("min(int8(-5), -200)", "int8(-128)"),
    ("max(true, false)", "true"),
    ("max([true false])", "true"),
    ("max(true, 2)", "2"),
    ("linspace(single(0), 1, 3)", "single([0 0.5 1])"),
    ("linspace(0, 1, 2.5)", "[0 1]"),
    ("norm(single([3 4]))", "single(5)"),
    ("sign(single(-2))", "single(-1)"),
    ("dot(single([1 2]), [3 4])", "single(11)"),
    ("cumsum(int8([100 100 -100]))", "int8([100 127 27])"),
    # sorting
    ("sort([Inf -Inf NaN 0], 'descend')", "[NaN Inf 0 -Inf]"),
    ("sort([1+2i 1-2i])", "[1-2i 1+2i]"),
    ("sort([true false true])", "[false true true]"),
    ("sort(uint8([200 5 100]), 'descend')", "uint8([200 100 5])"),
    ("sort('hello')", "'ehllo'"),
    # options
    ("std([1 2 3; 4 5 6], 0, 'all')", "std([1 2 3 4 5 6])"),
    ("var([1 2 3; 4 5 6], 0, 'all')", "3.5"),
    ("var([1 2 3; 4 5 6], 0, [1 2])", "3.5"),
    ("var([1 2 3 4], [1 1 1 1])", "1.25"),
    ("var([1 2 3 4], [0.1 0.2 0.3 0.4])", "1"),
    ("std([1 2 3 4], [1 2 3 4])", "1"),
    ("var([1 NaN 3 5], 'omitnan')", "4"),
    ("sum([1 2 3; 4 5 6], [1 2])", "21"),
    ("size(sum(ones(2, 3, 4), [1 3]))", "[1 3]"),
    ("max([1 2 3; 4 5 6], [], [1 2])", "6"),
    ("max([1 2 3; 4 5 6], [], 'all')", "6"),
    ("max([1 NaN 3], [], 'includenan')", "NaN"),
    ("max([1 NaN 3], [], 'omitnan')", "3"),
    ("cumsum([1 NaN 3], 'omitnan')", "[1 1 4]"),
    ("cumsum([1 2 3], 'reverse')", "[6 5 3]"),
    ("cummax([1 NaN 2])", "[1 1 2]"),
    ("round(123456, 2, 'significant')", "120000"),
    ("round(0.00012345, 3, 'significant')", "0.000123"),
    ("round(1234.567, -2)", "1200"),
    ("round(2.567, 2)", "2.57"),
    ("round(int16(1234), -2)", "int16(1200)"),
    ("round(0.49999999999999994)", "0"),
    ("1 ./ ceil(-0.5)", "-Inf"),
    ("1 ./ fix(-0.5)", "-Inf"),
    ("1 ./ round(-0.4)", "-Inf"),
    ("dot([1 2; 3 4], [1 2; 3 4], 2)", "[5; 25]"),
    ("dot([1 2; 3 4], [1 2; 3 4])", "[10 20]"),
    # trapz / cumtrapz
    ("trapz([1 2; 3 4], 2)", "[1.5; 3.5]"),
    ("trapz([1 2 3], 2)", "4"),
    ("trapz([0 1], [1 2; 3 4], 2)", "[1.5; 3.5]"),
    ("trapz(0.1, [1 2 3])", "0.4"),
    ("trapz([])", "0"),
    ("trapz([0 0.5 1], [1 2; 3 4; 5 6])", "[3 4]"),
    ("cumtrapz([1 2; 3 4], 2)", "[0 1.5; 0 3.5]"),
    ("cumtrapz([0 1 3], [1 2 3])", "[0 1.5 6.5]"),
    # elementary functions
    ("fix(-2.7 + 2.7i)", "-2 + 2i"),
    ("gamma(-1)", "Inf"),
    ("gamma(0)", "Inf"),
    ("realsqrt(9)", "3"),
    ("pow2(3)", "8"),
    ("pow2(3, 2)", "12"),
    # polynomials
    ("polyder([1 2], [1 3])", "[2 5]"),
    ("polyder(5)", "0"),
    ("polyder([1 2 3])", "[2 2]"),
    ("poly([])", "1"),
    ("size(roots([]))", "[0 0]"),
    ("size(roots([0 0]))", "[0 0]"),
    ("roots([1 -3 2])", "[2; 1]"),
    # empty-result sizes
    ("size(find([]))", "[0 0]"),
    ("size(find(0))", "[0 0]"),
    ("size(find([0 0 0]))", "[1 0]"),
    ("size(find([0; 0]))", "[0 1]"),
    ("size(diag([]))", "[0 0]"),
    ("size(diff(5))", "[0 0]"),
    ("size(diff(zeros(1, 0)))", "[1 0]"),
    ("size(max(zeros(3, 0)))", "[1 0]"),
    ("size(max(zeros(0, 3), [], 2))", "[0 1]"),
    ("std(zeros(0, 3))", "[NaN NaN NaN]"),
    ("var(zeros(0, 3))", "[NaN NaN NaN]"),
    ("mean(zeros(0, 3))", "[NaN NaN NaN]"),
    ("sum(zeros(0, 3))", "[0 0 0]"),
    ("isvector(zeros(1, 0))", "true"),
    ("isvector(zeros(0, 1))", "true"),
    ("isvector([])", "false"),
    ("interp2([1 2; 3 4], 1.5, 1.5, 'nearest')", "4"),
    ("num2cell([1 2; 3 4], 1)", "{[1; 3], [2; 4]}"),
    ("num2cell([1 2; 3 4], 2)", "{[1 2]; [3 4]}"),
    ("size([1 2 3; 4 5 6], [1 2])", "[2 3]"),
    ("size(ones(2, 3), 1, 2)", "[2 3]"),
    ("size(repmat([1 2], -1, 2))", "[0 4]"),
    ("cast(65, 'char')", "'A'"),
    ("cast(5, 'like', int16(1))", "int16(5)"),
    ("isreal({1})", "false"),
    # added functions
    ("isbanded([1 2 0; 3 4 5; 0 6 7], 1, 1)", "true"),
    ("blkdiag([1 2; 3 4], 5)", "[1 2 0; 3 4 0; 0 0 5]"),
    ("invhilb(2)", "[4 -6; -6 12]"),
    ("invhilb(3)", "[9 -36 30; -36 192 -180; 30 -180 180]"),
    ("compan([1 2 3])", "[-2 -3; 1 0]"),
    ("del2([1 4 9 16 25])", "[0.5 0.5 0.5 0.5 0.5]"),
    ("del2(magic(4))", "[15 -5.5 -6.5 9; 0.5 -5 -3 3.5; -3.5 3 5 -0.5; -9 6.5 5.5 -15]"),
    ("rat(pi)", "'3 + 1/(7 + 1/16)'"),
    ("rat(0.75)", "'1 + 1/(-4)'"),
    ("nnz(isoutlier([1 1 1 1 100]))", "1"),
    ("isoutlier([1 2 3 2 1 2 50])", "logical([0 0 0 0 0 0 1])"),
    ("standardizeMissing([1 -99 3], -99)", "[1 NaN 3]"),
    ("anynan([1 NaN])", "true"),
    ("anynan([1 2])", "false"),
    ("allfinite([1 2])", "true"),
    ("allfinite([1 Inf])", "false"),
    ("uniquetol([1 1 + 1e-14 2])", "[1 2]"),
    ("height([1 2; 3 4; 5 6])", "3"),
    ("width([1 2; 3 4; 5 6])", "2"),
]

CLOSE = [
    ("asin(2)", 1.5707963267948966 - 1.3169578969248166j),
    ("asin(-2)", -1.5707963267948966 + 1.3169578969248166j),
    ("acos(2)", 1.3169578969248166j),
    ("acos(-2)", np.pi - 1.3169578969248166j),
    ("acosh(-2)", 1.3169578969248166 + np.pi * 1j),
    ("acosh(0.5)", 1.0471975511965979j),
]

ERRORS = [
    "bitand(uint8(1), uint16(1))",
    "min(uint8(5), int8(3))",
    "mod(int8(1), int16(1))",
    "reshape(1:6, [], [])",
    "realsqrt(-1)",
]


@pytest.mark.parametrize("expr,expected", SAME)
def test_same_value_and_class(m, expr, expected):
    m.run(f"got = {expr}; want = {expected};")
    assert m.str("class(got)") == m.str("class(want)"), f"{expr}: class"
    assert bool(m.num("isequaln(got, want)").all()), f"{expr}: got {m.val('got')!r}"


@pytest.mark.parametrize("expr,expected", CLOSE)
def test_complex_branches(m, expr, expected):
    got = complex(m.num(expr).flat[0])
    assert abs(got - expected) < 1e-12, f"{expr}: got {got}"


@pytest.mark.parametrize("expr", ERRORS)
def test_errors(m, expr):
    with pytest.raises(MatlabError):
        m.run(f"tmp = {expr};")


def test_sort_descend_index_with_nan_and_inf(m):
    m.run("[s, i] = sort([Inf -Inf NaN 0], 'descend');")
    assert m.num("i").tolist() == [[3, 1, 4, 2]]


def test_sort_index_beyond_ndims(m):
    m.run("[s, i] = sort([3 1; 2 4], 3);")
    assert m.num("s").tolist() == [[3, 1], [2, 4]]
    assert m.num("i").tolist() == [[1, 1], [1, 1]]


def test_union_index_outputs(m):
    m.run("[c, ia, ib] = union([5 1 3], [3 5 7]);")
    assert m.num("c").tolist() == [[1, 3, 5, 7]]
    assert m.num("ia").tolist() == [[2], [3], [1]]
    assert m.num("ib").tolist() == [[3]]


def test_intersect_rows_indices(m):
    m.run("[c, ia, ib] = intersect([1 2; 3 4; 5 6], [3 4; 1 2; 7 8], 'rows');")
    assert m.num("ia").tolist() == [[1], [2]]
    assert m.num("ib").tolist() == [[2], [1]]


def test_unique_rows_stable_indices(m):
    m.run("[u, i, j] = unique([1 2; 3 4; 1 2; 0 9], 'rows', 'stable');")
    assert m.num("i").tolist() == [[1], [2], [4]]
    assert m.num("j").tolist() == [[1], [2], [1], [3]]


def test_unique_keeps_every_nan(m):
    m.run("[u, i] = unique([NaN 1 NaN], 'stable');")
    assert m.num("isequaln(u, [NaN 1 NaN])").all()
    assert m.num("i").tolist() == [[1], [2], [3]]


def test_find_third_output_is_logical_for_logical_input(m):
    m.run("[r, c, v] = find([true false; true true]);")
    assert m.str("class(v)") == "logical"
    assert m.num("r").tolist() == [[1], [2], [2]]


def test_mode_outputs(m):
    m.run("[v, f, c] = mode([1 1 2 2 3]);")
    assert m.num("v").item() == 1
    assert m.num("f").item() == 2
    assert m.num("isequal(c, {[1; 2]})").all()
    m.run("[v, f] = mode([1 2; 1 3; 2 3]);")
    assert m.num("v").tolist() == [[1, 3]]
    assert m.num("f").tolist() == [[2, 2]]


def test_log2_two_outputs(m):
    m.run("[f, e] = log2([1 0.5 10]);")
    assert np.allclose(m.num("f"), [[0.5, 0.5, 0.625]])
    assert m.num("e").tolist() == [[1, 0, 4]]


def test_max_includenan_index(m):
    m.run("[v, i] = max([1 NaN 3], [], 'includenan');")
    assert np.isnan(m.num("v").item())
    assert m.num("i").item() == 2


def test_max_all_index_is_linear(m):
    m.run("[v, i] = max([1 9; 4 2], [], 'all');")
    assert m.num("v").item() == 9
    assert m.num("i").item() == 3


def test_polyder_quotient(m):
    m.run("[q, d] = polyder([1], [1 2]);")
    assert m.num("q").tolist() == [[-1]]
    assert m.num("d").tolist() == [[1, 4, 4]]


def test_integral_array_valued(m):
    got = m.num("integral(@(x) [x, 2 * x; x.^2, 1], 0, 1, 'ArrayValued', true)")
    assert np.allclose(got, [[0.5, 1], [1 / 3, 1]])


def test_rat_two_outputs(m):
    m.run("[n, d] = rat([0.75 pi]);")
    assert m.num("n").tolist() == [[3, 355]]
    assert m.num("d").tolist() == [[4, 113]]


def test_size_with_dimension_list_outputs(m):
    m.run("[a, b] = size(ones(2, 3, 4), [1 3]);")
    assert (m.num("a").item(), m.num("b").item()) == (2, 4)


def test_uniquetol_indices(m):
    m.run("[c, ia, ic] = uniquetol([3 1 1 + 1e-13 3]);")
    assert m.num("c").tolist() == [[1, 3]]
    assert m.num("ic").tolist() == [[2], [1], [1], [2]]


def test_height_width_still_work_on_tables(m):
    m.run("t = table([1; 2; 3], [4; 5; 6]);")
    assert m.num("height(t)").item() == 3
    assert m.num("width(t)").item() == 2
