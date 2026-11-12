"""Regression tests for MATLAB-conformance fixes in the engine core.

Covers the interpreter, operators, indexing, the lexer and the type-construction
built-ins: output forwarding, 64-bit integers, single propagation, classes of
concatenations, assignment and deletion shapes, hexadecimal literals, and the
small argument-form gaps found by tests/conformance.
"""

import numpy as np
import pytest

from mylab import MatlabError
from mylab.values import class_name


def cls(m, expr):
    return class_name(m.interp.eval_expression(expr))


# -- 1. nargout forwarding ----------------------------------------------------------

WRAPPER = """
function varargout = w(f, varargin)
    [varargout{1:nargout}] = f(varargin{:});
end
"""


def test_varargout_forwards_nargout(m):
    m.run(WRAPPER)
    m.run("[a, b] = w(@max, [4 9 2]);")
    assert m.num("a").item() == 9 and m.num("b").item() == 2
    m.run("[r, c] = w(@size, ones(2, 3));")
    assert m.num("r").item() == 2 and m.num("c").item() == 3
    m.run("one = w(@max, [4 9 2]);")
    assert m.num("one").item() == 9
    m.run("z = w(@() 42);")
    assert m.num("z").item() == 42


def test_wrapper_with_no_outputs_requested(m):
    m.run(WRAPPER)
    assert "hello" in m.run("w(@disp, 'hello');")  # nargout is 0: disp is called with no outputs


def test_cell_list_assignment_creates_the_cell(m):
    m.run("[q{1:2}] = deal('p', 'q');")
    assert cls(m, "q") == "cell"
    assert m.num("size(q)").tolist() == [[1, 2]]
    assert m.str("q{2}") == "q"
    m.run("c = cell(2, 2); [c{:, 1}] = deal(1, 2);")
    assert m.num("c{2, 1}").item() == 2
    m.run("g = {7}; [g{2:3}] = deal(8, 9);")  # grows an existing cell
    assert m.num("[g{:}]").tolist() == [[7, 8, 9]]


# -- 2. 64-bit integers -------------------------------------------------------------

def test_int64_values_do_not_go_through_double(m):
    assert int(m.num("uint64(18446744073709551615)").item()) == 2 ** 64 - 1
    assert m.num("uint64(18446744073709551615) == intmax('uint64')").item()
    assert m.num("intmax('uint64') + 1 == intmax('uint64')").item()
    assert int(m.num("int64(9.3e18)").item()) == 2 ** 63 - 1
    assert int(m.num("int64(-9.3e18)").item()) == -2 ** 63
    assert int(m.num("int64(9007199254740992) + int64(1) - int64(9007199254740992)").item()) == 1
    assert int(m.num("intmax('int64') - int64(1)").item()) == 2 ** 63 - 2
    assert int(m.num("intmax('uint64') - 1").item()) == 2 ** 64 - 2
    assert int(m.num("-intmin('int64')").item()) == 2 ** 63 - 1
    assert int(m.num("intmin('int64') - 1").item()) == -2 ** 63
    assert cls(m, "int64(1) + 1") == "int64"


def test_int64_literals_are_exact(m):
    assert int(m.num("int64(9223372036854775806)").item()) == 9223372036854775806
    assert int(m.num("int64(-9223372036854775807)").item()) == -9223372036854775807
    assert int(m.num("uint64(18446744073709551614)").item()) == 18446744073709551614
    assert int(m.num("int64(9007199254740993)").item()) == 9007199254740993
    assert not m.num("int64(9007199254740993) == 9007199254740992").item()


def test_int64_arithmetic_rounds_and_converts(m):
    assert int(m.num("int64(7) / int64(2)").item()) == 4
    assert int(m.num("int64(5) * 0.5").item()) == 3
    assert int(m.num("uint64(18446744073709551615) / 2").item()) == 2 ** 63
    assert int(m.num("uint64(int64(-5))").item()) == 0
    assert int(m.num("int64(intmax('uint64'))").item()) == 2 ** 63 - 1
    assert int(m.num("int32(intmax('int64'))").item()) == 2 ** 31 - 1
    assert int(m.num("int64(2^52 + 1)").item()) == 2 ** 52 + 1


def test_sprintf_prints_all_int64_digits(m):
    assert m.str("sprintf('%d', intmax('int64'))") == "9223372036854775807"
    assert m.str("sprintf('%d', intmax('uint64'))") == "18446744073709551615"


# -- 3. single propagation ----------------------------------------------------------

def test_single_and_double_compare_in_single(m):
    assert m.num("single(0.1) == 0.1").item()
    assert not m.num("single(0.1) < 0.1").item()
    assert not m.num("double(single(0.1)) == 0.1").item()


def test_single_class_propagates(m):
    assert cls(m, "single(1):3") == "single"
    assert cls(m, "1:single(3)") == "single"
    assert cls(m, "single(1) + 1") == "single"
    assert cls(m, "single(2) .^ 2") == "single"
    assert cls(m, "eps('single')") == "single"
    assert cls(m, "realmax('single')") == "single"
    assert cls(m, "realmin('single')") == "single"
    assert cls(m, "eps(single(1))") == "single"
    assert cls(m, "linspace(single(0), 1, 3)") == "single"
    assert m.num("eps('single')").item() == np.finfo(np.float32).eps
    assert m.num("single(0.1) + 0.1").item() == np.float32(0.1) + np.float32(0.1)
    assert cls(m, "realmax") == "double"


# -- 4. concatenation with empty operands -------------------------------------------

@pytest.mark.parametrize("expr, expected", [
    ("[zeros(1, 0), true]", "double"),
    ("[true, zeros(1, 0)]", "double"),
    ("[[] true]", "double"),
    ("[int8([]), 5.5]", "int8"),
    ("[single([]), 5.5]", "single"),
    ("[true(0, 0), true]", "logical"),
    ("['', 65]", "char"),
    ("[[], []]", "double"),
    ("[int8([]); zeros(0, 3)]", "int8"),
])
def test_concat_class_counts_empty_operands(m, expr, expected):
    assert cls(m, expr) == expected


def test_concat_value_with_integer_empty(m):
    assert m.num("[int8([]), 5.5]").tolist() == [[6]]


# -- 5. assignment and deletion shapes ----------------------------------------------

def test_assignment_shape_mismatch(m):
    m.run("G = zeros(2, 3);")
    with pytest.raises(MatlabError) as e:
        m.run("G(1:2, 1:2) = [1 2 3 4];")
    assert e.value.identifier == "MATLAB:subsassigndimmismatch"
    with pytest.raises(MatlabError) as e:
        m.run("G(1:2, 1) = [1 2 3];")
    assert e.value.identifier == "MATLAB:subsassigndimmismatch"
    with pytest.raises(MatlabError) as e:
        m.run("G(1:2) = [1 2 3];")
    assert e.value.identifier == "MATLAB:subsassignnumelmismatch"
    with pytest.raises(MatlabError) as e:
        m.run("G(:) = [1 2 3];")
    assert e.value.identifier == "MATLAB:subsassignnumelmismatch"


def test_assignment_ignores_singleton_dimensions(m):
    m.run("G = zeros(2, 3); G(1, 1:3) = [1; 2; 3]; G(:, 2) = [7 8];")
    assert m.num("G").tolist() == [[1, 7, 3], [0, 8, 0]]
    m.run("H = zeros(2, 2, 2); H(:, :, 2) = [1 2; 3 4]; H(1, 2, :) = [5 6];")
    assert m.num("H(:, :, 2)").tolist() == [[1, 6], [3, 4]]


def test_delete_all_columns_or_rows(m):
    m.run("d = ones(2, 3); d(:, 1:3) = [];")
    assert m.num("size(d)").tolist() == [[2, 0]]
    m.run("d = ones(2, 3); d(1:2, :) = [];")
    assert m.num("size(d)").tolist() == [[0, 3]]
    m.run("d = ones(2, 3); d(:, :) = [];")
    assert m.num("size(d)").tolist() == [[0, 3]]
    m.run("d = ones(2, 3); d(:, 2) = [];")
    assert m.num("size(d)").tolist() == [[2, 2]]


def test_empty_column_grows_as_a_column(m):
    m.run("g = zeros(0, 1); g(2) = 4;")
    assert m.num("g").tolist() == [[0], [4]]
    m.run("h = zeros(1, 0); h(2) = 4; k = []; k(3) = 1;")
    assert m.num("size(h)").tolist() == [[1, 2]]
    assert m.num("size(k)").tolist() == [[1, 3]]


# -- 6. hexadecimal and binary literals ---------------------------------------------

@pytest.mark.parametrize("expr, klass, value", [
    ("0x1F", "uint8", 31),
    ("0b101", "uint8", 5),
    ("0xFF", "uint8", 255),
    ("0x100", "uint16", 256),
    ("0x1FFFF", "uint32", 131071),
    ("0x1FFFFFFFF", "uint64", 8589934591),
    ("0xFFFFFFFFFFFFFFFF", "uint64", 2 ** 64 - 1),
    ("0x7Fs8", "int8", 127),
    ("0xFFs8", "int8", -1),
    ("0xFFu16", "uint16", 255),
    ("0b11s32", "int32", 3),
    ("0X1f", "uint8", 31),
])
def test_hex_and_binary_literals(m, expr, klass, value):
    v = m.interp.eval_expression(expr)
    assert class_name(v) == klass
    assert int(v.item()) == value


def test_hex_literal_in_expressions(m):
    assert m.num("0xFF + 0").item() == 255 and cls(m, "0xFF + 1") == "uint8"
    assert m.num("[0x01 0x02]").tolist() == [[1, 2]]
    m.run("f = @() 0x1F;")
    assert "0x1F" in m.str("func2str(f)")
    m.run("x0 = 3; y = x0 * 2;")  # identifiers that merely contain 0x are not literals
    assert m.num("y").item() == 6


# -- 7. loop variable after an empty loop -------------------------------------------

def test_loop_variable_after_empty_range(m):
    m.run("for e1 = 1:0, end")
    assert m.num("size(e1)").tolist() == [[1, 0]]
    m.run("e3 = 5; for e3 = [], end")
    assert m.num("size(e3)").tolist() == [[0, 0]]


# -- 8. nargin / nargout of a function ----------------------------------------------

def test_nargin_nargout_of_functions(m):
    m.run("""
function [s, p, d] = multi(a, b)
    s = a + b; p = a * b; d = a - b;
end
function varargout = vfun(a, varargin)
    varargout = {a};
end
function noout(a, b, c)
end
""")
    assert m.num("nargin('multi')").item() == 2
    assert m.num("nargout('multi')").item() == 3
    assert m.num("nargin('vfun')").item() == -2
    assert m.num("nargout('vfun')").item() == -1
    assert m.num("nargin(@noout)").item() == 3
    assert m.num("nargout(@noout)").item() == 0
    m.run("g = @(x, y) x + y; k = @(varargin) numel(varargin); h = @(a, varargin) a;")
    assert m.num("nargin(g)").item() == 2
    assert m.num("nargin(k)").item() == -1
    assert m.num("nargin(h)").item() == -2
    assert m.num("nargout(g)").item() == -1
    with pytest.raises(MatlabError):
        m.run("nargin('no_such_function_xyz')")


# -- 9. mixed integer classes -------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "int8(1) + int16(1)",
    "uint8(1) .* int8(1)",
    "min(int8(1), int16(2))",
    "max(uint8(1), int8(2))",
    "int8(1):int16(3)",
])
def test_mixed_integer_classes_are_an_error(m, expr):
    with pytest.raises(MatlabError, match="(?i)integers can only be combined|Colon operands"):
        m.run(f"x = {expr};")


def test_integers_still_combine_with_double_char_logical(m):
    assert cls(m, "int8(1) + 1") == "int8"
    assert cls(m, "true + int8(1)") == "int8"
    assert cls(m, "'a' + int8(1)") == "int8"
    assert cls(m, "int8([1 2]) + [1 2]") == "int8"
    assert m.num("5 == int8(5)").item()
    assert m.num("max(int8(3), 7)").item() == 7
    assert m.num("uint8(200) + 100").item() == 255


# -- 10. negative zero and rounding -------------------------------------------------

def test_negative_zero_from_rounding(m):
    assert m.num("1 / ceil(-0.5)").item() == -np.inf
    assert m.num("1 / fix(-0.5)").item() == -np.inf
    assert m.num("1 / round(-0.2)").item() == -np.inf
    assert m.num("1 ./ ceil([-0.5 0.5])").tolist() == [[-np.inf, 1.0]]
    assert m.num("1 / floor(0.5)").item() == np.inf


def test_round_halves(m):
    assert m.num("round(0.49999999999999994)").item() == 0
    assert m.num("round([2.5 -2.5 0.5 -0.5 1.4])").tolist() == [[3, -3, 1, -1, 1]]
    assert m.num("round(4503599627370497)").item() == 4503599627370497
    assert m.num("round(2.567, 2)").item() == pytest.approx(2.57)


# -- 11. assorted built-in forms ----------------------------------------------------

def test_isequal_handles_and_isequaln_structs(m):
    assert m.num("isequal(@sin, @sin)").item()
    assert not m.num("isequal(@sin, @cos)").item()
    m.run("f = @(x) x; g = f;")
    assert m.num("isequal(f, g)").item()
    assert m.num("isequaln(struct('a', NaN), struct('a', NaN))").item()
    assert not m.num("isequal(struct('a', NaN), struct('a', NaN))").item()
    assert m.num("isequaln({1, NaN}, {1, NaN})").item()
    assert not m.num("isequaln({1, NaN}, {1, 2})").item()


def test_who_lists_declared_globals(m):
    m.run("global gg_shared; gg_shared = 3; local_v = 1; names = who;")
    names = [("".join(x[0]) if x.size else "") for x in m.val("names").data.flat]
    assert "gg_shared" in names and "local_v" in names


def test_size_forms(m):
    m.run("A = zeros(2, 3, 4);")
    assert m.num("size(A, [1 3])").tolist() == [[2, 4]]
    assert m.num("size(A, 1, 2)").tolist() == [[2, 3]]
    m.run("[r, c] = size(A, [1 3]); [p, q, s] = size(A, 3, 2, 5);")
    assert (m.num("r").item(), m.num("c").item()) == (2, 4)
    assert (m.num("p").item(), m.num("q").item(), m.num("s").item()) == (4, 3, 1)
    m.run("[a, b] = size(A);")
    assert (m.num("a").item(), m.num("b").item()) == (2, 12)


def test_cast_forms(m):
    assert m.str("cast(65, 'char')") == "A"
    assert cls(m, "cast(5, 'like', int16(1))") == "int16"
    assert cls(m, "cast(5, 'like', single(1))") == "single"
    assert cls(m, "cast(300, 'uint8')") == "uint8" and m.num("cast(300, 'uint8')").item() == 255


def test_struct_empty_repmat_negative_reshape_two_unknowns(m):
    m.run("s = struct([]);")
    assert cls(m, "s") == "struct" and m.num("size(s)").tolist() == [[0, 0]]
    assert m.num("isempty(fieldnames(s))").item()
    assert m.num("size(repmat([1 2], -1, 2))").tolist() == [[0, 4]]
    assert m.num("size(repmat(5, 2, -3))").tolist() == [[2, 0]]
    with pytest.raises(MatlabError):
        m.run("reshape(1:6, [], [])")
    assert m.num("size(reshape(1:6, [], 2))").tolist() == [[3, 2]]


# -- 12. other conformance items ----------------------------------------------------

def test_map_count_is_uint64(m):
    m.run("mp = containers.Map(); mp('a') = 1; mp('b') = 2;")
    assert cls(m, "mp.Count") == "uint64" and m.num("mp.Count").item() == 2
    assert cls(m, "length(mp)") == "double" and m.num("length(mp)").item() == 2
    assert m.num("mp.Count + 1").item() == 3


def test_switch_on_nonscalar_number_is_an_error(m):
    with pytest.raises(MatlabError, match="SWITCH expression"):
        m.run("switch [1 2], case 1, x = 1; otherwise, x = 2; end")
    m.run("switch 'ab', case 'ab', x = 1; otherwise, x = 2; end")
    assert m.num("x").item() == 1


def test_lasterr(m):
    m.run("try, error('L:e', 'last one'); catch, msg = lasterr; [m2, id2] = lasterr; end")
    assert m.str("msg") == "last one" and m.str("id2") == "L:e"


def test_small_predicates_and_forms(m):
    assert m.num("isvector(zeros(1, 0))").item() and m.num("isvector(zeros(0, 1))").item()
    assert not m.num("isvector(zeros(0, 0))").item()
    assert not m.num("isreal({1})").item() and not m.num("isreal(@sin)").item()
    assert m.num("isreal('a')").item() and not m.num("isreal(1i)").item()
    assert m.num("linspace(0, 1, 2.5)").tolist() == [[0, 1]]
    m.run("c = char({'ab', 'c'}, 'xyz');")
    assert m.num("size(c)").tolist() == [[3, 3]]
    assert m.str("c(3, :)") == "xyz"


def test_integer_with_single_and_mixed_comparisons_are_allowed():
    from mylab.api import Session
    r = Session().run("a = int8(5)*single(2.4); disp(class(a)); disp(a); disp(int8(1) == int16(1));"
                      "disp(uint64(18446744073709551615) > int64(-1))")
    assert r.error is None and r.output.split() == ["int8", "12", "1", "1"]
