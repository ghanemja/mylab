"""Core language behaviour: syntax, operators, indexing, control flow, functions."""

import numpy as np
import pytest

from mylab import MatlabError


def test_arithmetic_and_precedence(m):
    assert m.num("1 + 2 * 3").item() == 7
    assert m.num("-2^2").item() == -4
    assert m.num("2^-1").item() == 0.5
    assert m.num("2^3^2").item() == 64  # left associative
    assert m.num("(1:3)'").shape == (3, 1)
    assert m.num("~0 == 1").item()


def test_matrix_literals_and_whitespace(m):
    np.testing.assert_array_equal(m.num("[1 -2]"), [[1, -2]])
    np.testing.assert_array_equal(m.num("[1 - 2]"), [[-1]])
    np.testing.assert_array_equal(m.num("[1 , 2\n3 4]"), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(m.num("[1 2; 3 4]'"), [[1, 3], [2, 4]])
    m.run("a = [1 2 3];")
    np.testing.assert_array_equal(m.num("[a' a']"), [[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(m.num("[a(1) a(end)]"), [[1, 3]])
    assert m.num("[]").shape == (0, 0)


def test_strings_in_matrices(m):
    assert m.str("['abc' 'def']") == "abcdef"
    assert m.str("['x = ' num2str(5)]") == "x = 5"
    m.run("s = 'it''s';")
    assert m.str("s") == "it's"


def test_matrix_ops(m):
    m.run("A = [1 2; 3 4]; b = [5; 6];")
    np.testing.assert_allclose(m.num("A * b"), [[17], [39]])
    np.testing.assert_allclose(m.num("A .* A"), [[1, 4], [9, 16]])
    np.testing.assert_allclose(m.num("A \\ b"), np.linalg.solve([[1, 2], [3, 4]], [[5], [6]]))
    np.testing.assert_allclose(m.num("A ^ 2"), [[7, 10], [15, 22]])
    np.testing.assert_allclose(m.num("[1 2 3] + [10; 20]"), [[11, 12, 13], [21, 22, 23]])
    with pytest.raises(MatlabError, match="Incorrect dimensions"):
        m.run("[1 2 3] * [4 5 6]")


def test_complex_results(m):
    assert m.num("sqrt(-4)").item() == 2j
    assert m.num("(-8)^(1/3)").item() == pytest.approx(1 + 1.7320508j)
    assert m.num("3 + 4i").item() == 3 + 4j
    assert m.num("abs(3+4i)").item() == 5


def test_ranges(m):
    np.testing.assert_array_equal(m.num("1:5"), [[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(m.num("5:-2:1"), [[5, 3, 1]])
    assert m.num("5:1").shape == (1, 0)
    assert m.num("0:0.1:1").shape == (1, 11)
    assert m.str("'a':'e'") == "abcde"


def test_indexing_is_one_based_and_column_major(m):
    m.run("A = magic(3);")
    assert m.num("A(1, 1)").item() == 8
    assert m.num("A(2)").item() == 3  # column-major linear indexing
    np.testing.assert_array_equal(m.num("A(:, 2)"), [[1], [5], [9]])
    np.testing.assert_array_equal(m.num("A(end, :)"), [[4, 9, 2]])
    np.testing.assert_array_equal(m.num("A(end)"), [[2]])
    np.testing.assert_array_equal(m.num("A([1 3], [1 3])"), [[8, 6], [4, 2]])
    assert m.num("A(:)").shape == (9, 1)
    np.testing.assert_array_equal(m.num("A(A > 6)'"), [[8, 9, 7]])
    with pytest.raises(MatlabError, match="positive integers"):
        m.run("A(0)")
    with pytest.raises(MatlabError, match="exceeds"):
        m.run("A(10)")


def test_indexed_assignment_grows_and_deletes(m):
    m.run("x = []; x(3) = 7;")
    np.testing.assert_array_equal(m.val("x"), [[0, 0, 7]])
    m.run("x(end+1) = 8;")
    np.testing.assert_array_equal(m.val("x"), [[0, 0, 7, 8]])
    m.run("x(2) = [];")
    np.testing.assert_array_equal(m.val("x"), [[0, 7, 8]])
    m.run("M = zeros(2); M(3, 4) = 1;")
    assert m.val("M").shape == (3, 4)
    m.run("M(:, 1) = [];")
    assert m.val("M").shape == (3, 3)
    m.run("v = 1:5; v(v > 2) = 0;")
    np.testing.assert_array_equal(m.val("v"), [[1, 2, 0, 0, 0]])
    m.run("B = zeros(2); B(:) = 1:4;")
    np.testing.assert_array_equal(m.val("B"), [[1, 3], [2, 4]])
    m.run("C = []; C(:, 1) = [1; 2; 3];")
    np.testing.assert_array_equal(m.val("C"), [[1], [2], [3]])


def test_logical_ops(m):
    np.testing.assert_array_equal(m.num("[1 0 2] & [1 1 0]"), [[True, False, False]])
    assert m.num("true || error('not evaluated')").item()
    assert not m.num("false && error('not evaluated')").item()
    assert m.num("xor(true, false)").item()


def test_if_for_while_switch(m):
    out = m.run("""
total = 0;
for k = 1:10
    if mod(k, 2) == 0
        continue
    elseif k > 7
        break
    else
        total = total + k;
    end
end
disp(total)
""")
    assert out.strip() == "16"
    m.run("n = 0; while n < 5, n = n + 1; end")
    assert m.val("n").item() == 5
    m.run("""
s = 'b';
switch s
    case 'a'
        r = 1;
    case {'b', 'c'}
        r = 2;
    otherwise
        r = 3;
end
""")
    assert m.val("r").item() == 2
    m.run("for col = [1 2; 3 4], last = col; end")
    np.testing.assert_array_equal(m.val("last"), [[2], [4]])


def test_try_catch(m):
    m.run("""
try
    error('myid:sub', 'Value %d is bad', 42);
catch err
    msg = err.message;
    id = err.identifier;
end
""")
    assert m.str("msg") == "Value 42 is bad"
    assert m.str("id") == "myid:sub"
    m.run("try, undefined_thing_xyz; catch e, ok = 1; end")
    assert m.val("ok").item() == 1


def test_functions_in_script(m):
    out = m.run("""
[s, p] = sum_and_product(3, 4);
fprintf('%d %d\\n', s, p);
disp(fact(5))

function [s, p] = sum_and_product(a, b)
    s = a + b;
    p = a * b;
end

function r = fact(n)
    if n <= 1
        r = 1;
    else
        r = n * fact(n - 1);
    end
end
""")
    assert out == "7 12\n   120\n"


def test_nargin_varargin_varargout(m):
    m.run("""
function out = f(a, b, varargin)
    if nargin < 2
        b = 10;
    end
    out = a + b + numel(varargin);
end
function varargout = g()
    for k = 1:nargout
        varargout{k} = k;
    end
end
""")
    assert m.num("f(1)").item() == 11
    assert m.num("f(1, 2, 'x', 'y')").item() == 5
    m.run("[a, b, c] = g();")
    assert m.val("c").item() == 3


def test_anonymous_functions_capture(m):
    m.run("k = 3; f = @(x) k * x; k = 100;")
    assert m.num("f(2)").item() == 6
    m.run("g = @(x, y) x + y; h = @sin;")
    assert m.num("g(1, 2)").item() == 3
    assert m.num("h(0)").item() == 0
    m.run("compose = @(f, g) @(x) f(g(x)); sq = compose(@sqrt, @abs);")
    assert m.num("sq(-16)").item() == 4
    m.run("[mx, ix] = feval(@max, [3 9 2]);")
    assert m.val("ix").item() == 2


def test_cells(m):
    m.run("c = {1, 'two', [3 4 5]};")
    assert m.str("c{2}") == "two"
    np.testing.assert_array_equal(m.num("c{3}(2)"), [[4]])
    m.run("c{5} = 'five';")
    assert m.num("numel(c)").item() == 5
    m.run("d = c(1:2);")
    assert m.str("class(d)") == "cell"
    np.testing.assert_array_equal(m.num("[c{[1 3]}]"), [[1, 3, 4, 5]])
    m.run("e = {}; e{end+1} = 'a'; e{end+1} = 'b';")
    assert m.num("numel(e)").item() == 2
    m.run("names = {'x', 'y'}; joined = strjoin(names, '-');")
    assert m.str("joined") == "x-y"


def test_structs(m):
    m.run("s.a = 1; s.b.c = 'deep'; s.v(3) = 9;")
    assert m.str("s.b.c") == "deep"
    np.testing.assert_array_equal(m.num("s.v"), [[0, 0, 9]])
    m.run("t(2).name = 'second'; t(1).name = 'first';")
    assert m.num("numel(t)").item() == 2
    assert m.str("t(2).name") == "second"
    m.run("f = 'a'; dyn = s.(f);")
    assert m.val("dyn").item() == 1
    m.run("p = struct('x', {10, 20});")
    assert m.num("numel(p)").item() == 2
    np.testing.assert_array_equal(m.num("[p.x]"), [[10, 20]])
    assert m.num("isfield(s, 'a')").item()


def test_global_variables(m):
    m.run("""
global counter
counter = 1;
bump();
bump();
function bump()
    global counter
    counter = counter + 1;
end
""")
    assert m.val("counter").item() == 3


def test_display_format(m):
    assert m.run("x = 5") == "x = 5\n"
    assert m.run("y = 3.14159") == "y = 3.1416\n"
    assert m.run("z = [1 2 3]") == "z =\n\n     1     2     3\n\n"
    assert m.run("s = 'hi'") == "s = 'hi'\n"
    assert m.run("x;") == ""
    assert m.run("3 + 4") == "ans = 7\n"
    assert m.run("b = true") == "b =\n\n  logical\n\n   1\n\n"


def test_command_syntax(m):
    m.run("format long")
    assert "3.141592653589793" in m.run("pi")
    m.run("format short")
    assert m.run("pi") == "ans = 3.1416\n"
    m.run("x = 1; clear x")
    assert not m.interp.base.has("x")


def test_comments_and_continuation(m):
    m.run("""
% a comment
x = 1 + ... continued
    2; # also a comment
%{
block comment
y = 999;
%}
""")
    assert m.val("x").item() == 3
    assert not m.interp.base.has("y")


def test_errors_are_readable(m):
    with pytest.raises(MatlabError, match="Unrecognized function or variable 'nope'"):
        m.run("nope + 1")
    with pytest.raises(MatlabError, match="incompatible sizes"):
        m.run("[1 2 3] + [1 2]")


def test_integer_types_saturate(m):
    assert m.num("int8(200)").item() == 127
    assert m.num("uint8(5) - uint8(10)").item() == 0
    assert m.str("class(int16(3) * 2)") == "int16"
    assert m.num("int32(7) / int32(2)").item() == 4  # rounds


def test_end_in_nested_index(m):
    m.run("v = 10:10:100;")
    assert m.num("v(end/2)").item() == 50
    assert m.num("v(min(end, 3))").item() == 30
    np.testing.assert_array_equal(m.num("v(end-2:end)"), [[80, 90, 100]])
