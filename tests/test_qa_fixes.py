"""Regression tests for issues found in pre-release QA."""

import pytest

from mylab import MatlabError


def test_integer_rounding_half_away(m):
    assert m.num("uint8(2.5)").item() == 3
    assert m.num("int8(-2.5)").item() == -3
    assert m.num("int32(10) / 4").item() == 3
    assert m.num("int32(5) / int32(2)").item() == 3
    assert m.num("int16([1.5 2.5 3.5])").tolist() == [[2, 3, 4]]


def test_anonymous_function_returning_char(m):
    m.run("f = @() 'hello'; g = @(x) 'z';")
    assert m.str("f()") == "hello"
    m.run("c = cellfun(@(x) 'z', {1, 2}, 'UniformOutput', false);")
    assert m.str("c{2}") == "z"
    m.run("h = @(x) x'; v = h([1 2 3]);")  # transpose inside a body still works
    assert m.num("size(v)").tolist() == [[3, 1]]


def test_power_and_transpose_left_to_right(m):
    m.run("n = 1:3; a = n.^2'; b = n'.^2;")
    assert m.num("size(a)").tolist() == [[3, 1]]
    assert m.num("size(b)").tolist() == [[3, 1]]
    assert m.num("2^3'").item() == 8


def test_logical_stays_logical(m):
    m.run("x = [5 6 7]; mask = true(size(x)); mask(2) = 0; y = x(mask);")
    assert m.num("y").tolist() == [[5, 7]]
    assert m.str("class(mask)") == "logical"
    m.run("q = false(1, 3); q([1 3]) = 1;")
    assert m.str("class(q)") == "logical"


def test_command_syntax_for_any_function(m):
    from mylab.parser import parse
    body = parse("shading interp\ncolormap jet\nlegend off\nclearvars -except y\n").body
    assert len(body) == 4
    m.run("a = 5; b = a -1;")  # expressions are not commands
    assert m.val("b").item() == 4
    m.run("x = 3; y = 4; clearvars -except y")
    assert not m.interp.base.has("x") and m.interp.base.has("y")


def test_deep_recursion(m):
    m.run("function s = f(n)\nif n == 0\ns = 0;\nelse\ns = n + f(n - 1);\nend\nend\nr = f(400);")
    assert m.val("r").item() == 400 * 401 / 2
    with pytest.raises(MatlabError, match="recursion"):
        m.run("function g(n)\ng(n + 1);\nend\ng(1)")


def test_comma_list_assignment(m):
    m.run("s = struct('v', {1, 2}); [s.v] = deal(7, 8);")
    assert m.num("[s.v]").tolist() == [[7, 8]]
    m.run("p = struct('age', {1, 2, 3}); [p.age] = deal(0);")
    assert m.num("[p.age]").tolist() == [[0, 0, 0]]
    m.run("c = cell(1, 3); [c{:}] = deal(5);")
    assert m.num("[c{:}]").tolist() == [[5, 5, 5]]
    m.run("[q, w] = deal(1, 2);")
    assert m.val("w").item() == 2


def test_not_enough_inputs_message(m):
    with pytest.raises(MatlabError, match="Not enough input arguments"):
        m.run("function r = add2(a, b)\nr = a + b;\nend\nadd2(1)")
