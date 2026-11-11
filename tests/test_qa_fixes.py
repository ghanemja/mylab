"""Regression tests for issues found in pre-release QA."""

import numpy as np
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


def test_mexception_object(m):
    m.run("try\n  x = [1 2 3]; y = x(5);\ncatch ME\n"
          "  id = ME.identifier; c = class(ME); ok = isa(ME, 'MException');\nend")
    assert m.str("id") == "MATLAB:badsubscript"
    assert m.str("c") == "MException" and m.val("ok").item()
    m.run("try, [1 2] + [1 2 3]; catch e, id2 = e.identifier; end")
    assert m.str("id2") == "MATLAB:sizeDimensionsMustMatch"
    m.run("try, [1 2] * [3 4]; catch e, id3 = e.identifier; end")
    assert m.str("id3") == "MATLAB:innerdim"
    m.run("ME2 = MException('my:id', 'Bad value %d', 7);")
    with pytest.raises(MatlabError, match="Bad value 7"):
        m.run("throw(ME2)")
    m.run("try, error('a:b', 'outer'); catch E, r = E.getReport(); end")
    assert "outer" in m.str("r")
    m.run("try, error('Simple %d'); catch E2, msg = E2.message; end")
    assert m.str("msg") == "Simple %d"
    m.run("try, error('x:y', 'v=%d', 3); catch E3, try, rethrow(E3); catch E4, id4 = E4.identifier; end, end")
    assert m.str("id4") == "x:y"


def test_warning_identifiers(m):
    out = m.run("warning('off', 'My:id'); warning('My:id', 'hidden'); warning('Other:id', 'shown');")
    assert "hidden" not in out and "shown" in out
    out = m.run("warning('off', 'MATLAB:singularMatrix'); x = [1 2; 2 4] \\ [1; 2];")
    assert "singular" not in out
    m.run("warning('Last:one', 'msg %d', 2); [msg, id] = lastwarn;")
    assert m.str("msg") == "msg 2" and m.str("id") == "Last:one"


def test_object_concatenation_is_a_clean_error(m):
    m.run("mp = containers.Map();")
    with pytest.raises(MatlabError, match="cell array"):
        m.run("x = [mp, mp];")


def test_nested_functions_share_variables(m):
    m.run("""
function total = tally(values)
    total = 0;
    for v = values
        add(v);
    end
    function add(x)
        total = total + x;
    end
end
function out = counter_demo()
    count = 0;
    bump(); bump();
    out = count;
    function bump()
        count = count + 1;
    end
end
r = tally([1 2 3]);
c = counter_demo();
""")
    assert m.val("r").item() == 6
    assert m.val("c").item() == 2


def test_functions_without_end_still_work(m):
    m.run("function a = one()\na = 1;\nfunction b = two()\nb = 2;\n")
    m.run("x = one() + two();")
    assert m.val("x").item() == 3


def test_complex_and_format_display(m):
    assert m.run("a = 1.5 + 2i") == "a = 1.5000 + 2.0000i\n"
    assert m.run("b = sqrt(-2)") == "b = 0.0000 + 1.4142i\n"
    assert m.run("c = 3 - 4i") == "c = 3 - 4i\n"
    out = m.run("r = [2.0946; -1.0473+1.1359i]")
    assert "2.0946 + 0.0000i" in out and "-1.0473 + 1.1359i" in out
    m.run("format long")
    assert m.run("pi*1000") == "ans = 3.141592653589793e+03\n"
    assert m.run("pi") == "ans = 3.141592653589793\n"
    m.run("format short g")
    assert "2.5" in m.run("[1e-5 2.5 1000]") and "0.0000" not in m.run("[1e-5 2.5 1000]")
    m.run("format short")
    assert m.run("pi") == "ans = 3.1416\n"


def test_small_numeric_cases(m):
    assert m.num("-int8(-128)").item() == 127
    assert m.str("class(rand(2, 'single'))") == "single"
    assert m.str("class([single(1) 2])") == "single"
    m.run("[fid, msg] = fopen('definitely_missing_file.txt');")
    assert m.val("fid").item() == -1 and "Errno" not in m.str("msg")


def test_persistent_is_per_function(m):
    m.run("""
function n = counter_a()
    persistent k
    if isempty(k), k = 0; end
    k = k + 1; n = k;
end
function n = counter_b()
    persistent k
    if isempty(k), k = 100; end
    k = k + 1; n = k;
end
a1 = counter_a(); a2 = counter_a(); b1 = counter_b(); a3 = counter_a();
""")
    assert (m.val("a1").item(), m.val("a2").item(), m.val("b1").item(), m.val("a3").item()) == (1, 2, 101, 3)
    assert not m.interp.base.has("k")


def test_command_syntax_after_semicolon_and_type_headers(m):
    from mylab.parser import parse
    assert len(parse("surf(peaks(10)); shading interp\n").body) == 2
    assert "3×1 logical array" in m.run("b = [true; true; false]")
    assert "1×3 int32 row vector" in m.run("k = int32([1 2 3])")
    assert m.run("t = true") == "t =\n\n  logical\n\n   1\n\n"
    assert "{0×0 char}" in m.run("c = {''}")


def test_cross_columnwise(m):
    r = m.num("cross([1 2 3; 4 5 6]', [7 8 9; 1 2 3]')")
    np.testing.assert_array_equal(r, np.cross([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]).T)


def test_import_statement_is_accepted(m):
    out = m.run("import matlab.desktop.editor.*\nx = 1;")
    assert out == "" and m.val("x").item() == 1


def test_corpus_syntax_forms(m):
    from mylab.parser import parse
    # command syntax with paths, URLs, numbers and quoted text
    for line in ["addpath ../lib", "addpath ./utils", "web http://example.org/page?x=1 -new",
                 "format long g", "title 'My results'", "mex -largeArrayDims file.c", "im plc 2 2"]:
        body = parse(line + "\n").body
        assert len(body) == 1 and type(body[0].expr).__name__ == "Index", line
    # but these stay expressions
    m.run("a = 5; b = 2; c = a -1; d = a - b; e = a ./ b; f = [a -1];")
    assert (m.val("c").item(), m.val("d").item(), m.val("e").item()) == (4, 3, 2.5)
    assert m.num("f").tolist() == [[5, -1]]
    # Name=Value arguments
    m.run("s = struct(alpha=1, beta='x');")
    assert m.val("s").fields == ["alpha", "beta"]
    # Octave-style block ends
    m.run("function r = twice(x)\n  if x > 0\n    r = 2*x;\n  else\n    r = 0;\n  endif\nendfunction\nq = twice(4);")
    assert m.val("q").item() == 8


def test_arguments_block(m):
    m.run("""
function out = scale_data(x, factor, opts)
    arguments
        x (1,:) double
        factor (1,1) double {mustBePositive} = 2
        opts.Offset (1,1) double = 0
        opts.Label string = "data"
    end
    out = struct('values', x * factor + opts.Offset, 'label', opts.Label);
end
r1 = scale_data([1 2 3]);
r2 = scale_data([1 2 3], 10, Offset=1, Label='custom');
r3 = scale_data(int8([1 2]), 3, 'offset', 5);
""")
    assert m.num("r1.values").tolist() == [[2, 4, 6]]
    assert m.str("r1.label") == "data"
    assert m.num("r2.values").tolist() == [[11, 21, 31]] and m.str("r2.label") == "custom"
    assert m.str("class(r3.values)") == "double" and m.num("r3.values").tolist() == [[8, 11]]
    with pytest.raises(MatlabError, match="positive"):
        m.run("scale_data([1 2], -1)")
    with pytest.raises(MatlabError, match="Invalid name-value argument"):
        m.run("scale_data([1 2], 1, Bogus=3)")
    with pytest.raises(MatlabError, match="position 1"):
        m.run("scale_data([1 2; 3 4])")


def test_empty_cell_concatenation_keeps_shape(m):
    assert m.num("size([cell(1,0), cell(1,0)])").tolist() == [[1, 0]]
    assert m.num("size([cell(0,1); cell(0,1)])").tolist() == [[0, 1]]
    assert m.num("size([cell(1,0), {}])").tolist() == [[1, 0]]
    assert m.num("size([{}, {}])").tolist() == [[0, 0]]


def test_empty_array_concatenation_keeps_dimensions(m):
    assert m.num("size([zeros(0,1) zeros(0,1)])").tolist() == [[0, 2]]
    assert m.num("size([zeros(0,1), zeros(0,3)])").tolist() == [[0, 4]]
    assert m.num("size([zeros(1,0); zeros(1,0)])").tolist() == [[2, 0]]
    assert m.num("size([[] [1 2]])").tolist() == [[1, 2]]
    m.run("x = []; for k = 1:3, x = [x; k k^2]; end")
    assert m.num("size(x)").tolist() == [[3, 2]]
    assert m.str("['' 'ab']") == "ab"


def test_bit_functions_and_table_sort_directions(m):
    assert m.num("bitget(6, 1:3)").tolist() == [[0, 1, 1]]
    assert m.num("bitset(8, 1)").item() == 9 and m.num("bitset(9, 1, 0)").item() == 8
    assert m.num("bitcmp(uint8(5))").item() == 250
    m.run("T = table([2;1;3], {'b';'a';'c'}, 'VariableNames', {'A', 'B'});")
    m.run("S = sortrows(T, {'A'}, {'descend'}); S2 = sortrows(T, -1);")
    assert m.num("S.A").tolist() == [[3], [2], [1]] and m.num("S2.A").tolist() == [[3], [2], [1]]
    m.run("U = table([1;1;2], {'x';'y';'z'}, 'VariableNames', {'Q', 'N'});")
    m.run("V = sortrows(U, {'Q', 'N'}, {'descend', 'ascend'});")
    assert m.str("V.N{1}") == "z" and m.str("V.N{2}") == "x"


def test_value_class_super_constructor_statement(m, tmp_path):
    (tmp_path / "Base1.m").write_text("classdef Base1\n properties, Name = '', end\n methods\n"
                                      "  function obj = Base1(name), if nargin > 0, obj.Name = name; end, end\n"
                                      " end\nend\n")
    (tmp_path / "Sub1.m").write_text("classdef Sub1 < Base1\n properties, R, end\n methods\n"
                                     "  function obj = Sub1(r), obj@Base1('sub'); obj.R = r; end\n end\nend\n")
    m.interp.path.insert(0, str(tmp_path))
    m.run("s = Sub1(3);")
    assert m.str("s.Name") == "sub" and m.num("s.R").item() == 3


def test_empties_dropped_next_to_nonempty(m):
    assert m.num("size([zeros(1,0); 5])").tolist() == [[1, 1]]
    assert m.num("size([zeros(1,0), [1;2]])").tolist() == [[2, 1]]
