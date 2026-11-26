"""Language gaps found by running public MATLAB code unchanged: [] with cell arrays, num2cell for
every class, the old rand('state') syntaxes, round's TieBreaker, empty sprintf results and complex
max/min."""

import numpy as np
import pytest

from mylab.errors import MatlabError
from mylab.values import Cell, Struct


# -- [] and other empty arrays concatenated with cell arrays -----------------------------------------

def test_empty_double_adds_nothing_to_a_cell_concatenation(m):
    m.run("x = []; x = [x; {}']; a = [[], {1}]; b = [{1}, []]; c = [{}, {1}];")
    for name in ("a", "b", "c"):
        v = m.val(name)
        assert isinstance(v, Cell) and v.shape == (1, 1)
        assert float(v.data[0, 0][0, 0]) == 1
    x = m.val("x")
    assert isinstance(x, Cell) and x.shape == (0, 0)


def test_growing_a_list_of_names_from_an_empty_double(m):
    m.run("""
files = [];
files = [files; {}'];
files = [files; {'a.png'; 'b.png'}];
names = {'c.png', 'd.png'};
files = [files; names'];
""")
    files = m.val("files")
    assert isinstance(files, Cell) and files.shape == (4, 1)
    assert [m.str(f"files{{{k}}}") for k in range(1, 5)] == ["a.png", "b.png", "c.png", "d.png"]
    assert m.num("iscellstr(files)")


def test_empty_arrays_of_any_size_and_class_are_dropped_next_to_a_cell(m):
    m.run("e = [zeros(1,0); {1}]; f = [{1}; zeros(0,3)]; g = [{1}, '']; h = [{1}, struct([])]; "
          "k = [{}, []]; n = horzcat({1}, []); p = cat(1, [], {2}); q = [{1}, false(0,1)];")
    for name in ("e", "f", "g", "h", "n", "p", "q"):
        v = m.val(name)
        assert isinstance(v, Cell) and v.shape == (1, 1), name
    k = m.val("k")
    assert isinstance(k, Cell) and k.shape == (0, 0)


def test_a_non_empty_value_next_to_a_cell_is_wrapped(m):
    m.run("d = [{1}, 2]; s = [{1}; 'abc']; t = [{1}, struct('a', 3)]; z = [{1}, 0];")
    d = m.val("d")
    assert d.shape == (1, 2) and float(d.data[0, 1][0, 0]) == 2
    assert m.val("s").shape == (2, 1) and m.str("s{2}") == "abc"
    assert isinstance(m.val("t").data[0, 1], Struct)
    assert m.val("z").shape == (1, 2)  # a non-empty zero is a value, not an empty array


def test_struct_grown_from_an_empty_double(m):
    m.run("s = []; s = [s, struct('a', 1)]; s = [s, struct('a', 2)]; t = [struct('a', 5); []];")
    s = m.val("s")
    assert isinstance(s, Struct) and s.shape == (1, 2)
    assert m.num("[s.a]").tolist() == [[1.0, 2.0]]
    assert m.val("t").shape == (1, 1)


def test_mismatched_cell_rows_still_fail(m):
    with pytest.raises(MatlabError, match="not consistent"):
        m.run("v = [{1}; {2}, {3}];")


# -- num2cell -----------------------------------------------------------------------------------------

def test_num2cell_of_a_struct_array(m):
    m.run("s = struct('a', {1, 2, 3}); c = num2cell(s); c{2}.a = 20;")
    c = m.val("c")
    assert isinstance(c, Cell) and c.shape == (1, 3)
    assert all(isinstance(x, Struct) and x.shape == (1, 1) for x in c.data.flat)
    assert m.num("c{2}.a") == 20 and m.num("s(2).a") == 2  # each cell holds a copy
    m.run("t = num2cell(reshape(struct('a', {1, 2, 3, 4}), 2, 2), 1);")
    t = m.val("t")
    assert t.shape == (1, 2) and t.data[0, 1].shape == (2, 1)
    assert m.num("[t{2}.a]").tolist() == [[3.0, 4.0]]


def test_num2cell_keeps_the_class_of_each_element(m):
    m.run("l = num2cell([true false]); ch = num2cell('ab'); k = num2cell(int8([1 2 3])); "
          "cc = num2cell({1, 'x'}); g = num2cell([\"a\" \"b\"]); h = num2cell(@sin);")
    assert m.str("class(l{1})") == "logical" and m.val("l").shape == (1, 2)
    assert m.str("ch{2}") == "b" and m.str("class(ch{1})") == "char"
    assert m.str("class(k{3})") == "int8" and m.num("k{3}").item() == 3
    assert m.str("class(cc{2})") == "cell" and m.str("cc{2}{1}") == "x"
    assert m.str("class(g{2})") == "string" and m.str("g{2}") == "b"
    assert m.str("class(h{1})") == "function_handle"


def test_num2cell_with_dimensions(m):
    m.run("A = magic(3); r = num2cell(A, 2); c = num2cell(A, 1); w = num2cell(A, [1 2]); "
          "z = num2cell(zeros(0, 3), 1); t = num2cell(A, 3); s = num2cell([\"a\" \"b\"; \"c\" \"d\"], 1);")
    assert m.val("r").shape == (3, 1) and m.num("r{2}").tolist() == [[3.0, 5.0, 7.0]]
    assert m.val("c").shape == (1, 3) and m.num("c{3}").shape == (3, 1)
    assert m.val("w").shape == (1, 1) and m.num("w{1}").shape == (3, 3)
    assert m.val("z").shape == (1, 3) and m.num("z{1}").shape == (0, 1)
    assert m.val("t").shape == (3, 3)
    assert m.val("s").shape == (1, 2) and m.str("class(s{1})") == "string" and m.num("size(s{1})").tolist() == [[2, 1]]
    with pytest.raises(MatlabError, match="positive integers"):
        m.run("num2cell(A, 0)")


def test_num2cell_of_objects(m, tmp_path):
    (tmp_path / "PtN2C.m").write_text(
        "classdef PtN2C\n    properties\n        v\n    end\n    methods\n"
        "        function o = PtN2C(v)\n            o.v = v;\n        end\n    end\nend\n")
    m.run(f"addpath('{tmp_path.as_posix()}'); arr = [PtN2C(1), PtN2C(2)]; x = num2cell(arr); "
          "y = num2cell([PtN2C(1); PtN2C(2)], 1); c = num2cell(categorical({'a', 'b'})); "
          "d = num2cell(datetime(2020, 1, 1:3));")
    assert m.val("x").shape == (1, 2) and m.str("class(x{2})") == "PtN2C" and m.num("x{2}.v").item() == 2
    assert m.val("y").shape == (1, 1) and m.num("size(y{1})").tolist() == [[2, 1]]
    assert m.str("class(c{2})") == "categorical" and m.str("char(c{2})") == "b"
    assert m.val("d").shape == (1, 3) and m.str("class(d{3})") == "datetime" and m.num("day(d{3})").item() == 3


# -- the old rand('state') / rand('seed') / rand('twister') syntaxes ----------------------------------------

def test_rand_state_query_and_restore(m):
    m.run("rngstate = rand('state'); a = rand(1, 4); rand('state', rngstate); b = rand(1, 4);")
    assert m.num("size(rngstate)").tolist() == [[625, 1]] and m.str("class(rngstate)") == "double"
    np.testing.assert_array_equal(m.num("a"), m.num("b"))
    m.run("t = rand('twister'); a = randn(2); rand('twister', t); b = randn(2);")
    assert m.str("class(t)") == "uint32"
    np.testing.assert_array_equal(m.num("a"), m.num("b"))


def test_rand_seed_query_and_restore(m):
    m.run("s = rand('seed'); a = rand(1, 3); rand('seed', s); b = rand(1, 3);"
          "s2 = randn('seed'); c = randn(1, 3); randn('seed', s2); d = randn(1, 3);"
          "st = randn('state'); e = randn(3, 1); randn('state', st); f = randn(3, 1);")
    assert m.num("size(s)").tolist() == [[1, 1]]
    np.testing.assert_array_equal(m.num("a"), m.num("b"))
    np.testing.assert_array_equal(m.num("c"), m.num("d"))
    np.testing.assert_array_equal(m.num("e"), m.num("f"))


def test_rand_legacy_seeds_match_rng(m):
    m.run("rand('twister', 5489); a = rand(1, 3); rng(5489); b = rand(1, 3); rand('state', 0); c = rand; "
          "rng('default'); d = rand; rand('state', 1:35); e = rand; rand('state', 1:35); f = rand;")
    np.testing.assert_array_equal(m.num("a"), m.num("b"))
    assert m.num("c") == m.num("d") and m.num("e") == m.num("f")
    with pytest.raises(MatlabError):
        m.run("rand('state', 1, 2)")


def test_rand_help_mentions_the_old_syntaxes():
    from mylab.help_data import HELP
    for name in ("rand", "randn"):
        usage, text = HELP[name]
        assert any("'state'" in u for u in usage)
        assert "not those of MATLAB's retired" in text


# -- round with 'significant', 'decimals' and 'TieBreaker' ---------------------------------------------------

TIES = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.4, -2.6]


@pytest.mark.parametrize("tie, expected", [
    ("fromzero", [-3, -2, -1, 1, 2, 3, 2, -3]),
    ("tozero", [-2, -1, 0, 0, 1, 2, 2, -3]),
    ("even", [-2, -2, 0, 0, 2, 2, 2, -3]),
    ("odd", [-3, -1, -1, 1, 1, 3, 2, -3]),
    ("plusinf", [-2, -1, 0, 1, 2, 3, 2, -3]),
    ("minusinf", [-3, -2, -1, 0, 1, 2, 2, -3]),
])
def test_round_tiebreaker(m, tie, expected):
    m.run(f"x = {TIES}; y = round(x, 'TieBreaker', '{tie}');")
    assert m.num("y").ravel().tolist() == expected


def test_round_tiebreaker_with_digits_and_classes(m):
    assert m.num("round(0.125, 2, 'TieBreaker', 'even')") == pytest.approx(0.12)
    assert m.num("round(1250, 2, 'significant', 'TieBreaker', 'even')") == 1200
    assert m.num("round(1250, 2, 'significant')") == 1300
    assert m.num("round(25, -1, 'decimals', 'TieBreaker', 'tozero')") == 20
    m.run("k = round(int8([25 35]), -1, 'TieBreaker', 'even'); j = round(int16(7), 'TieBreaker', 'odd');")
    assert m.str("class(k)") == "int8" and m.num("k").tolist() == [[20, 40]]
    assert m.str("class(j)") == "int16" and m.num("j").item() == 7
    assert m.num("round(2.5 + 3.5i, 'TieBreaker', 'even')").item() == 2 + 4j
    assert m.num("round(5/2 - 2, 'TieBreaker', 'minusinf')") == 0
    assert m.num("round(-0.5, 'tiebreaker', 'EVEN')") == 0


def test_round_rejects_bad_options(m):
    with pytest.raises(MatlabError, match="NoSuchOption"):
        m.run("round(2.5, 'NoSuchOption', 1)")
    with pytest.raises(MatlabError, match="TieBreaker"):
        m.run("round(2.5, 'TieBreaker', 'sideways')")
    with pytest.raises(MatlabError, match="needs a value"):
        m.run("round(2.5, 'TieBreaker')")
    with pytest.raises(MatlabError, match="sometimes"):
        m.run("round(2.5, 1, 'sometimes')")


# -- empty text results are 1-by-0 --------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    "sprintf('%d', [])", "sprintf('%s', '')", "sprintf('')", "sprintf('%s', zeros(1, 0))",
    "strjoin({}, ',')", "strjoin({''}, ', ')", "blanks(0)",
])
def test_empty_character_vector_results_are_1_by_0(m, expr):
    m.run(f"r = {expr};")
    assert m.str("class(r)") == "char"
    assert m.num("size(r)").tolist() == [[1, 0]]


def test_empty_text_results_documented_as_arrays_are_unchanged(m):
    assert m.num("size(num2str([]))").tolist() == [[0, 0]]
    assert m.num("size(sprintf('%d\\n', []))").tolist() == [[1, 1]]
    m.run("c = compose('', 1); s = sprintf(\"\");")
    assert m.num("size(c{1})").tolist() == [[1, 0]]
    assert m.str("class(s)") == "string" and m.num("strlength(s)") == 0


# -- max and min of complex values ------------------------------------------------------------------------

def test_complex_max_with_zero_imaginary_part_is_real(m):
    m.run("a = max([1+2i, 3]); [b, k] = max([1+2i, 3]); c = min([5, 1+2i, 0.5]); d = max(1-2i, 3); "
          "e = max([1+2i, 3; 4, 5], [], 1); f = max([1+2i, 4i]);")
    for name in ("a", "b", "c", "d", "e"):
        v = m.num(name)
        assert v.dtype.kind == "f", name
        assert not np.any(np.signbit(np.imag(v)))
    assert m.num("a").item() == 3 and m.num("k").item() == 2 and m.num("c").item() == 0.5
    assert m.num("e").tolist() == [[4.0, 5.0]]
    assert m.num("f").item() == 4j  # a value with a non-zero imaginary part stays complex
    assert m.num("isreal(a)") and not m.num("isreal(f)")


# -- found on the way: regexprep ${...} in the caller's workspace, fopen with an empty encoding -------------

def test_regexprep_dynamic_expression_sees_the_callers_variables(m):
    out = m.run("""
disp(decode('x_0x41_y'))
function s = decode(name)
    h = @(t) char(hex2dec(t));
    s = regexprep(name, '_0x([0-9A-F]+)_', '${h($1)}');
end
""")
    assert out.strip() == "xAy"


def test_fopen_with_an_empty_encoding_uses_the_default(m, tmp_path):
    path = (tmp_path / "enc.txt").as_posix()
    m.run(f"fid = fopen('{path}', 'w', 'n', ''); fprintf(fid, 'hi'); fclose(fid); t = fileread('{path}');")
    assert m.str("t") == "hi"
