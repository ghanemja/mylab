"""Results that two copies of the same built-in used to disagree on (see test_registry_unique.py).

Each test pins the behaviour that was kept when the copies were merged into one implementation.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def test_filter2_complex_and_single(m):
    # filter2 is correlation with the kernel as given (no conjugation); one copy rejected complex data
    v = m.num("filter2([1 1i], magic(4))")
    assert np.iscomplexobj(v) and v[0, 0] == 16 + 2j and v[0, 3] == 13
    assert m.str("class(filter2([1 2], single(magic(3))))") == "single"
    assert m.str("class(filter2(single([1 2]), magic(3)))") == "single"
    assert np.array_equal(m.num("filter2([1 2], single(magic(3)))"), [[10, 13, 6], [13, 19, 7], [22, 13, 2]])


def test_rats_layout_and_overflow(m):
    # the slash stays in a fixed column; a value that needs more room than strlen is an asterisk
    assert m.str("rats(pi)") == "    355/113   "
    assert m.str("strtrim(rats(1e-10))") == "*"
    assert m.str("strtrim(rats(1e10))") == "*"


def test_histcounts2_bins_each_dimension_like_histcounts(m):
    m.run("x = [1 2 2 3 4 5 5.5 0.2]; y = [3 1 2 2 4 1 0.5 2];"
          "[N, xe, ye] = histcounts2(x, y, 'BinWidth', 2); [~, ex1] = histcounts(x, 'BinWidth', 2);"
          "[~, ey1] = histcounts(y, 'BinWidth', 2);")
    assert m.num("isequal(xe, ex1) && isequal(ye, ey1)")
    assert np.array_equal(m.num("ye"), [[0, 2, 4]])
    assert np.array_equal(m.num("N"), [[0, 2], [1, 2], [2, 1]])
    assert m.num("sum(N(:))") == 8  # y = 4 falls in the last bin, which includes its right edge


def test_isoutlier_rejects_complex(m):
    with pytest.raises(MatlabError):
        m.run("isoutlier([1 2 3 100] + 1i)")
    assert np.array_equal(m.num("isoutlier(int8([1 2 3 100]))"), [[False, False, False, True]])


def test_standardize_missing_complex_and_strings(m):
    v = m.num("standardizeMissing([1 2 3] + 1i, 2 + 1i)")
    assert np.isnan(v[0, 1]) and v[0, 0] == 1 + 1j and v[0, 2] == 3 + 1j
    assert m.num("isequal(ismissing(standardizeMissing([\"a\" \"NA\"], \"NA\")), [false true])")
    m.run("c = standardizeMissing({'a', 'NA', 5}, {'NA', 5});")
    assert m.num("isempty(c{2}) && ischar(c{2}) && isequal(c{3}, 5)")


def test_groupcounts_matrix_columns_are_grouping_variables(m):
    m.run("[b, bg, bp] = groupcounts([1 2; 1 2; 2 3]);")
    assert np.array_equal(m.num("b"), [[2], [1]])
    assert m.str("class(bg)") == "cell" and m.num("numel(bg)") == 2
    assert np.array_equal(m.num("bg{1}"), [[1], [2]]) and np.array_equal(m.num("bg{2}"), [[2], [3]])
    assert np.allclose(m.num("bp"), [[200 / 3], [100 / 3]])
    m.run("[b, bg] = groupcounts([1; 5; 7; 12], [0 6 20]);")
    assert np.array_equal(m.num("b"), [[2], [2]])
    assert np.array_equal(m.num("groupcounts(categorical({'x'; 'y'; 'x'}))"), [[2], [1]])


def test_gscatter_several_grouping_variables_and_rgb(m):
    m.run("figure; h = gscatter([1 2 3 4], [1 2 3 4], {[1 1 2 2], {'a', 'b', 'a', 'b'}});")
    assert m.num("numel(h)") == 4
    m.run("h = gscatter([1 2], [3 4], {'u', 'v'}, [1 0 0; 0 0 1]);")
    assert np.allclose(m.num("h(2).Color"), [[0, 0, 1]])


def test_swapbytes_floating_point(m):
    assert m.num("isequal(typecast(swapbytes(single(1)), 'uint8'), uint8([63 128 0 0]))")
    assert m.num("swapbytes(uint16(1))") == 256


def test_num2cell_string_array(m):
    m.run('c = num2cell(["ab" "c"; "d" missing]);')
    assert m.num("isequal(size(c), [2 2])") and m.str("class(c{1, 2})") == "string"
    assert m.num("ismissing(c{2, 2})")


def test_strfind_one_implementation_for_all_inputs(m):
    assert np.array_equal(m.num("strfind('abababa', 'aba')"), [[1, 3, 5]])
    assert np.array_equal(m.num("strfind('aaaa', 'aa', 'Overlaps', false)"), [[1, 3]])
    assert m.str("class(strfind([\"xax\" \"aa\"], 'a'))") == "cell"
    assert np.array_equal(m.num("strfind('a1b22', digitsPattern)"), [[2, 4]])


def test_lasterr_sets_message(m):
    m.run("lasterr('boom', 'my:id'); [msg, id] = lasterr;")
    assert m.str("msg") == "boom" and m.str("id") == "my:id"


def test_must_be_float_accepts_subclass_of_double(m, tmp_path):
    (tmp_path / "Metres.m").write_text(
        "classdef Metres < double\n    methods\n        function o = Metres(v)\n"
        "            o = o@double(v);\n        end\n    end\nend\n")
    m.run(f"addpath('{tmp_path}'); mustBeFloat(Metres(2));")
    with pytest.raises(MatlabError):
        m.run("mustBeFloat(int8(1))")
