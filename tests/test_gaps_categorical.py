"""categorical arrays."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def test_construct_categories_counts(m):
    m.run("c = categorical({'red', 'blue', 'red', '', 'green'}); cats = categories(c); n = countcats(c);"
          "u = isundefined(c); d = double(c); k = numel(c); cls = class(c);")
    assert [m.str(f"cats{{{i}}}") for i in (1, 2, 3)] == ["blue", "green", "red"]
    np.testing.assert_array_equal(m.val("n"), [[1, 1, 2]])
    np.testing.assert_array_equal(m.val("u"), [[False, False, False, True, False]])
    np.testing.assert_array_equal(m.val("d"), [[3, 1, 3, np.nan, 2]])
    assert m.num("k") == 5 and m.str("cls") == "categorical" and m.num("iscategorical(c)")
    assert not m.num("iscategorical(5)")


def test_construct_from_numbers_valueset_and_names(m):
    m.run("a = categorical([3 1 3 2]); ca = categories(a);"
          "b = categorical([1 2 3 2 9], [1 2 3], {'low', 'mid', 'high'}); cb = categories(b);"
          "s = categorical([\"x\" \"y\" \"x\"]); l = categorical([true false]);"
          "o = categorical({'S', 'L', 'M'}, {'S', 'M', 'L'}, 'Ordinal', true);")
    assert m.str("ca{1}") == "1" and m.str("ca{3}") == "3"
    assert m.str("cb{3}") == "high" and m.num("isundefined(b(5))") and m.str("char(b(2))") == "mid"
    assert m.num("numel(categories(s))") == 2 and m.str("char(l(1))") == "true"
    np.testing.assert_array_equal(m.num("o > 'S'"), [[False, True, True]])
    np.testing.assert_array_equal(m.num("o(1) < o"), [[False, True, True]])
    assert m.num("isordinal(o)")
    with pytest.raises(MatlabError, match="ordinal"):
        m.run("categorical({'a'}) < 'b'")


def test_compare_with_text_and_ismember(m):
    m.run("c = categorical({'red'; 'blue'; 'red'}); e = c == 'red'; ne = c ~= \"red\"; "
          "e2 = c == categorical({'red'; 'red'; 'blue'}); mm = ismember(c, {'blue', 'green'});"
          "k = sum(c == 'red'); f = find(c == 'blue'); s = cellstr(c); st = string(c);")
    np.testing.assert_array_equal(m.val("e"), [[True], [False], [True]])
    np.testing.assert_array_equal(m.val("ne"), [[False], [True], [False]])
    np.testing.assert_array_equal(m.val("e2"), [[True], [False], [False]])
    np.testing.assert_array_equal(m.val("mm"), [[False], [True], [False]])
    assert m.num("k") == 2 and m.num("f") == 2 and m.str("s{2}") == "blue" and m.num("st(3) == \"red\"")
    m.run("r = 0; switch char(c(2)), case 'blue', r = 1; end")
    assert m.num("r") == 1


def test_index_assign_concat_unique_sort(m):
    m.run("c = categorical({'b', 'a', 'c'}); c(2) = 'c'; c(5) = 'new'; cats = categories(c);"
          "d = [c, categorical({'z'})]; cd = categories(d); c(1) = [];"
          "[u, ia, ic] = unique(categorical({'b', 'a', 'b', 'c'}));"
          "[s, order] = sort(categorical({'b', 'a', 'c', 'a'})); t = c';")
    assert m.num("numel(cats)") == 4 and m.str("cats{4}") == "new"
    assert m.num("isundefined(c(3))") and m.str("char(c(1))") == "c"
    assert m.num("numel(d)") == 6 and m.str("cd{5}") == "z"
    assert m.str("char(u(1))") == "a" and m.num("numel(u)") == 3
    np.testing.assert_array_equal(m.val("ic").T, [[2, 1, 2, 3]])
    np.testing.assert_array_equal(m.val("order"), [[2, 4, 1, 3]])
    assert m.str("char(s(4))") == "c"
    np.testing.assert_array_equal(m.num("size(t)"), [[4, 1]])
    assert m.num("isequal(categorical({'a'}), categorical({'a'}))")


def test_category_editing(m):
    m.run("c = categorical({'a', 'b', 'a', 'c'});"
          "r = renamecats(c, 'a', 'alpha'); x = removecats(c, 'b'); m2 = mergecats(c, {'a', 'b'}, 'ab');"
          "ad = addcats(c, {'z'}); ro = reordercats(c, {'c', 'b', 'a'}); un = removecats(c(1:2));")
    assert m.str("char(r(1))") == "alpha" and m.num("isundefined(x(2))")
    np.testing.assert_array_equal(m.num("countcats(m2)"), [[3, 1]])
    assert m.num("numel(categories(ad))") == 4
    np.testing.assert_array_equal(m.num("double(ro)"), [[3, 2, 3, 1]])
    assert m.num("numel(categories(un))") == 2
    with pytest.raises(MatlabError, match="is not a category"):
        m.run("renamecats(c, 'nope', 'x')")
    with pytest.raises(MatlabError, match="needs a categorical array"):
        m.run("categories(5)")


def test_display(m):
    out = m.run("c = categorical({'red', 'blue'; 'x', ''})")
    assert "2×2 categorical array" in out and "<undefined>" in out and "red" in out
    assert "categorical" in m.run("c = categorical({'a'})")
    out = m.run("summary(categorical({'a', 'b', 'a'}))")
    assert out.split() == ["a", "b", "2", "1"]


def test_categorical_in_tables_and_grouping(m):
    m.run("g = categorical({'x'; 'y'; 'x'; 'y'; 'x'}, {'y', 'x'}); v = [1; 2; 3; 4; 5];"
          "T = table(g, v); S = groupsummary(T, 'g', 'mean'); gn = cellstr(S.g);"
          "[idx, names] = grp2idx(g); G = findgroups(g); m1 = splitapply(@sum, v, G);"
          "T2 = sortrows(T, 'g'); sub = T(T.g == 'x', :); T.g(2) = 'x';")
    assert m.str("gn{1}") == "y" and m.str("gn{2}") == "x"   # groups follow the category order
    np.testing.assert_array_equal(m.num("S.GroupCount"), [[2], [3]])
    np.testing.assert_array_equal(m.num("S.mean_v"), [[3], [3]])
    np.testing.assert_array_equal(m.val("idx"), [[2], [1], [2], [1], [2]])
    assert m.str("names{1}") == "y"
    np.testing.assert_array_equal(m.val("m1"), [[6], [9]])
    np.testing.assert_array_equal(m.num("T2.v"), [[2], [4], [1], [3], [5]])
    assert m.num("height(sub)") == 3 and m.num("sum(T.g == 'x')") == 4
    out = m.run("T")
    assert "x" in out and "g" in out


def test_categorical_plot_and_loops(m):
    m.run("c = categorical({'a', 'b', 'a'}); n = 0; for k = 1:numel(c), if c(k) == 'a', n = n + 1; end, end; "
          "figure; histogram(c); close all; counts = histcounts(c);")
    assert m.num("n") == 2
    np.testing.assert_array_equal(m.val("counts"), [[2, 1]])
