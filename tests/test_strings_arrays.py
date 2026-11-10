"""String arrays ("..." with more than one element)."""

import numpy as np

from mylab.values import MString, StringArray


def test_build_index_assign(m):
    m.run('s = ["red" "green" "blue"];')
    assert isinstance(m.val("s"), StringArray)
    assert m.num("numel(s)").item() == 3
    assert m.str("class(s)") == "string"
    assert isinstance(m.interp.eval_expression("s(2)"), MString)
    assert m.str("s(2)") == "green"
    assert m.str("s{3}") == "blue"
    assert m.str("class(s{3})") == "char"
    m.run('s(5) = "pink";')
    assert m.num("numel(s)").item() == 5
    assert m.num("ismissing(s)").tolist() == [[False, False, False, True, False]]
    m.run("s(4) = [];")
    assert m.num("numel(s)").item() == 4
    m.run('t = strings(2, 2); t(1, 2) = "x";')
    assert m.num("size(t)").tolist() == [[2, 2]]
    m.run('s{1} = "crimson";')
    assert m.str("s(1)") == "crimson"
    m.run('v = []; v(2) = "b";')
    assert m.num("numel(v)").item() == 2


def test_operators(m):
    m.run('s = ["a" "b"];')
    np.testing.assert_array_equal(m.num('s == "b"'), [[False, True]])
    np.testing.assert_array_equal(m.num('s ~= "b"'), [[True, False]])
    m.run('t = s + "!";')
    assert m.str("t(2)") == "b!"
    m.run('u = ["x"; "y"] + ["1" "2"];')
    assert m.num("size(u)").tolist() == [[2, 2]]
    assert m.str("u(2, 1)") == "y1"
    assert m.str('"n = " + 5') == "n = 5"
    m.run('w = s\';')
    assert m.num("size(w)").tolist() == [[2, 1]]


def test_functions(m):
    m.run('s = ["Apple" "kiwi" "fig"];')
    np.testing.assert_array_equal(m.num("strlength(s)"), [[5, 4, 3]])
    assert m.str("upper(s(2))") == "KIWI"
    np.testing.assert_array_equal(m.num('strcmp(s, "fig")'), [[False, False, True]])
    np.testing.assert_array_equal(m.num('contains(s, "i")'), [[False, True, True]])
    assert m.str('strjoin(s, ", ")') == "Apple, kiwi, fig"
    m.run("c = cellstr(s);")
    assert m.str("class(c)") == "cell" and m.str("c{3}") == "fig"
    m.run("z = string({'p', 'q'});")
    assert isinstance(m.val("z"), StringArray)
    m.run("n = string([1 2.5 3]);")
    assert m.str("n(2)") == "2.5"
    np.testing.assert_array_equal(m.num('double(["1.5" "x"])'), [[1.5, np.nan]])
    m.run("srt = sort(s);")
    assert m.str("srt(1)") == "Apple"
    m.run('[uu, ia] = unique(["b" "a" "b"]);')
    assert m.num("numel(uu)").item() == 2
    assert m.num('ismember("kiwi", s)').item()
    assert m.str("sprintf('%s;', s)") == "Apple;kiwi;fig;"
    m.run('parts = strsplit("a b c");')
    assert m.num("numel(parts)").item() == 3 and m.str("class(parts)") == "string"
    m.run("ch = char(s);")
    assert m.num("size(ch)").tolist() == [[3, 5]]
    assert m.num("isstring(s)").item()


def test_display_and_loop(m):
    out = m.run('s = ["a" "bb"]')
    assert "1×2 string array" in out and '"bb"' in out
    out = m.run('for w = ["p" "q"], disp(w), end')
    assert out == "p\nq\n"
    m.run("""
switch "beta"
    case "alpha"
        r = 1;
    case ["beta", "gamma"]
        r = 2;
end
""")
    assert m.val("r").item() == 2
