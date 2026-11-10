"""The table type."""

import numpy as np
import pytest

from mylab import MatlabError


@pytest.fixture
def t(m):
    m.run("Name = {'Ann'; 'Bob'; 'Cara'}; Age = [31; 45; 27]; Score = [88.5; 92; 79];")
    m.run("T = table(Name, Age, Score);")
    return m


def test_construct_and_access(t):
    m = t
    assert m.str("class(T)") == "table"
    assert m.num("size(T)").tolist() == [[3, 3]]
    assert m.num("height(T)").item() == 3 and m.num("width(T)").item() == 3
    assert m.num("T.Age(2)").item() == 45
    assert m.str("T.Name{3}") == "Cara"
    np.testing.assert_array_equal(m.num("T{:, 'Age'}"), [[31], [45], [27]])
    np.testing.assert_array_equal(m.num("T{2, {'Age', 'Score'}}"), [[45, 92]])
    m.run("names = T.Properties.VariableNames;")
    assert m.str("names{3}") == "Score"
    assert m.num("istable(T)").item()
    m.run("U = table([1;2], [3;4]);")
    assert m.str("U.Properties.VariableNames{2}") == "Var2"


def test_subsets_and_edits(t):
    m = t
    m.run("old = T(T.Age > 30, :);")
    assert m.num("height(old)").item() == 2
    m.run("sub = T(:, {'Name', 'Score'});")
    assert m.num("width(sub)").item() == 2
    m.run("T.Passed = T.Score > 80;")
    assert m.num("width(T)").item() == 4
    m.run("T.Age(1) = 32;")
    assert m.num("T.Age(1)").item() == 32
    m.run("T(2, :) = [];")
    assert m.num("height(T)").item() == 2
    m.run("T = removevars(T, 'Passed');")
    assert m.num("width(T)").item() == 3
    m.run("T.Properties.VariableNames{3} = 'Points';")
    assert m.num("T.Points(2)").item() == 79
    m.run("T = renamevars(T, 'Points', 'Pts');")
    assert m.num("T.Pts(1)").item() == 88.5
    m.run("T(end+1, :) = {'Dan', 50, 70};")
    assert m.num("height(T)").item() == 3 and m.str("T.Name{3}") == "Dan"
    with pytest.raises(MatlabError, match="number of rows"):
        m.run("T.Bad = [1; 2];")


def test_sort_concat_convert(t):
    m = t
    m.run("S = sortrows(T, 'Score', 'descend');")
    assert m.str("S.Name{1}") == "Bob"
    m.run("both = [T; T];")
    assert m.num("height(both)").item() == 6
    m.run("A = table2array(T(:, {'Age', 'Score'}));")
    assert m.num("size(A)").tolist() == [[3, 2]]
    m.run("B = array2table(magic(3), 'VariableNames', {'a', 'b', 'c'});")
    assert m.num("B.c(1)").item() == 6
    m.run("s = table2struct(T);")
    assert m.num("numel(s)").item() == 3 and m.str("s(2).Name") == "Bob"
    m.run("T3 = struct2table(struct('x', {1, 2}, 'y', {'p', 'q'}));")
    assert m.num("height(T3)").item() == 2
    m.run("C = cell2table({'a', 1; 'b', 2}, 'VariableNames', {'k', 'v'});")
    assert m.num("sum(C.v)").item() == 3
    m.run("h = head(T, 2);")
    assert m.num("height(h)").item() == 2


def test_csv_round_trip(t, tmp_path):
    m = t
    path = tmp_path / "t.csv"
    m.run(f"writetable(T, '{path}'); R = readtable('{path}');")
    assert m.num("height(R)").item() == 3
    assert m.str("R.Name{2}") == "Bob"
    assert m.num("R.Score(1)").item() == 88.5


def test_display(t):
    out = t.run("T")
    assert "3×3 table" in out and "Name" in out and "88.5000" in out and "92.0000" in out
