"""JSON, audio (WAV) and Excel (.xlsx) files."""

import numpy as np
import pytest

from mylab.builtins import excel
from mylab.errors import MatlabError


@pytest.fixture
def d(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


# -- JSON -----------------------------------------------------------------------------------------

def test_jsondecode_types(m):
    m.run("""s = jsondecode(['{"name": "Ada", "age": 36, "ok": true, "tags": ["a", "b"], "none": null, '...
        ' "pt": {"x": 1.5}, "m": [[1, 2, 3], [4, 5, 6]], "v": [1, null, 3], "mixed": [1, "two"], '...
        ' "people": [{"n": 1}, {"n": 2}], "flags": [true, false], "my key": 2, "e": []}']);""")
    assert m.str("s.name") == "Ada" and m.num("s.age") == 36 and m.num("s.ok").dtype == bool
    assert m.str("class(s.tags)") == "cell" and m.str("s.tags{2}") == "b"
    assert m.num("s.none").shape == (0, 0) and m.num("s.pt.x") == 1.5
    np.testing.assert_array_equal(m.num("s.m"), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(m.num("s.v"), [[1], [np.nan], [3]])
    assert m.str("class(s.mixed)") == "cell" and m.num("s.people(2).n") == 2
    np.testing.assert_array_equal(m.num("size(s.people)"), [[2, 1]])
    assert m.num("s.flags").dtype == bool and m.num("s.my_key") == 2 and m.num("s.e").size == 0
    assert m.num("jsondecode('42')") == 42 and m.str("jsondecode('\"hi\"')") == "hi"
    with pytest.raises(MatlabError, match="JSON syntax error"):
        m.run("jsondecode('{\"a\": }')")


def test_jsonencode_values(m):
    assert m.str("jsonencode(struct('a', 1, 'b', 'text', 'c', [1 2 3]))") == '{"a":1,"b":"text","c":[1,2,3]}'
    assert m.str("jsonencode([1 2; 3 4])") == "[[1,2],[3,4]]"
    assert m.str("jsonencode({1, 'a', true, []})") == '[1,"a",true,[]]'
    assert m.str("jsonencode([1.5 NaN Inf])") == "[1.5,null,null]"
    assert m.str("jsonencode(struct('a', {1, 2}))") == '[{"a":1},{"a":2}]'
    assert m.str("jsonencode(\"s\")") == '"s"' and m.str("jsonencode(pi)") == "3.14159265358979"
    assert m.str("jsonencode(int8([1 2]))") == "[1,2]"
    assert m.str("jsonencode(containers.Map({'k'}, {5}))") == '{"k":5}'
    pretty = m.str("jsonencode(struct('a', [1 2]), 'PrettyPrint', true)")
    assert pretty == '{\n  "a": [\n    1,\n    2\n  ]\n}'
    m.run("s = struct('x', [1 2 3], 'name', 'n', 'nested', struct('ok', true)); "
          "t = jsondecode(jsonencode(s));")
    assert m.num("isequal(t.x', s.x)") and m.num("t.nested.ok")


# -- audio ----------------------------------------------------------------------------------------

def test_audio_roundtrip(d):
    m = d
    m.run("fs = 8000; t = (0:799)'/fs; y = 0.5*sin(2*pi*440*t); audiowrite('tone.wav', y, fs);"
          "[z, fs2] = audioread('tone.wav'); info = audioinfo('tone.wav');"
          "[part, ~] = audioread('tone.wav', [11 20]); nat = audioread('tone.wav', 'native');")
    assert m.num("fs2") == 8000 and m.val("z").shape == (800, 1)
    np.testing.assert_allclose(m.val("z"), m.val("y"), atol=1 / 32768)
    assert m.num("info.NumChannels") == 1 and m.num("info.TotalSamples") == 800
    assert m.num("info.BitsPerSample") == 16 and abs(m.num("info.Duration") - 0.1) < 1e-12
    np.testing.assert_array_equal(m.val("part"), m.val("z")[10:20])
    assert m.val("nat").dtype == np.int16


def test_audio_stereo_bits_and_errors(d):
    m = d
    m.run("y = [0.25 -0.25; 0.5 -0.5; -1 1]; audiowrite('st.wav', y, 44100, 'BitsPerSample', 24);"
          "[z, fs] = audioread('st.wav'); i24 = audioinfo('st.wav');"
          "audiowrite('f.wav', single(y), 22050, 'BitsPerSample', 32); zf = audioread('f.wav');"
          "audiowrite('u8.wav', y, 8000, 'BitsPerSample', 8); z8 = audioread('u8.wav');"
          "wavwrite(y, 8000, 16, 'old.wav'); [zo, fo, nb] = wavread('old.wav');")
    np.testing.assert_allclose(m.val("z"), m.val("y"), atol=2 ** -22)
    assert m.num("i24.BitsPerSample") == 24 and m.num("i24.NumChannels") == 2
    np.testing.assert_allclose(m.val("zf"), m.val("y"), atol=1e-7)
    np.testing.assert_allclose(m.val("z8"), m.val("y"), atol=1 / 100)
    assert m.num("fo") == 8000 and m.num("nb") == 16
    out = m.run("audiowrite('clip.wav', [2; -2], 8000);")
    assert "clipped" in out
    with pytest.raises(MatlabError, match="not found"):
        m.run("audioread('missing.wav')")
    with pytest.raises(MatlabError, match="Only WAV"):
        m.run("audiowrite('x.mp3', [0; 0], 8000)")


def test_sound_is_a_no_op_with_one_note(m):
    out = m.run("sound([0 0.1 0], 8000); soundsc(rand(10, 1)); p = audioplayer(zeros(10, 1), 8000); "
                "play(p); playblocking(p); stop(p); r = p.SampleRate;")
    assert out.count("no audio device") == 1 and m.num("r") == 8000
    with pytest.raises(MatlabError, match="sample rate"):
        m.run("sound([0 1], 5)")


# -- Excel ----------------------------------------------------------------------------------------

@pytest.fixture(params=["builtin", "openpyxl"])
def xl(request, d, monkeypatch):
    if request.param == "builtin":
        monkeypatch.setattr(excel, "_openpyxl", lambda: None)
    else:
        pytest.importorskip("openpyxl")
    return d


def test_xlswrite_xlsread(xl):
    m = xl
    m.run("xlswrite('a.xlsx', [1 2 3; 4 5 6]); A = xlsread('a.xlsx');"
          "xlswrite('b.xlsx', {'name', 'score'; 'ann', 9.5; 'bob', 7});"
          "[num, txt, raw] = xlsread('b.xlsx');"
          "xlswrite('b.xlsx', [10; 20], 'Extra', 'B2'); B = xlsread('b.xlsx', 'Extra');"
          "[n1, t1] = xlsread('b.xlsx', 1); part = xlsread('a.xlsx', 'B1:C2'); names = sheetnames('b.xlsx');")
    np.testing.assert_array_equal(m.val("A"), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(m.val("num"), [[9.5], [7]])
    assert m.str("txt{1, 2}") == "score" and m.str("txt{3, 1}") == "bob" and m.str("txt{2, 2}") == ""
    assert m.num("raw{2, 2}") == 9.5 and m.str("raw{1, 1}") == "name"
    np.testing.assert_array_equal(m.val("B"), [[10], [20]])
    np.testing.assert_array_equal(m.val("n1"), [[9.5], [7]])  # the first sheet survived the second write
    np.testing.assert_array_equal(m.val("part"), [[2, 3], [5, 6]])
    assert m.num("numel(names)") == 2 and m.num("names(2) == \"Extra\"")


def test_tables_and_matrices_in_excel(xl):
    m = xl
    m.run("T = table([1; 2; 3], {'a'; 'b'; 'c'}, [true; false; true], 'VariableNames', {'id', 'label', 'ok'});"
          "writetable(T, 't.xlsx'); R = readtable('t.xlsx');"
          "writetable(T, 't.xlsx', 'Sheet', 'Second'); R2 = readtable('t.xlsx', 'Sheet', 'Second');"
          "writematrix(magic(3), 'm.xlsx'); M = readmatrix('m.xlsx'); M2 = readmatrix('m.xlsx', 'Range', 'A2:B3');"
          "writecell({1, 'x'; 'y', 2}, 'c.xlsx'); C = readcell('c.xlsx');")
    assert m.val("R").names == ["id", "label", "ok"]
    np.testing.assert_array_equal(m.num("R.id"), [[1], [2], [3]])
    assert m.str("R.label{2}") == "b" and m.num("R.ok").dtype == bool
    assert m.num("height(R2)") == 3
    np.testing.assert_array_equal(m.val("M"), [[8, 1, 6], [3, 5, 7], [4, 9, 2]])
    np.testing.assert_array_equal(m.val("M2"), [[3, 5], [4, 9]])
    assert m.str("C{1, 2}") == "x" and m.num("C{2, 2}") == 2


def test_excel_errors(xl):
    m = xl
    with pytest.raises(MatlabError, match="not found"):
        m.run("xlsread('missing.xlsx')")
    m.run("xlswrite('one.xlsx', 1);")
    with pytest.raises(MatlabError, match="Worksheet 'Nope' was not found"):
        m.run("xlsread('one.xlsx', 'Nope')")
    with pytest.raises(MatlabError, match="not supported"):
        m.run("fid = fopen('old.xls', 'w'); fclose(fid); xlsread('old.xls')")
    with pytest.raises(MatlabError, match="not a cell range"):
        m.run("readmatrix('one.xlsx', 'Range', 'nonsense')")
