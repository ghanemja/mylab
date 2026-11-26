"""classdef objects (and the other object values) through save and load, in version 7 and 7.3 MAT-files.

Each test saves in one session and loads in a new one, so nothing survives in memory between the
two. The classes are written for these tests.
"""

import gc
import os

import numpy as np
import pytest

from mylab import Interpreter

VERSIONS = ["-v7", "-v7.3"]

CLASSES = {
    "Pt.m": "classdef Pt\n  properties\n    x = 1\n    y = 2\n  end\nend\n",
    "Hd.m": "classdef Hd < handle\n  properties\n    v\n    w\n  end\nend\n",
    "Box.m": "classdef Box\n  properties\n    v\n  end\nend\n",
    "Calc.m": ("classdef Calc\n  properties\n    a = 2\n  end\n  properties (Dependent)\n    twice\n  end\n"
               "  properties (Transient)\n    scratch = 5\n  end\n  properties (Constant)\n    K = 10\n  end\n"
               "  methods\n    function v = get.twice(o)\n      v = 2 * o.a;\n    end\n  end\nend\n"),
    "Animal.m": "classdef Animal\n  properties\n    legs = 4\n  end\nend\n",
    "Dog.m": "classdef Dog < Animal\n  properties\n    name = 'rex'\n  end\nend\n",
    "HAnimal.m": "classdef HAnimal < handle\n  properties\n    legs = 4\n  end\nend\n",
    "HDog.m": "classdef HDog < HAnimal\n  properties\n    name = 'rex'\n  end\nend\n",
    "Mood.m": "classdef Mood\n  enumeration\n    Calm, Busy, Gone\n  end\nend\n",
    "Rank.m": "classdef Rank < int32\n  enumeration\n    Low (1), High (9)\n  end\nend\n",
    "Celsius.m": ("classdef Celsius\n  properties\n    deg = 0\n  end\n  methods\n"
                  "    function s = saveobj(o)\n      s.kelvin = o.deg + 273.15;\n    end\n  end\n"
                  "  methods (Static)\n    function o = loadobj(s)\n      o = Celsius; o.deg = s.kelvin - 273.15;\n"
                  "    end\n  end\nend\n"),
    "Meters.m": ("classdef Meters < double\n  properties\n    label = 'len'\n  end\n  methods\n"
                 "    function o = Meters(v)\n      if nargin == 0, v = 0; end\n      o = o@double(v);\n    end\n"
                 "  end\nend\n"),
    "Code16.m": ("classdef Code16 < int16\n  methods\n    function o = Code16(v)\n      if nargin == 0, v = 0; end\n"
                 "      o = o@int16(v);\n    end\n  end\nend\n"),
    "Bag.m": "classdef Bag < dynamicprops\n  properties\n    base = 1\n  end\nend\n",
    "Cached.m": ("classdef Cached\n  properties\n    data = []\n  end\n  properties (Transient)\n    total = 0\n  end\n"
                 "  methods\n    function o = set.data(o, v)\n      o.data = v;\n      o.total = sum(v);\n    end\n"
                 "  end\nend\n"),
    "Noisy.m": ("classdef Noisy < handle\n  properties\n    tag = 0\n  end\n  methods\n"
                "    function set.tag(o, v)\n      fprintf('setting %d\\n', v);\n      o.tag = v;\n    end\n"
                "    function delete(o)\n      fprintf('deleting %d\\n', o.tag);\n    end\n  end\nend\n"),
    "Range.m": ("classdef Range\n  properties\n    lo = 0\n    hi = 1\n  end\n  methods\n"
                "    function o = set.lo(o, v)\n      if v > o.hi, error('Range:order', 'lo is above hi'); end\n"
                "      o.lo = v;\n    end\n  end\nend\n"),
    "Shape.m": "classdef Shape < matlab.mixin.Heterogeneous\n  properties\n    id = 0\n  end\nend\n",
    "Disk.m": "classdef Disk < Shape\n  properties\n    r = 1\n  end\nend\n",
    "Square.m": "classdef Square < Shape\n  properties\n    side = 2\n  end\nend\n",
    "@Legacy/Legacy.m": "function o = Legacy(v)\n  s.val = v;\n  s.note = 'old';\n  o = class(s, 'Legacy');\nend\n",
    "@Legacy/value.m": "function v = value(o)\n  v = o.val;\nend\n",
}


class Fresh:
    """A new interpreter whose output is collected (one per save or load step)."""

    def __init__(self):
        self.output: list[str] = []
        self.interp = Interpreter(out=self.output.append)

    def run(self, code: str) -> str:
        self.output.clear()
        self.interp.execute(code)
        return "".join(self.output)

    def ok(self, expr: str) -> bool:
        return bool(np.asarray(self.interp.eval_expression(expr)).all())


@pytest.fixture
def folder(tmp_path, monkeypatch):
    for name, body in CLASSES.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(body)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def in_session(code: str) -> None:
    """Run code (which saves) in a session of its own, and end the session: its objects are destroyed
    now, so a delete method they run does not print into a later session."""
    s = Fresh()
    s.run(code)
    s.run("clear all")
    gc.collect()


def saved(code: str) -> Fresh:
    """Run code (which saves) in one session; return a new session to load in."""
    in_session(code)
    return Fresh()


def roundtrip(setup: str, names: str, version: str) -> Fresh:
    """Make variables, save them (``names``: quoted list), and load them all in a new session."""
    s = saved(f"{setup}; save('f.mat', {names}, '{version}');")
    s.run("load('f.mat');")
    return s


# -- cases that worked before, kept working --------------------------------------------------------

@pytest.mark.parametrize("version", VERSIONS)
def test_value_and_handle_objects_round_trip(folder, version):
    s = roundtrip("p = Pt; q = Pt; q.x = 30; arr = [Pt Pt]; arr(2).y = 7; h = Hd; h.v = 'hi';"
                  "st.inner = q; ce = {p, h}; e = Pt.empty(0, 3); g = repmat(Pt, 2, 3); g(2, 3).x = 5",
                  "'p', 'q', 'arr', 'h', 'st', 'ce', 'e', 'g'", version)
    assert s.ok("isa(p, 'Pt') && p.x == 1 && p.y == 2 && q.x == 30")
    assert s.ok("isa(arr, 'Pt') && isequal(size(arr), [1 2]) && arr(2).y == 7")
    assert s.ok("isa(h, 'Hd') && strcmp(h.v, 'hi') && isa(st.inner, 'Pt') && st.inner.x == 30")
    assert s.ok("isa(ce{1}, 'Pt') && ce{2} == h")       # the handle in the cell is the variable h
    assert s.ok("isa(e, 'Pt') && isequal(size(e), [0 3]) && isequal(size(g), [2 3]) && g(2, 3).x == 5")
    s.run("S = load('f.mat'); m = matfile('f.mat');")
    assert s.ok("isa(S.p, 'Pt') && S.q.x == 30 && isa(m.p, 'Pt') && m.arr(1, 2).y == 7")


@pytest.mark.parametrize("version", VERSIONS)
def test_property_value_kinds(folder, version):
    s = roundtrip(
        "b1 = Box; b1.v = datetime(2024, 3, 4, 5, 6, 7, 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd HH:mm');"
        "b2 = Box; b2.v = minutes([90 30]); b2.v.Format = 'hh:mm';"
        "b3 = Box; b3.v = categorical({'lo', 'hi', ''}, {'lo', 'hi', 'top'}, 'Ordinal', true);"
        "b4 = Box; b4.v = [\"a\" missing; \"ccc\" \"\"];"
        "b5 = Box; b5.v = containers.Map({'one', 'two'}, {1, [2 3]});"
        "k = 4; b6 = Box; b6.v = @(t) t * k; b7 = Box; b7.v = @cos;"
        "b8 = Box; b8.v = sparse([0 2; 3 0]); b9 = Box; b9.v = {int8(-3), uint64(18446744073709551615), single(2.5)};"
        "b10 = Box; b10.v = {[], zeros(0, 3), '', {}, struct('f', {}), strings(0, 1)};"
        "b11 = Box; b11.v = Box; b11.v.v = Pt; b12 = Box; b12.v = [Mood.Busy Mood.Gone]; b13 = Box; b13.v = Rank.High;"
        "b14 = Box; b14.v = calmonths(2) + caldays(3); b15 = Box; b15.v = 1 + 2i;"
        "b16 = Box; b16.v = reshape(int16(1:8), 2, 2, 2); b17 = Box; b17.v = struct('a', {1, 'x'});"
        "b18 = Box; b18.v = MException('t:id', 'bad %d', 3)",
        "'b*'", version)
    assert s.ok("isdatetime(b1.v) && b1.v == datetime(2024, 3, 4, 5, 6, 7, 'TimeZone', 'UTC')")
    assert s.ok("strcmp(b1.v.TimeZone, 'UTC') && strcmp(b1.v.Format, 'yyyy-MM-dd HH:mm')")
    assert s.ok("isduration(b2.v) && isequal(b2.v, minutes([90 30])) && strcmp(b2.v.Format, 'hh:mm')")
    assert s.ok("iscategorical(b3.v) && isordinal(b3.v) && isequal(categories(b3.v), {'lo'; 'hi'; 'top'})")
    assert s.ok("isequal(isundefined(b3.v), [false false true]) && b3.v(1) < b3.v(2)")
    assert s.ok("isstring(b4.v) && isequal(size(b4.v), [2 2]) && ismissing(b4.v(1, 2)) && b4.v(2, 1) == \"ccc\"")
    assert s.ok("b5.v.Count == 2 && b5.v('one') == 1 && isequal(b5.v('two'), [2 3])")
    assert s.ok("b6.v(2) == 8 && b7.v(0) == 1 && strcmp(func2str(b7.v), 'cos')")
    assert s.ok("issparse(b8.v) && isequal(full(b8.v), [0 2; 3 0])")
    assert s.ok("isequal(cellfun(@class, b9.v, 'UniformOutput', false), {'int8', 'uint64', 'single'})")
    assert s.ok("b9.v{2} == intmax('uint64') && b9.v{1} == -3")
    assert s.ok("isequal(size(b10.v{2}), [0 3]) && ischar(b10.v{3}) && iscell(b10.v{4}) && isstruct(b10.v{5})")
    assert s.ok("isfield(b10.v{5}, 'f') && isstring(b10.v{6}) && isequal(size(b10.v{6}), [0 1])")
    assert s.ok("isa(b11.v, 'Box') && isa(b11.v.v, 'Pt') && b11.v.v.y == 2")
    assert s.ok("isa(b12.v, 'Mood') && isequal(b12.v, [Mood.Busy Mood.Gone]) && b13.v == Rank.High")
    assert s.ok("int32(b13.v) == 9 && isequal(b14.v, calmonths(2) + caldays(3)) && b15.v == 1 + 2i")
    assert s.ok("isa(b16.v, 'int16') && isequal(b16.v, reshape(int16(1:8), 2, 2, 2)) && strcmp(b17.v(2).a, 'x')")
    assert s.ok("isa(b18.v, 'MException') && strcmp(b18.v.identifier, 't:id') && strcmp(b18.v.message, 'bad 3')")


@pytest.mark.parametrize("version", VERSIONS)
def test_handles_shared_and_cycles(folder, version):
    s = roundtrip("h = Hd; h.v = 1; g = h; a = Box; a.v = h; b = Box; b.v = h; me = Hd; me.v = me;"
                  "p = Hd; q = Hd; p.v = q; q.v = p; c = {h, h}; s.one = h; s.two = h; arr = [h h]",
                  "'h', 'g', 'a', 'b', 'me', 'p', 'c', 's', 'arr'", version)
    s.run("h.v = 42;")
    assert s.ok("g.v == 42 && a.v.v == 42 && b.v == h && c{1} == c{2} && c{2}.v == 42")
    assert s.ok("s.one == s.two && s.two == h && arr(1) == arr(2) && arr(2) == h")
    assert s.ok("me.v == me && p.v.v == p && isa(p.v, 'Hd')")


@pytest.mark.parametrize("version", VERSIONS)
def test_dependent_transient_constant_saveobj_and_subclasses(folder, version):
    s = roundtrip("c = Calc; c.a = 4; c.scratch = 99; t = Celsius; t.deg = 25; ts = [Celsius Celsius]; ts(2).deg = 3;"
                  "d = Dog; d.legs = 3; d.name = 'ace'; hd = HDog; hd.legs = 5", "'c', 't', 'ts', 'd', 'hd'", version)
    assert s.ok("c.a == 4 && c.twice == 8 && c.scratch == 5 && c.K == 10")    # Transient: the default again
    assert s.ok("abs(t.deg - 25) < 1e-9 && isa(ts, 'Celsius') && abs(ts(2).deg - 3) < 1e-9")
    assert s.ok("isa(d, 'Dog') && isa(d, 'Animal') && d.legs == 3 && strcmp(d.name, 'ace')")
    assert s.ok("isa(hd, 'HDog') && hd.legs == 5 && strcmp(hd.name, 'rex')")


@pytest.mark.parametrize("version", VERSIONS)
def test_objects_in_tables_maps_functions_and_heterogeneous_arrays(folder, version):
    s = roundtrip("h = Hd; h.v = 3; t = table({h; h}, [1; 2]); mp = containers.Map({'k'}, {h}); f = @() h.v;"
                  "het = [Disk Square]; het(2).side = 5", "'t', 'mp', 'f', 'het'", version)
    s.run("t.Var1{1}.v = 9;")
    assert s.ok("t.Var1{2}.v == 9 && mp('k').v == 9 && f() == 9")
    assert s.ok("isa(het, 'Shape') && isa(het(1), 'Disk') && isa(het(2), 'Square') && het(2).side == 5")


# -- objects that lost their data --------------------------------------------------------------------

@pytest.mark.parametrize("version", VERSIONS)
def test_objects_of_classes_derived_from_builtin_types_keep_their_data(folder, version):
    s = roundtrip("m = Meters([1.5 2 3]); m.label = 'run'; c = Code16(7); b = Box; b.v = Meters(4)",
                  "'m', 'c', 'b'", version)
    assert s.ok("isa(m, 'Meters') && isequal(double(m), [1.5 2 3]) && isequal(size(m), [1 3])")
    assert s.ok("strcmp(m.label, 'run') && isa(c, 'Code16') && int16(c) == 7 && double(b.v) == 4")


@pytest.mark.parametrize("version", VERSIONS)
def test_dynamic_properties_are_saved(folder, version):
    s = roundtrip("x = Bag; addprop(x, 'extra'); x.extra = {1, 'two'}; x.base = 3", "'x'", version)
    assert s.ok("isprop(x, 'extra') && isequal(x.extra, {1, 'two'}) && x.base == 3")
    s.run("x.extra = 5;")
    assert s.ok("x.extra == 5")


@pytest.mark.parametrize("version", VERSIONS)
def test_old_style_objects_keep_their_fields(folder, version):
    s = roundtrip("o = Legacy(7)", "'o'", version)
    assert s.ok("isa(o, 'Legacy') && value(o) == 7")


@pytest.mark.parametrize("version", VERSIONS)
def test_timetables_and_table_properties(folder, version):
    s = roundtrip("tt = timetable(seconds([1; 2; 4]), [10; 20; 40], 'VariableNames', {'level'});"
                  "tt.Properties.DimensionNames{1} = 'When'; tt.Properties.VariableUnits = {'cm'};"
                  "t = table([1; 2], {'p'; 'q'}, 'VariableNames', {'n', 'c'}, 'RowNames', {'r1', 'r2'});"
                  "t.Properties.Description = 'about t'; t.Properties.UserData = struct('k', 5);"
                  "t.Properties.VariableUnits = {'m', ''}; t.Properties.VariableDescriptions = {'count', 'code'};"
                  "t.Properties.DimensionNames = {'Name', 'Data'};"
                  "t = addprop(t, {'Source', 'Scale'}, {'table', 'variable'});"
                  "t.Properties.CustomProperties.Source = 'lab'; t.Properties.CustomProperties.Scale = [2 3];"
                  "b = Box; b.v = tt", "'tt', 't', 'b'", version)
    assert s.ok("istimetable(tt) && isequal(tt.When, seconds([1; 2; 4])) && isequal(tt.level, [10; 20; 40])")
    assert s.ok("isequal(tt.Properties.VariableUnits, {'cm'}) && istimetable(b.v) && isequal(b.v.level, [10; 20; 40])")
    assert s.ok("istable(t) && isequal(t.n, [1; 2]) && isequal(t.Properties.RowNames, {'r1'; 'r2'})")
    assert s.ok("strcmp(t.Properties.Description, 'about t') && t.Properties.UserData.k == 5")
    assert s.ok("isequal(t.Properties.VariableUnits, {'m', ''})")
    assert s.ok("isequal(t.Properties.VariableDescriptions, {'count', 'code'})")
    assert s.ok("isequal(t.Properties.DimensionNames, {'Name', 'Data'}) && isequal(t.Name, {'r1'; 'r2'})")
    assert s.ok("strcmp(t.Properties.CustomProperties.Source, 'lab')")
    assert s.ok("isequal(t.Properties.CustomProperties.Scale, [2 3])")


@pytest.mark.parametrize("version", VERSIONS)
def test_dictionaries(folder, version):
    s = roundtrip("d = dictionary([1 2], [\"a\" \"b\"]); h = Hd; e = dictionary(\"k\", {h}); u = dictionary;"
                  "b = Box; b.v = d",
                  "'d', 'e', 'u', 'b', 'h'", version)
    assert s.ok("isa(d, 'dictionary') && d(2) == \"b\" && numEntries(d) == 2 && isequal(keys(d), [1; 2])")
    assert s.ok("e(\"k\"){1} == h && ~isConfigured(u) && b.v(1) == \"a\"")


@pytest.mark.parametrize("version", VERSIONS)
def test_shared_maps_stay_shared(folder, version):
    s = roundtrip("m1 = containers.Map(); m2 = m1; m1('a') = 1; h = Hd; h.v = m1; m1('h') = h",
                  "'m1', 'm2', 'h'", version)
    s.run("m1('b') = 2;")
    assert s.ok("m2.Count == 3 && h.v.Count == 3 && m1('h') == h")


@pytest.mark.parametrize("version", VERSIONS)
def test_exception_details(folder, version):
    s = roundtrip("e = MException('a:b', 'outer'); e = addCause(e, MException('c:d', 'inner'))", "'e'", version)
    assert s.ok("strcmp(e.identifier, 'a:b') && numel(e.cause) == 1 && strcmp(e.cause{1}.message, 'inner')")


# -- one variable of several, matfile and whos ---------------------------------------------------------

@pytest.mark.parametrize("version", VERSIONS)
def test_loading_one_variable_that_shares_a_handle_with_another(folder, version):
    # g and h are one object; it is written in full in only one of the two variables
    in_session(f"h = Hd; h.v = 5; g = h; a = Hd; a.v = 'in a'; b = Box; b.v = a; c = Box; c.v = a;"
                f"save('f.mat', 'h', 'g', 'b', 'c', '{version}');")
    for name in ("g", "h", "b", "c"):
        s = Fresh()
        s.run(f"load('f.mat', '{name}');")
        assert s.interp.get_var(name) is not None
    s = Fresh()
    s.run("load('f.mat', 'c'); S = load('f.mat', 'g'); m = matfile('f.mat'); w = whos('-file', 'f.mat', 'h');")
    assert s.ok("strcmp(c.v.v, 'in a') && S.g.v == 5 && m.h.v == 5 && isa(m.c.v, 'Hd') && strcmp(w.class, 'Hd')")


@pytest.mark.parametrize("version", VERSIONS)
def test_save_struct_fields_sharing_a_handle(folder, version):
    s = saved(f"h = Hd; h.v = 1; s.zz = h; s.aa = h; save('f.mat', '-struct', 's', '{version}');")
    s.run("load('f.mat'); aa.v = 2;")
    assert s.ok("zz.v == 2")


@pytest.mark.parametrize("version", VERSIONS)
def test_handles_added_by_append_stay_separate(folder, version):
    # three save calls; each writes its own handle objects, which must not be mixed up on load
    in_session(f"k = Hd; k.v = 'k'; save('f.mat', 'k', '{version}');")
    in_session("b = Hd; b.v = 'b'; y = b; save('f.mat', 'b', 'y', '-append');")
    in_session("z = Hd; z.v = 'z'; a = z; save('f.mat', 'a', 'z', '-append');")
    s = Fresh()
    s.run("load('f.mat');")
    assert s.ok("strcmp(k.v, 'k') && strcmp(y.v, 'b') && y == b && strcmp(a.v, 'z') && a == z && k ~= b")


@pytest.mark.parametrize("version", VERSIONS)
def test_whos_size_and_disp_of_matfile_with_objects(folder, version):
    s = saved(f"p = Pt; arr = [Pt Pt Pt]; str = [\"a\" \"b\"]; tt = timetable(seconds([1; 2]), [3; 4]);"
              f"d = datetime(2024, 1, 1) + days(0:3); save('f.mat', 'p', 'arr', 'str', 'tt', 'd', '{version}');")
    s.run("m = matfile('f.mat'); w = whos(m); f = whos('-file', 'f.mat'); txt = evalc('disp(m)');")
    for k in range(5):
        assert s.ok(f"strcmp(w({k + 1}).class, f({k + 1}).class) && isequal(w({k + 1}).size, f({k + 1}).size)")
    s.run("names = {w.name}; cls = {w.class};")
    assert s.ok("isequal(names, {'arr', 'd', 'p', 'str', 'tt'})")
    assert s.ok("isequal(cls, {'Pt', 'datetime', 'Pt', 'string', 'timetable'})")
    assert s.ok("isequal(size(m, 'arr'), [1 3]) && isequal(size(m, 'd'), [1 4]) && size(m, 'tt', 1) == 2")
    assert "[1x3 Pt]" in s.run("disp(txt)") and "[2x1 timetable]" in s.run("disp(txt)")


@pytest.mark.parametrize("version", VERSIONS)
def test_matfile_reads_and_writes_objects(folder, version):
    s = saved(f"arr = [Pt Pt]; arr(2).y = 8; h = Hd; h.v = 5; save('f.mat', 'arr', 'h', '{version}');")
    s.run("m = matfile('f.mat', 'Writable', true); z = m.arr(1, 2); a = m.h; a.v = 9; b = m.h;")
    assert s.ok("isa(z, 'Pt') && z.y == 8 && b.v == 5 && a ~= b")    # each read loads the object anew
    s.run("m.n = Hd; m.n2 = Meters(3); S = load('f.mat');")
    assert s.ok("isa(S.n, 'Hd') && double(S.n2) == 3 && S.arr(2).y == 8")


def test_whos_file_runs_no_code_of_the_class(folder):
    s = saved("n = Noisy; n.tag = 4; save('f.mat', 'n'); save('g.mat', 'n', '-v7.3'); clear n")
    for f in ("f.mat", "g.mat"):
        out = s.run(f"w = whos('-file', '{f}'); m = matfile('{f}'); size(m, 'n'); whos(m); clear m w")
        assert "setting" not in out and "deleting" not in out and "Noisy" in out


# -- what load does with the class -------------------------------------------------------------------

@pytest.mark.parametrize("version", VERSIONS)
def test_set_methods_run_on_load(folder, version):
    s = roundtrip("c = Cached; c.data = [1 2 3]; n = Noisy; n.tag = 6", "'c', 'n'", version)
    assert s.ok("c.total == 6")       # the Transient property is computed again by the set method
    out = s.run("clear n")
    assert "deleting 6" in out        # a loaded handle object runs its delete method when it is destroyed


@pytest.mark.parametrize("version", VERSIONS)
def test_set_method_that_fails_on_load_keeps_the_saved_value(folder, version):
    # lo is restored before hi: its set method compares it with the default hi and fails
    s = saved(f"r = Range; r.hi = 10; r.lo = 5; save('f.mat', 'r', '{version}');")
    out = s.run("load('f.mat');")
    assert "Warning" in out and "lo is above hi" in out
    assert s.ok("isa(r, 'Range') && r.lo == 5 && r.hi == 10")


def _change_class(folder, body: str) -> None:
    (folder / "Chg.m").write_text(body)


@pytest.mark.parametrize("version", VERSIONS)
def test_class_changed_since_saving(folder, version):
    _change_class(folder, "classdef Chg\n  properties\n    a = 1\n    gone = 2\n  end\nend\n")
    in_session(f"x = Chg; x.a = -5; x.gone = 'g'; y = [Chg Chg]; save('f.mat', 'x', 'y', '{version}');")
    # a property removed and one added: the removed one is dropped, the added one has its default
    _change_class(folder, "classdef Chg\n  properties\n    a = 1\n    added = 3\n  end\nend\n")
    s = Fresh()
    out = s.run("load('f.mat');")
    assert out == "" and s.ok("x.a == -5 && x.added == 3 && ~isprop(x, 'gone') && isequal(size(y), [1 2])")
    # with a loadobj method, the saved values that no longer fit arrive as a struct
    _change_class(folder, "classdef Chg\n  properties\n    a = 1\n  end\n  methods (Static)\n"
                          "    function o = loadobj(s)\n      o = Chg;\n"
                          "      if isstruct(s), o.a = numel(fieldnames(s)); end\n"
                          "    end\n  end\nend\n")
    s = Fresh()
    s.run("load('f.mat');")
    assert s.ok("x.a == 2 && y(2).a == 2")
    # a value that fails the property's validation now: the default, with a warning
    _change_class(folder, "classdef Chg\n  properties\n    a (1,1) double {mustBePositive} = 1\n    gone\n  end\nend\n")
    s = Fresh()
    out = s.run("load('f.mat', 'x');")
    assert "Warning" in out and "'a'" in out and s.ok("x.a == 1 && strcmp(x.gone, 'g')")


@pytest.mark.parametrize("version", VERSIONS)
def test_class_not_on_the_path(folder, version):
    in_session(f"x = Pt; x.x = 9; y = [Pt Pt]; y(2).y = 4; z = {{Pt}}; save('f.mat', 'x', 'y', 'z', '{version}');")
    os.remove(folder / "Pt.m")
    s = Fresh()
    out = s.run("load('f.mat');")
    assert out.count("Warning") == 3 and "'Pt'" in out          # one warning for each variable
    assert s.ok("isstruct(x) && x.x == 9 && isstruct(y) && isequal(size(y), [1 2]) && y(2).y == 4")
    assert s.ok("iscell(z) && isstruct(z{1}) && z{1}.x == 1")
