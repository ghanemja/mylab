"""Object arrays, enumerations, events, @Class folders, packages."""

import os

import numpy as np
import pytest

from mylab.errors import MatlabError

POINT = """
classdef Point
    properties
        x = 0
        y = 0
    end
    methods
        function obj = Point(x, y)
            if nargin > 0
                obj.x = x; obj.y = y;
            end
        end
        function d = dist(obj)
            d = sqrt([obj.x].^2 + [obj.y].^2);
        end
        function r = plus(a, b)
            r = Point(a.x + b.x, a.y + b.y);
        end
    end
end
"""

COUNTER = """
classdef Counter < handle
    properties
        n = 0
    end
    events
        Ticked
        Reset
    end
    methods
        function tick(obj)
            obj.n = obj.n + 1;
            notify(obj, 'Ticked');
        end
    end
end
"""

COLOR = """
classdef Color
    properties
        rgb
    end
    enumeration
        Red (1, 0, 0)
        Green (0, 1, 0), Blue (0, 0, 1)
    end
    methods
        function c = Color(r, g, b)
            c.rgb = [r g b];
        end
        function tf = isWarm(c)
            tf = c.rgb(1) > 0;
        end
    end
end
"""


@pytest.fixture
def cls(m, tmp_path):
    for name, src in (("Point", POINT), ("Counter", COUNTER), ("Color", COLOR)):
        (tmp_path / f"{name}.m").write_text(src)
    (tmp_path / "Level.m").write_text(
        "classdef Level < uint8\n    enumeration\n        Low (1)\n        High (5)\n    end\nend\n")
    (tmp_path / "Weekday.m").write_text(
        "classdef Weekday\n    enumeration\n        Mon, Tue\n        Wed\n    end\nend\n")
    m.interp.path.insert(0, str(tmp_path))
    m.tmp = tmp_path
    return m


# -- object arrays ----------------------------------------------------------------------------

def test_object_array_grow_and_index(cls):
    m = cls
    m.run("objs(3) = Point(3, 4);")
    assert m.num("numel(objs)") == 3
    assert m.num("size(objs, 2)") == 3
    assert m.num("objs(1).x") == 0  # default-constructed filler
    assert m.num("objs(3).y") == 4
    assert m.str("class(objs)") == "Point"
    assert m.num("isa(objs, 'Point')")
    m.run("objs(2).x = 7; p = objs(2);")
    assert m.num("p.x") == 7
    assert m.num("objs(end).x") == 3


def test_object_array_concat_and_property_list(cls):
    m = cls
    m.run("a = [Point(1,2), Point(3,4), Point(5,6)]; xs = [a.x]; c = {a.y};")
    np.testing.assert_array_equal(m.val("xs"), [[1, 3, 5]])
    assert m.num("numel(c)") == 3
    m.run("b = [a, Point(7,8)]; col = [Point(1,1); Point(2,2)];")
    assert m.num("numel(b)") == 4
    np.testing.assert_array_equal(m.num("size(col)"), [[2, 1]])
    m.run("d = dist(a(1:2));")  # methods get the whole array
    np.testing.assert_allclose(m.val("d"), [[np.sqrt(5), 5]])
    m.run("e = []; for k = 1:3, e = [e, Point(k, k)]; end")
    assert m.num("numel(e)") == 3


def test_object_array_loop_arrayfun_delete_empty(cls):
    m = cls
    m.run("a = [Point(1,2), Point(3,4), Point(5,6)]; s = 0; for p = a, s = s + p.x; end")
    assert m.num("s") == 9
    m.run("v = arrayfun(@(p) p.x + p.y, a);")
    np.testing.assert_array_equal(m.val("v"), [[3, 7, 11]])
    m.run("a(2) = []; e = Point.empty; e2 = Point.empty(0, 3);")
    assert m.num("numel(a)") == 2 and m.num("a(2).x") == 5
    assert m.num("isempty(e)") and m.num("size(e2, 2)") == 3
    m.run("e(end+1) = Point(9, 9); e(end+1) = Point(8, 8);")
    assert m.num("numel(e)") == 2 and m.num("e(2).x") == 8
    m.run("[a.x] = deal(0);")
    np.testing.assert_array_equal(m.num("[a.x]"), [[0, 0]])
    out = m.run("a")
    assert "1×2 Point array with properties" in out and "x" in out
    assert m.num("isequal(a, a)")
    m.run("q = a';")
    np.testing.assert_array_equal(m.num("size(q)"), [[2, 1]])


def test_handle_object_array(cls):
    m = cls
    m.run("h = Counter(); hs = [h, Counter(), h]; hs(2).tick(); hs(1).tick();")
    assert m.num("h.n") == 1
    np.testing.assert_array_equal(m.num("[hs.n]"), [[1, 1, 1]])
    np.testing.assert_array_equal(m.num("hs == h"), [[True, False, True]])


def test_object_array_errors(cls):
    m = cls
    m.run("a = [Point(1,2), Point(3,4)];")
    with pytest.raises(MatlabError, match="Index"):
        m.run("a(5)")
    with pytest.raises(MatlabError, match="Cannot concatenate"):
        m.run("b = [a, 5];")
    with pytest.raises(MatlabError):
        m.run("a.x = 1;")


# -- enumerations -------------------------------------------------------------------------------

def test_enumeration_basics(cls):
    m = cls
    m.run("c = Color.Red; d = Color.Blue;")
    assert m.num("c == Color.Red") and m.num("c ~= d") and not m.num("c == d")
    np.testing.assert_array_equal(m.num("d.rgb"), [[0, 0, 1]])
    assert m.num("c.isWarm()") and not m.num("isWarm(d)")
    assert m.str("char(c)") == "Red"
    assert m.num("string(d) == \"Blue\"")
    assert m.num("isequal(c, Color.Red)") and m.num("isenum(c)")
    assert m.num("c == 'Red'")
    assert "Red" in m.run("c") and "enumeration" in m.run("c")
    m.run("e = Color('Green');")
    assert m.num("e == Color.Green")


def test_enumeration_switch_and_listing(cls):
    m = cls
    m.run("""
c = Color.Green; r = 0;
switch c
    case Color.Red
        r = 1;
    case {Color.Green, Color.Blue}
        r = 2;
end
[members, names] = enumeration('Color');
w = Weekday.Tue;
""")
    assert m.num("r") == 2
    assert m.num("numel(members)") == 3 and m.str("names{3}") == "Blue"
    assert m.str("char(w)") == "Tue"
    assert "Green" in m.run("enumeration Color")
    assert m.num("ismember(Color.Red, [Color.Red, Color.Blue])")
    out = m.run("arr = [Color.Red, Color.Blue]")
    assert "enumeration array" in out and "Blue" in out
    np.testing.assert_array_equal(m.num("arr == Color.Blue"), [[False, True]])


def test_enumeration_with_numeric_base(cls):
    m = cls
    m.run("lv = Level.High;")
    assert m.num("double(lv)") == 5
    assert m.num("lv == 5")
    assert m.num("Level(1) == Level.Low")
    assert m.str("class(uint8(lv))") == "uint8"
    with pytest.raises(MatlabError, match="not a member"):
        m.run("Color('Purple')")


# -- events ----------------------------------------------------------------------------------------

def test_events_notify_and_listeners(cls):
    m = cls
    m.run("""
c = Counter(); global log; log = {};
L = addlistener(c, 'Ticked', @(src, evt) record(src, evt));
c.tick(); c.tick();
L.Enabled = false; c.tick();
L.Enabled = true; delete(L); c.tick();
function record(src, evt)
    global log
    log{end+1} = sprintf('%s:%d', evt.EventName, src.n);
end
""")
    assert m.num("numel(log)") == 2
    assert m.str("log{2}") == "Ticked:2"
    assert m.num("c.n") == 4


def test_events_method_syntax_and_errors(cls):
    m = cls
    m.run("c = Counter(); hits = 0; c.addlistener('Reset', @(~, ~) assignin('base', 'hits', 1)); "
          "c.notify('Reset');")
    assert m.num("hits") == 1
    with pytest.raises(MatlabError, match="not defined"):
        m.run("notify(c, 'Nope')")
    with pytest.raises(MatlabError, match="handle class"):
        m.run("addlistener(5, 'X', @disp)")


def test_property_postset_listener(cls):
    m = cls
    m.run("c = Counter(); seen = []; "
          "addlistener(c, 'n', 'PostSet', @(p, e) assignin('base', 'seen', e.AffectedObject.n)); c.n = 42;")
    assert m.num("seen") == 42


# -- @Class folders and packages ---------------------------------------------------------------------

def test_class_folder_methods(cls):
    m = cls
    folder = m.tmp / "@Shape"
    folder.mkdir()
    (folder / "Shape.m").write_text("""
classdef Shape
    properties
        side = 2
    end
    methods
        function obj = Shape(s)
            if nargin, obj.side = s; end
        end
        a = area(obj)
    end
    methods (Static)
        s = unit()
    end
end
""")
    (folder / "area.m").write_text("function a = area(obj)\n    a = helper(obj.side);\nend\n"
                                   "function r = helper(s)\n    r = s^2;\nend\n")
    (folder / "unit.m").write_text("function s = unit()\n    s = Shape(1);\nend\n")
    (folder / "describe.m").write_text("function t = describe(obj)\n    t = sprintf('side %d', obj.side);\nend\n")
    m.run("s = Shape(3); a = s.area(); u = Shape.unit(); t = describe(s);")
    assert m.num("a") == 9 and m.num("u.side") == 1 and m.str("t") == "side 3"


def test_package_functions_and_classes(cls):
    m = cls
    pkg = m.tmp / "+geom"
    (pkg / "+util").mkdir(parents=True)
    (pkg / "twice.m").write_text("function y = twice(x)\n    y = 2 * x;\nend\n")
    (pkg / "+util" / "inc.m").write_text("function y = inc(x)\n    y = x + 1;\nend\n")
    (pkg / "Circle.m").write_text("""
classdef Circle
    properties
        r = 1
    end
    methods
        function obj = Circle(r)
            if nargin, obj.r = r; end
        end
        function a = area(obj)
            a = pi * obj.r^2;
        end
    end
    methods (Static)
        function c = unit()
            c = geom.Circle(1);
        end
    end
end
""")
    m.run("a = geom.twice(4); b = geom.util.inc(1); c = geom.Circle(2); f = @geom.twice; d = f(5);")
    assert m.num("a") == 8 and m.num("b") == 2 and m.num("d") == 10
    np.testing.assert_allclose(m.num("c.area()"), 4 * np.pi)
    assert m.str("class(c)") == "geom.Circle"
    assert m.num("isa(c, 'geom.Circle')")
    assert m.num("geom.Circle.unit().r") == 1
    assert os.path.isdir(pkg)
    m.run("import geom.*; e = twice(3); import geom.util.inc; f2 = inc(4); k = Circle(3);")
    assert m.num("e") == 6 and m.num("f2") == 5 and m.num("k.r") == 3


# -- larger scenarios -------------------------------------------------------------------------------

def test_object_array_property_and_helpers(cls):
    m = cls
    (m.tmp / "Queue.m").write_text("""
classdef Queue < handle
    properties
        items = Point.empty
    end
    methods
        function push(o, it)
            o.items(end+1) = it;
        end
        function t = total(o)
            t = sum([o.items.x]);
        end
    end
end
""")
    m.run("q = Queue(); q.push(Point(3, 0)); q.push(Point(5, 0)); q.push(Point(7, 0)); n = numel(q.items);"
          "tot = q.total(); q.items(2) = []; n2 = numel(q.items); r = repmat(Point(1, 1), 1, 3);"
          "c = num2cell(q.items); col = Point.empty(0, 1); for k = 1:3, col(end+1, 1) = Point(k, k); end;"
          "both = [col; Point(10, 0)]; pick = both([1 3]); big = both([both.x] > 2);"
          "st.list = [Point(1, 0), Point(2, 0)]; z = [st.list.x]; rs = reshape(both, 2, 2); fl = fliplr(r);"
          "names = arrayfun(@(p) sprintf('p%d', p.x), st.list, 'UniformOutput', false);")
    assert m.num("n") == 3 and m.num("tot") == 15 and m.num("n2") == 2
    np.testing.assert_array_equal(m.num("size(r)"), [[1, 3]])
    assert m.str("class(c)") == "cell" and m.num("numel(c)") == 2
    np.testing.assert_array_equal(m.num("size(col)"), [[3, 1]])
    np.testing.assert_array_equal(m.num("[pick.x]"), [[1, 3]])
    assert m.num("numel(big)") == 2 and m.str("names{2}") == "p2"
    np.testing.assert_array_equal(m.val("z"), [[1, 2]])
    np.testing.assert_array_equal(m.num("size(rs)"), [[2, 2]])
    assert m.num("isrow(r)") and not m.num("iscolumn(r)") and m.num("isobject(r)")
    assert m.str("class(properties(r))") == "cell"


def test_events_with_custom_data_and_delete(cls):
    m = cls
    (m.tmp / "PressData.m").write_text(
        "classdef PressData < event.EventData\n    properties\n        Amount\n    end\n    methods\n"
        "        function o = PressData(k)\n            o.Amount = k;\n        end\n    end\nend\n")
    (m.tmp / "Button.m").write_text("""
classdef Button < handle
    properties
        count = 0
    end
    events
        Pressed
    end
    methods
        function press(o, k)
            o.count = o.count + 1;
            notify(o, 'Pressed', PressData(k));
        end
    end
end
""")
    out = m.run("""
b = Button(); lh = addlistener(b, 'Pressed', @onPress);
b.press(3); delete(lh); b.press(5);
hs = [Button(), Button()]; ok = isvalid(hs); delete(hs); gone = isvalid(hs);
b.delete(); v = isvalid(b);
function onPress(src, evt)
    fprintf('%d %d %s\\n', src.count, evt.Amount, evt.EventName);
end
""")
    assert out.strip() == "1 3 Pressed"
    assert not m.num("v")
    np.testing.assert_array_equal(m.val("ok"), [[True, True]])
    np.testing.assert_array_equal(m.val("gone"), [[False, False]])


def test_enumeration_in_containers_and_dynamic_access(cls):
    m = cls
    out = m.run("s = struct('c', Color.Blue); k = {Color.Red, Color.Green}; "
                "for e = enumeration('Color')', fprintf('%s:%d ', char(e), e.rgb(1)); end; fprintf('\\n');"
                "n = 'Green'; g = Color.(n); t = [Color.Red Color.Blue Color.Red]; cnt = sum(t == Color.Red);"
                "disp(Color.Red); s")
    assert "Red:1 Green:0 Blue:0" in out and "    Red\n" in out and "c: Blue" in out
    assert m.num("g == Color.Green") and m.num("cnt") == 2 and m.str("class(k{1})") == "Color"
    assert m.num("isa(Color.Red, 'Color')") and m.num("s.c == Color.Blue")


def test_delete_method_syntax_runs_class_delete(m):
    out = m.run("""
r = Res(); r.delete(); a = isvalid(r); q = Res(); delete(q); b = isvalid(q);
classdef Res < handle
    properties
        listener = 5
    end
    methods
        function delete(obj)
            disp('closing')
        end
    end
end
""")
    assert out.split() == ["closing", "closing"] and not m.num("a") and not m.num("b")
    m.run("w = Res(); v = w.listener(1);")  # a property named like a handle method is still a property
    assert m.num("v") == 5
