"""containers.Map and classdef classes."""

from pathlib import Path

import numpy as np
import pytest

from mylab import MatlabError

CLASSES = Path(__file__).parent / "fixtures" / "classes"


def test_map_basics(m):
    m.run("mp = containers.Map(); mp('apple') = 3; mp('kiwi') = 5;")
    assert m.num("mp('kiwi')").item() == 5
    assert m.num("mp.Count").item() == 2
    assert m.num("length(mp)").item() == 2
    assert m.num("isKey(mp, 'apple')").item()
    assert not m.num("isKey(mp, 'fig')").item()
    np.testing.assert_array_equal(m.num("isKey(mp, {'apple', 'fig'})"), [[True, False]])
    m.run("k = keys(mp); v = values(mp);")
    assert m.str("k{1}") == "apple" and m.num("v{2}").item() == 5
    m.run("alias = mp; alias('fig') = 1;")  # handle semantics
    assert m.num("mp.Count").item() == 3
    m.run("remove(mp, 'apple');")
    assert m.num("mp.Count").item() == 2
    assert m.str("class(mp)") == "containers.Map"
    assert m.num("isobject(mp)").item()
    with pytest.raises(MatlabError, match="not present"):
        m.run("mp('nope')")


def test_map_construction_forms(m):
    m.run("q = containers.Map({'a', 'b'}, [10 20]);")
    assert m.num("q('b')").item() == 20
    m.run("r = containers.Map('KeyType', 'double', 'ValueType', 'any'); r(3) = 'three';")
    assert m.str("r(3)") == "three"
    m.run("s = containers.Map([1 2 3], {'x', 'y', 'z'});")
    assert m.str("s(2)") == "y"
    assert m.str("q.KeyType") == "char"
    out = m.run("q")
    assert "Map with properties" in out and "Count: 2" in out


@pytest.fixture
def cm(m):
    m.interp.path.insert(0, str(CLASSES))
    return m


def test_value_class(cm):
    m = cm
    m.run("a = Vec2(3, 4); b = Vec2(1, 1); c = a + b; d = a * 2;")
    assert (m.num("c.x").item(), m.num("c.y").item()) == (4, 5)
    assert m.num("d.y").item() == 8
    assert m.num("a.len").item() == 5  # dependent property via get.len
    assert m.str("Vec2.ZERO_NAME") == "origin"
    m.run("u = Vec2.unitX();")
    assert m.num("u.x").item() == 1
    m.run("a2 = a.scale(10); a3 = scale(a, 2);")
    assert m.num("a.x").item() == 3  # value semantics: a unchanged
    assert m.num("a2.x").item() == 30 and m.num("a3.y").item() == 8
    m.run("a.x = 7;")
    assert m.num("a.x").item() == 7
    assert m.str("class(a)") == "Vec2" and m.num("isa(a, 'Vec2')").item()
    with pytest.raises(MatlabError, match="Unrecognized property"):
        m.run("a.nope = 1;")
    out = m.run("a")
    assert "Vec2 with properties" in out and "len: 8.0623" in out


def test_handle_class_and_custom_disp(cm):
    m = cm
    m.run("k = Counter(); k.increment(); k.increment(5); k2 = k; k2.increment();")
    assert m.num("k.count").item() == 7
    assert m.num("k == k2").item()
    assert m.num("isa(k, 'handle')").item()
    assert m.run("k") == "k = \n\n  Counter at 7\n\n"
    assert m.run("disp(k)") == "  Counter at 7\n"


def test_inheritance(cm):
    m = cm
    m.run("dg = Dog('Rex');")
    assert m.str("dg.speak()") == "Rex says woof"
    assert m.str("dg.describe()") == "I am Rex"
    assert m.str("dg.both()") == "Rex makes a sound / Rex says woof"
    assert m.num("isa(dg, 'Animal')").item()
    assert m.str("dg.tricks{1}") == "sit"


def test_classdef_in_session(m):
    m.run("""
classdef Account < handle
    properties
        balance = 0
    end
    methods
        function deposit(obj, amt)
            if amt <= 0
                error('Account:bad', 'Deposit must be positive');
            end
            obj.balance = obj.balance + amt;
        end
    end
end
""")
    m.run("acc = Account(); acc.deposit(50); deposit(acc, 25);")
    assert m.num("acc.balance").item() == 75
    with pytest.raises(MatlabError, match="positive"):
        m.run("acc.deposit(-1)")


def test_access_abstract_validation(cm):
    m = cm
    with pytest.raises(MatlabError, match="abstract"):
        m.run("s = Shape();")
    m.run("c = Circle(2);")
    assert m.str("c.name") == "circle"
    with pytest.raises(MatlabError, match="cannot set"):
        m.run("c.name = 'x';")
    with pytest.raises(MatlabError, match="cannot access"):
        m.run("v = c.secret;")
    assert m.num("c.peek()").item() == 42  # inside a method of the defining class
    with pytest.raises(MatlabError, match="Cannot access method"):
        m.run("c.hidden();")
    with pytest.raises(MatlabError, match="positive"):
        m.run("c.r = -1;")
    with pytest.raises(MatlabError, match="must be 1x1"):
        m.run("c.r = [1 2];")
    m.run("c.r = int8(3);")
    assert m.str("class(c.r)") == "double"
    m.run("c.tag = 'abc';")
    assert m.str("class(c.tag)") == "string"
    out = m.run("c.describe();")
    assert out == "circle with area 28.27\n(radius 3)\n"
    out = m.run("c")
    assert "secret" not in out and "r: 3" in out


def test_object_utilities(cm):
    m = cm
    m.run("c = Circle(1); d = c; e = Circle(1);")
    assert m.num("isequal(c, d)").item() and not m.num("isequal(c, e)").item()
    assert m.num("isvalid(c)").item()
    assert m.num("isprop(c, 'r')").item() and not m.num("isprop(c, 'secret')").item()
    assert m.num("ismethod(c, 'area')").item() and not m.num("ismethod(c, 'hidden')").item()
    m.run("p = properties(c); mm = methods(c);")
    assert "r" in [m.str(f"p{{{k}}}") for k in range(1, int(m.num('numel(p)').item()) + 1)]
    out = m.run("delete(c)")
    assert out == "bye\n"
    assert not m.num("isvalid(d)").item()
    with pytest.raises(MatlabError, match="deleted"):
        m.run("d.r")
    m.run("a = Vec2(1, 2); b = Vec2(1, 2);")
    assert m.num("isequal(a, b)").item()
