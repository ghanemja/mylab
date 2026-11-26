# ruff: noqa: E501  (the MATLAB programs below are kept as written, one statement group per line)
"""Class features: meta classes, mixin base classes, handle methods, access and class attributes, property validation, object arrays, enumerations, old-style classes, saving objects and inputParser.

Each case below is a small multi-file MATLAB program: the lines after ``@file name`` are written to that
file in a temporary folder and the lines after ``@main`` are run there as a script. A case passes when
the script reaches its end without an error; the expected values in the asserts were worked out by hand
from MATLAB's documented behaviour.
"""

import os

import pytest

from mylab import Interpreter

CASES = r"""
### meta/basic
@file Animal.m
classdef Animal < handle
    properties
        name = 'x'
    end
    properties (SetAccess = private, Hidden)
        id = 7
    end
    properties (Dependent)
        upper
    end
    properties (Constant)
        Kingdom = 'animalia'
    end
    events
        Spoke
    end
    methods
        function o = Animal(n), if nargin, o.name = n; end, end
        function u = get.upper(o), u = upper(o.name); end
        function s = speak(o), s = 'hi'; end
    end
    methods (Access = protected)
        function r = secret(o), r = 1; end
    end
    methods (Static)
        function r = make(), r = Animal('m'); end
    end
    methods (Hidden)
        function r = hid(o), r = 2; end
    end
end
@file Dog.m
classdef (Sealed) Dog < Animal
    properties
        tricks = {}
    end
    methods
        function s = speak(o), s = 'woof'; end
    end
end
@file Kind.m
classdef Kind
    enumeration
        A, B
    end
end
@main
mc = ?Dog; assert(strcmp(mc.Name, 'Dog')); assert(mc.Sealed && ~mc.Abstract && mc.HandleCompatible);
assert(isequal(superclasses('Dog'), {'Animal'; 'handle'})); assert(isequal(superclasses(Dog()), {'Animal'; 'handle'}));
assert(isempty(superclasses('Kind')));
pl = mc.PropertyList; assert(numel(pl) == 5); names = {pl.Name}; assert(isequal(names, {'tricks', 'name', 'id', 'upper', 'Kingdom'}));
p = findobj(pl, 'Name', 'id'); assert(strcmp(p.SetAccess, 'private') && strcmp(p.GetAccess, 'public') && p.Hidden);
u = pl(4); assert(u.Dependent && ~u.Constant && ~u.HasDefault); k = pl(5); assert(k.Constant && k.HasDefault && strcmp(k.DefaultValue, 'animalia'));
assert(strcmp(pl(2).DefiningClass.Name, 'Animal')); assert(isequal(pl(1).DefaultValue, {}));
ml = mc.MethodList; mn = {ml.Name}; assert(all(ismember({'speak', 'secret', 'make', 'hid', 'Dog', 'delete'}, mn)));
sp = findobj(ml, 'Name', 'speak'); assert(strcmp(sp.DefiningClass.Name, 'Dog') && strcmp(sp.Access, 'public') && ~sp.Static);
sec = findobj(ml, 'Name', 'secret'); assert(strcmp(sec.Access, 'protected')); mk = findobj(ml, 'Name', 'make'); assert(mk.Static);
assert(isequal(sp.InputNames, {'o'}) && isequal(sp.OutputNames, {'s'}));
ev = {mc.EventList.Name}; assert(any(strcmp(ev, 'Spoke')) && any(strcmp(ev, 'ObjectBeingDestroyed')));
sl = mc.SuperclassList; assert(numel(sl) == 1 && strcmp(sl(1).Name, 'Animal'));
assert(mc == ?Dog && mc ~= ?Animal); assert(?Dog < ?Animal && ?Animal > ?Dog && ~(?Animal < ?Dog));
m2 = metaclass(Dog()); assert(m2 == mc); assert(strcmp(class(mc), 'meta.class'));
km = ?Kind; assert(km.Enumeration && numel(km.EnumerationMemberList) == 2 && strcmp(km.EnumerationMemberList(2).Name, 'B'));
assert(isempty(meta.class.fromName('NoSuchClassXYZ'))); assert(strcmp(meta.class.fromName('Animal').Name, 'Animal'));
d = Dog(); assert(isequal(properties(d), {'tricks'; 'name'; 'upper'; 'Kingdom'})); assert(isequal(properties('Dog'), properties(d)));
ms = methods(d); assert(all(ismember({'speak', 'make', 'Dog', 'addlistener', 'delete', 'isvalid'}, ms)) && ~any(strcmp(ms, 'secret')) && ~any(strcmp(ms, 'hid')));
assert(isprop(d, 'name') && isprop(d, 'id') && ~isprop(d, 'nope')); assert(ismethod(d, 'speak') && ~ismethod(d, 'secret') && ~ismethod(d, 'nope'));
assert(isequal(events('Dog'), {'Spoke'; 'ObjectBeingDestroyed'}));
fp = findprop(d, 'name'); assert(strcmp(fp.Name, 'name')); assert(isempty(findprop(d, 'zzz')));
assert(isa(d, 'Animal') && isa(d, 'handle') && ~isa(d, 'Kind')); assert(isa(5, 'numeric') && isa(single(1), 'float') && isa(int8(1), 'integer') && ~isa(5, 'integer'));
mi = metaclass(5); assert(strcmp(mi.Name, 'double'));
validateattributes(d, {'Animal'}, {'scalar'}); validateattributes(d, {'handle'}, {'nonempty'});
t = false; try, validateattributes(d, {'Kind'}, {}); catch, t = true; end; assert(t);

### meta/packaged
@file +shp/Box.m
classdef Box
    properties
        w = 2
    end
end
@file +shp/area.m
function a = area(x)
a = x;
end
@main
b = shp.Box(); assert(strcmp(class(b), 'shp.Box')); mc = metaclass(b); assert(strcmp(mc.Name, 'shp.Box'));
assert(strcmp(mc.ContainingPackage.Name, 'shp')); assert(isa(b, 'shp.Box')); m2 = ?shp.Box; assert(m2 == mc);
pk = meta.package.fromName('shp'); assert(strcmp(pk.ClassList(1).Name, 'shp.Box')); assert(strcmp(pk.FunctionList(1).Name, 'area'));

### mix/copyable
@file Node.m
classdef Node < matlab.mixin.Copyable
    properties
        v = 1
        child
    end
    properties (NonCopyable)
        cache = 5
    end
end
@file Deep.m
classdef Deep < matlab.mixin.Copyable
    properties
        v = 1
        kid
    end
    methods (Access = protected)
        function cp = copyElement(obj)
            cp = copyElement@matlab.mixin.Copyable(obj);
            if ~isempty(obj.kid), cp.kid = copy(obj.kid); end
        end
    end
end
@main
n = Node(); n.cache = 9; n.child = Node(); m = copy(n); m.v = 5; assert(n.v == 1); k = n; k.v = 7; assert(n.v == 7);
assert(m.child == n.child); assert(m.cache == 5 && n.cache == 9); assert(isa(n, 'matlab.mixin.Copyable') && isa(n, 'handle'));
d = Deep(); d.kid = Deep(); d.kid.v = 3; e = copy(d); assert(e.kid ~= d.kid && e.kid.v == 3); e.kid.v = 4; assert(d.kid.v == 3);
arr = [Node(), Node()]; c = copy(arr); assert(numel(c) == 2 && c(1) ~= arr(1)); m3 = n.copy(); assert(m3 ~= n && m3.v == 7);

### mix/setget
@file Opt.m
classdef Opt < matlab.mixin.SetGet
    properties
        Gain = 1
        Name = 'n'
    end
    properties (SetAccess = private)
        Locked = 3
    end
    methods
        function set.Gain(o, v)
            if v < 0, error('Opt:neg', 'neg'); end
            o.Gain = v;
        end
    end
end
@file Exact.m
classdef Exact < matlab.mixin.SetGetExactNames
    properties
        Gain = 1
    end
end
@file Old.m
classdef Old < hgsetget
    properties
        A = 1
    end
end
@main
o = Opt(); set(o, 'Gain', 4); assert(get(o, 'Gain') == 4); set(o, 'Gain', 2, 'Name', 'z'); assert(strcmp(o.Name, 'z'));
set(o, 'gain', 6); assert(o.Gain == 6); set(o, 'Na', 'q'); assert(strcmp(get(o, 'name'), 'q'));
s = struct('Gain', 8, 'Name', 'st'); set(o, s); assert(o.Gain == 8 && strcmp(o.Name, 'st'));
set(o, {'Gain', 'Name'}, {9, 'c'}); assert(o.Gain == 9 && strcmp(o.Name, 'c'));
g = get(o); assert(isstruct(g) && g.Gain == 9 && g.Locked == 3); v = get(o, {'Gain', 'Name'}); assert(isequal(v, {9, 'c'}));
t = false; try, set(o, 'Gain', -1); catch e, t = strcmp(e.identifier, 'Opt:neg'); end; assert(t);
t = false; try, set(o, 'Nope', 1); catch, t = true; end; assert(t); t = false; try, set(o, 'Locked', 1); catch, t = true; end; assert(t);
o.set('Gain', 11); assert(o.get('Gain') == 11);
x = Exact(); set(x, 'Gain', 2); t = false; try, set(x, 'gain', 3); catch, t = true; end; assert(t && x.Gain == 2);
h = Old(); set(h, 'A', 5); assert(get(h, 'A') == 5);
two = [Opt(), Opt()]; set(two, 'Gain', 3); assert(isequal(get(two, 'Gain'), {3; 3}));

### mix/dynamicprops
@file Dyn.m
classdef Dyn < dynamicprops
    properties
        base = 1
    end
end
@main
d = Dyn(); p = addprop(d, 'extra'); d.extra = 7; assert(d.extra == 7); assert(isprop(d, 'extra')); assert(strcmp(class(p), 'meta.DynamicProperty'));
assert(strcmp(p.Name, 'extra')); assert(any(strcmp(properties(d), 'extra'))); d2 = Dyn(); assert(~isprop(d2, 'extra'));
t = false; try, d2.extra = 1; catch, t = true; end; assert(t); t = false; try, addprop(d, 'base'); catch, t = true; end; assert(t);
q = d.addprop('ro'); d.ro = 3; q.SetAccess = 'private'; t = false; try, d.ro = 4; catch, t = true; end; assert(t && d.ro == 3);
assert(findprop(d, 'extra') == p); delete(p); assert(~isprop(d, 'extra')); t = false; try, x = d.extra; catch, t = true; end; assert(t);
h = addprop(d, 'hid'); h.Hidden = true; assert(~any(strcmp(properties(d), 'hid')) && isprop(d, 'hid'));
s = evalc('disp(d)'); assert(contains(s, 'ro') && contains(s, 'base') && ~contains(s, 'hid'));

### mix/heterogeneous
@file Base.m
classdef Base < matlab.mixin.Heterogeneous
    properties
        n = 0
    end
    methods
        function v = id(~), v = 0; end
    end
    methods (Sealed)
        function v = ids(arr)
            v = zeros(1, numel(arr));
            for k = 1:numel(arr), v(k) = arr(k).id(); end
        end
    end
    methods (Static, Sealed, Access = protected)
        function d = getDefaultScalarElement()
            d = D1();
        end
    end
end
@file D1.m
classdef D1 < Base
    methods
        function v = id(~), v = 1; end
    end
end
@file D2.m
classdef D2 < Base
    methods
        function v = id(~), v = 2; end
    end
end
@file Other.m
classdef Other
end
@main
a = [D1(), D2()]; assert(isa(a, 'Base')); assert(numel(a) == 2); assert(a(2).id() == 2); assert(strcmp(class(a), 'Base'));
assert(strcmp(class(a(1)), 'D1')); assert(isequal(a.ids(), [1 2])); assert(isequal(ids(a), [1 2]));
t = false; try, a.id(); catch, t = true; end; assert(t); b = [a, Base()]; assert(numel(b) == 3 && isequal(ids(b), [1 2 0]));
h = [D1(), D1()]; assert(strcmp(class(h), 'D1')); assert(isequal(h.id(), 1));
c = D2(); c(4) = D2(); assert(numel(c) == 4 && strcmp(class(c(2)), 'D1') && strcmp(class(c), 'Base'));
v = a; v2 = v; v2(1).n = 5; assert(v(1).n == 0); assert(~isa(a, 'handle'));
t = false; try, z = [D1(), Other()]; catch, t = true; end; assert(t);
s = evalc('disp(a)'); assert(contains(s, 'heterogeneous') && contains(s, 'D1') && contains(s, 'D2'));

### mix/customdisplay
@file Emp.m
classdef Emp < matlab.mixin.CustomDisplay
    properties
        Name = 'Al'
        Salary = 10
        Secret = 1
    end
    methods (Access = protected)
        function g = getPropertyGroups(obj)
            if isscalar(obj)
                g = matlab.mixin.util.PropertyGroup({'Name', 'Salary'}, 'Public Info');
            else
                g = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
            end
        end
        function f = getFooter(~)
            f = 'the footer';
        end
    end
end
@file Hdr.m
classdef Hdr < matlab.mixin.CustomDisplay
    properties
        A = 1
    end
    methods (Access = protected)
        function h = getHeader(obj)
            h = ['custom ' matlab.mixin.CustomDisplay.getClassNameForHeader(obj)];
        end
        function displayScalarObject(obj)
            disp(getHeader(obj)); g = getPropertyGroups(obj); matlab.mixin.CustomDisplay.displayPropertyGroups(obj, g);
        end
    end
end
@file Sv.m
classdef Sv < matlab.mixin.CustomDisplay
    properties
        A = 1
        B = 2
    end
    methods (Access = protected)
        function g = getPropertyGroups(obj)
            s = struct('Total', obj.A + obj.B);
            g = matlab.mixin.util.PropertyGroup(s);
        end
    end
end
@main
e = Emp(); s = evalc('disp(e)'); assert(contains(s, 'Emp with properties') && contains(s, 'Public Info') && contains(s, 'Salary: 10') && ~contains(s, 'Secret') && contains(s, 'the footer'));
s2 = evalc('e'); assert(contains(s2, 'e =') && contains(s2, 'Name: ''Al'''));
h = Hdr(); s = evalc('disp(h)'); assert(contains(s, 'custom Hdr') && contains(s, 'A: 1'));
v = Sv(); s = evalc('v'); assert(contains(s, 'Total: 3') && ~contains(s, 'B:'));
arr = [Emp(), Emp()]; s = evalc('disp(arr)'); assert(contains(s, '1×2 Emp array') && contains(s, 'Secret'));

### hdl/handle_methods
@file Btn.m
classdef Btn < handle
    properties (SetObservable, GetObservable)
        val = 0
    end
    properties (SetObservable, AbortSet)
        quiet = 1
    end
    properties
        log = {}
        tag = ''
    end
    events
        Pressed
    end
    methods
        function o = Btn(t), if nargin, o.tag = t; end, end
        function press(o, d)
            if nargin > 1, notify(o, 'Pressed', d); else, notify(o, 'Pressed'); end
        end
        function delete(o)
            global DELETED; DELETED = [DELETED {o.tag}];
        end
    end
end
@file Payload.m
classdef (ConstructOnLoad) Payload < event.EventData
    properties
        Value
    end
    methods
        function o = Payload(v), if nargin, o.Value = v; end, end
    end
end
@file note.m
function note(b, what)
b.log{end+1} = what;
end
@main
global DELETED; DELETED = {};
b = Btn('b1'); L = addlistener(b, 'Pressed', @(src, evt) note(src, evt.EventName)); b.press(); assert(isequal(b.log, {'Pressed'}));
L2 = listener(b, 'Pressed', @(src, evt) note(src, evt.Value)); b.press(Payload(42)); assert(isequal(b.log(end), {42}));
L.Enabled = false; delete(L2); b.press(); assert(numel(b.log) == 3);
P1 = addlistener(b, 'val', 'PreSet', @(mp, evt) note(evt.AffectedObject, ['pre ' mp.Name]));
P2 = addlistener(b, 'val', 'PostSet', @(mp, evt) note(evt.AffectedObject, evt.AffectedObject.val)); b.val = 3;
assert(isequal(b.log(end-1:end), {'pre val', 3}));
G = addlistener(b, 'val', 'PostGet', @(mp, evt) note(evt.AffectedObject, 'got')); x = b.val; assert(strcmp(b.log{end}, 'got'));
delete(G); delete(P1); delete(P2);
n0 = numel(b.log); Q = addlistener(b, 'quiet', 'PostSet', @(mp, evt) note(evt.AffectedObject, 'q')); b.quiet = 1; assert(numel(b.log) == n0); b.quiet = 2; assert(numel(b.log) == n0 + 1);
D = addlistener(b, 'ObjectBeingDestroyed', @(src, evt) assignin('base', 'gone', 1));
c = Btn('c1'); arr = [b, c]; f = findobj(arr, 'tag', 'c1'); assert(f == c); assert(isempty(findobj(arr, 'tag', 'zz')));
assert(numel(findobj(arr, '-isa', 'Btn')) == 2); assert(isequal(isvalid(arr), [true true])); assert(isequal(arr == b, [true false]));
assert(eq(b, b) && ne(b, c)); assert(isequal(b, b) && ~isequal(b, c));
delete(b); assert(gone == 1); assert(isequal(DELETED, {'b1'})); assert(~isvalid(b) && isvalid(c)); t = false; try, b.val = 1; catch e, t = contains(e.message, 'deleted'); end; assert(t);
s = evalc('disp(b)'); assert(contains(s, 'deleted')); delete(b); assert(numel(DELETED) == 1);
e1 = event.EventData; assert(isa(e1, 'event.EventData'));

### attr/access_lists
@file Vault.m
classdef Vault
    properties (Access = ?Friend)
        gold = 5
    end
    properties (SetAccess = immutable)
        serial
    end
    properties (GetAccess = public, SetAccess = protected)
        level = 1
    end
    methods
        function o = Vault(s), if nargin, o.serial = s; end, end
    end
    methods (Access = {?Friend, ?Pal})
        function r = open(o), r = o.gold * 2; end
    end
    methods (Access = private, Static)
        function r = hiddenStatic(), r = 1; end
    end
end
@file Friend.m
classdef Friend
    methods
        function r = peek(~, v), r = v.gold + v.open(); end
    end
end
@file Pal.m
classdef Pal
    methods
        function r = peek(~, v), r = v.open(); end
        function r = steal(~, v), r = v.gold; end
    end
end
@file SubVault.m
classdef SubVault < Vault
    methods
        function o = up(o), o.level = o.level + 1; end
    end
end
@file Single.m
classdef Single < handle
    properties
        n = 0
    end
    methods (Access = private)
        function o = Single(), end
    end
    methods (Static)
        function o = instance()
            persistent inst
            if isempty(inst) || ~isvalid(inst), inst = Single(); end
            o = inst;
        end
    end
end
@main
v = Vault(9); assert(Friend().peek(v) == 15); assert(Pal().peek(v) == 10);
t = false; try, Pal().steal(v); catch, t = true; end; assert(t); t = false; try, v.gold; catch, t = true; end; assert(t);
t = false; try, v.open(); catch, t = true; end; assert(t); assert(v.serial == 9); t = false; try, v.serial = 1; catch, t = true; end; assert(t);
t = false; try, v.level = 3; catch, t = true; end; assert(t); s = SubVault(); s = s.up(); assert(s.level == 2);
t = false; try, Vault.hiddenStatic(); catch, t = true; end; assert(t);
a = Single.instance(); b = Single.instance(); a.n = 4; assert(b.n == 4 && a == b); t = false; try, Single(); catch, t = true; end; assert(t);

### attr/class_attrs
@file Fin.m
classdef (Sealed) Fin
end
@file Bad.m
classdef Bad < Fin
end
@file Par.m
classdef (AllowedSubclasses = {?Ok}) Par
end
@file Ok.m
classdef Ok < Par
end
@file No.m
classdef No < Par
end
@file Abs.m
classdef Abs
    properties (Abstract)
        p
    end
    methods (Abstract)
        r = f(o)
    end
    methods (Sealed)
        function r = g(o), r = 1; end
    end
end
@file Con.m
classdef Con < Abs
    properties
        p = 3
    end
    methods
        function r = f(o), r = o.p; end
    end
end
@file Half.m
classdef Half < Abs
    methods
        function r = f(o), r = 1; end
    end
end
@file Over.m
classdef Over < Abs
    properties
        p = 1
    end
    methods
        function r = f(o), r = 1; end
        function r = g(o), r = 2; end
    end
end
@main
t = false; try, Bad(); catch, t = true; end; assert(t); Ok(); t = false; try, No(); catch, t = true; end; assert(t);
c = Con(); assert(c.f() == 3 && c.g() == 1); t = false; try, Abs(); catch, t = true; end; assert(t);
t = false; try, Half(); catch, t = true; end; assert(t); t = false; try, Over(); catch, t = true; end; assert(t);
m = ?Abs; assert(m.Abstract); m2 = ?Fin; assert(m2.Sealed);

### attr/validation
@file Cfg.m
classdef Cfg
    properties
        n (1,1) double {mustBePositive, mustBeInteger} = 1
        name (1,:) char = 'x'
        mode {mustBeMember(mode, {'a', 'b'})} = 'a'
        row (1,:) double = [1 2 3]
        col (:,1) double
        three (1,3) double = [0 0 0]
        s (1,1) string = "s"
        flag (1,1) logical = false
        k int8 = 5
        names (1,:) string = ["a" "b"]
        c cell = {}
        st struct
        fh function_handle = @sin
        col2 (:,1) cell
        e (1,1) Lvl = Lvl.Lo
        any2 {mustBeNumeric, mustBeFinite}
        rng (1,2) double {mustBeNonnegative} = [0 1]
        txt {mustBeTextScalar} = 'a'
        obj (1,1) Inner
    end
end
@file Lvl.m
classdef Lvl
    enumeration
        Lo, Hi
    end
end
@file Inner.m
classdef Inner
    properties
        v = 1
    end
    methods
        function o = Inner(v), if nargin, o.v = v; end, end
    end
end
@main
c = Cfg(); c.n = 5; threw = false; try, c.n = -1; catch, threw = true; end; assert(threw, 'mustBePositive');
threw = false; try, c.n = [1 2]; catch, threw = true; end; assert(threw, 'size'); c.n = int8(3); assert(isa(c.n, 'double'));
threw = false; try, c.mode = 'z'; catch, threw = true; end; assert(threw, 'mustBeMember');
c.row = [4; 5; 6]; assert(isequal(c.row, [4 5 6])); c.col = [1 2 3]; assert(isequal(c.col, [1; 2; 3])); assert(isequal(size(Cfg().col), [0 1]));
c.three = 7; assert(isequal(c.three, [7 7 7])); c.three = [1; 2; 3]; assert(isequal(c.three, [1 2 3])); threw = false; try, c.three = [1 2]; catch, threw = true; end; assert(threw);
c.name = "str"; assert(ischar(c.name) && strcmp(c.name, 'str')); c.s = 'chr'; assert(isstring(c.s) && c.s == "chr"); c.s = 5; assert(c.s == "5");
c.flag = 1; assert(islogical(c.flag) && c.flag); c.k = 300; assert(isa(c.k, 'int8') && c.k == 127); c.names = {'p', 'q'}; assert(isstring(c.names) && numel(c.names) == 2);
threw = false; try, c.c = 5; catch, threw = true; end; assert(threw); c.c = {1, 2}; threw = false; try, c.fh = 5; catch, threw = true; end; assert(threw);
c.e = 'Hi'; assert(c.e == Lvl.Hi); c.e = "Lo"; assert(c.e == Lvl.Lo); threw = false; try, c.e = 'Mid'; catch, threw = true; end; assert(threw);
c.any2 = 5; threw = false; try, c.any2 = Inf; catch, threw = true; end; assert(threw); threw = false; try, c.any2 = 'a'; catch, threw = true; end; assert(threw);
threw = false; try, c.rng = [-1 2]; catch, threw = true; end; assert(threw); c.rng = [2; 3]; assert(isequal(c.rng, [2 3]));
d = Cfg(); assert(d.n == 1 && isequal(d.three, [0 0 0]) && isa(d.k, 'int8') && d.k == 5 && isequal(d.c, {}) && isempty(d.any2));
assert(isa(d.obj, 'Inner') && d.obj.v == 1); d.obj = 9; assert(isa(d.obj, 'Inner') && d.obj.v == 9); assert(isstruct(d.st));
threw = false; try, c.txt = 5; catch e, threw = contains(e.message, 'txt'); end; assert(threw);
mustBeScalarOrEmpty([]); mustBeRow([1 2]); mustBeColumn([1; 2]); mustBeFloat(single(1)); mustBeNonmissing([1 2]); mustBeUnderlyingType(1, 'double'); mustBeValidVariableName('abc');
threw = false; try, mustBeRow([1; 2]); catch, threw = true; end; assert(threw); threw = false; try, mustBeNonmissing([1 NaN]); catch, threw = true; end; assert(threw);
mustBeInRange(5, 1, 10); threw = false; try, mustBeInRange(10, 1, 10, 'exclusive'); catch, threw = true; end; assert(threw); mustBeInRange(10, 1, 10, 'exclude-lower');

### attr/defaults_once
@file Tick.m
classdef Tick < handle
    properties
        stamp = Tick.next()
        shared = Counter()
    end
    properties (Constant)
        Reg = Counter()
    end
    methods (Static)
        function v = next()
            persistent k
            if isempty(k), k = 0; end
            k = k + 1; v = k;
        end
        function out = cfg(val)
            persistent V
            if nargin, V = val; end
            out = V;
        end
    end
end
@file Counter.m
classdef Counter < handle
    properties
        n = 0
    end
    methods
        function bump(o), o.n = o.n + 1; end
    end
end
@main
a = Tick(); b = Tick(); assert(a.stamp == 1 && b.stamp == 1); assert(a.shared == b.shared); a.shared.bump(); assert(b.shared.n == 1);
Tick.Reg.bump(); Tick.Reg.bump(); assert(Tick.Reg.n == 2); r = Tick.Reg; r.bump(); assert(Tick.Reg.n == 3); assert(a.Reg.n == 3);
Tick.cfg(7); assert(Tick.cfg() == 7);

### arr/object_arrays
@file Pt.m
classdef Pt
    properties
        x = 0
        y = 0
    end
    methods
        function obj = Pt(x, y)
            if nargin > 0, obj.x = x; obj.y = y; end
        end
        function r = norms(objs)
            r = arrayfun(@(o) hypot(o.x, o.y), objs);
        end
        function r = last(objs)
            r = objs(end);
        end
        function objs = grow(objs)
            objs(end + 1) = Pt(-1, -1);
        end
    end
end
@file Grid.m
classdef Grid
    properties
        v = 1
    end
    methods
        function obj = Grid(n)
            if nargin > 0
                obj(n) = Grid();
                for k = 1:n, obj(k).v = k; end
            end
        end
    end
end
@main
objs = [Pt(1, 2), Pt(3, 4)]; assert(numel(objs) == 2); assert(isequal([objs.x], [1 3])); objs(3) = Pt(5, 6); assert(objs(3).y == 6);
objs(2).x = 30; assert(isequal([objs.x], [1 30 5])); objs(5).x = 9; assert(numel(objs) == 5 && objs(4).x == 0 && objs(5).x == 9 && objs(5).y == 0);
[objs.y] = deal(7); assert(all([objs.y] == 7)); objs(2) = []; assert(numel(objs) == 4); assert(objs(end).x == 9); assert(objs.last().x == 9);
g = objs.grow(); assert(numel(g) == 5 && g(end).x == -1); assert(isequal(norms(Pt(3, 4)), 5));
e = Pt.empty; assert(isempty(e) && isequal(size(e), [0 0])); e2 = Pt.empty(0, 3); assert(isequal(size(e2), [0 3])); e(1) = Pt(1, 1); assert(numel(e) == 1);
m = [Pt(1, 1), Pt(2, 2); Pt(3, 3), Pt(4, 4)]; assert(isequal(size(m), [2 2])); assert(m(2, 1).x == 3); assert(isequal([m(:, 2).x], [2 4])); assert(m(end, end).x == 4);
t = m'; assert(t(1, 2).x == 3); col = m(:); assert(isequal(size(col), [4 1]));
k = objs([objs.x] > 5); assert(numel(k) == 1); g2 = Grid(3); assert(numel(g2) == 3 && isequal([g2.v], [1 2 3]));
c = num2cell(objs); assert(iscell(c) && numel(c) == 4); s = 0; for o = objs, s = s + o.x; end; assert(s == 1 + 5 + 0 + 9);
z(3) = Pt(1, 1); assert(numel(z) == 3 && z(1).x == 0); r = repmat(Pt(2, 2), 2, 2); assert(isequal(size(r), [2 2]));
assert(isequal(Pt(1, 2), Pt(1, 2)) && ~isequal(Pt(1, 2), Pt(1, 3))); assert(isequal(objs, objs));
st = struct(Pt(4, 5)); assert(st.x == 4 && st.y == 5); assert(isequal(fieldnames(Pt()), {'x'; 'y'}));

### enum/methods
@file Lvl.m
classdef Lvl < uint8
    enumeration
        Low (1)
        Mid (5)
        High (10)
    end
    methods
        function r = isHigh(o), r = o == Lvl.High; end
        function r = label(o), r = lower(char(o)); end
    end
    methods (Static)
        function r = default(), r = Lvl.Mid; end
    end
end
@file Wk.m
classdef Wk
    properties
        short
    end
    methods
        function o = Wk(s), o.short = s; end
        function r = isWeekend(o), r = any(strcmp(o.short, {'sa', 'su'})); end
    end
    enumeration
        Mon ('mo'), Sat ('sa'), Sun ('su')
    end
end
@main
h = Lvl.High; assert(h.isHigh() && ~Lvl.Low.isHigh()); assert(strcmp(h.label(), 'high')); assert(Lvl.default() == Lvl.Mid);
assert(double(h) == 10 && uint8(Lvl.Mid) == 5); assert(h > Lvl.Low); assert(Lvl(5) == Lvl.Mid); assert(Lvl('Low') == Lvl.Low);
[m, n] = enumeration('Lvl'); assert(numel(m) == 3 && isequal(n, {'Low'; 'Mid'; 'High'})); assert(m(3) == Lvl.High);
assert(Wk.Sat.isWeekend() && ~Wk.Mon.isWeekend()); assert(strcmp(Wk.Sun.short, 'su'));
a = [Lvl.Low, Lvl.High]; assert(isequal(a == Lvl.High, [false true])); assert(isenum(a) && strcmp(class(a), 'Lvl'));
s = string(Lvl.Mid); assert(s == "Mid"); assert(ismember(Lvl.Mid, [Lvl.Mid, Lvl.High]));
x = 0; switch h, case Lvl.Low, x = 1; case {Lvl.Mid, Lvl.High}, x = 2; end; assert(x == 2);
e = enumeration(Lvl.Low); assert(numel(e) == 3); c = cellfun(@char, num2cell(m'), 'UniformOutput', false); assert(isequal(c, {'Low', 'Mid', 'High'}));

### dep/dependent
@file Rect.m
classdef Rect
    properties
        w = 2
        h = 3
    end
    properties (Dependent)
        area
    end
    properties (Dependent, SetAccess = private)
        perim
    end
    methods
        function a = get.area(o), a = o.w * o.h; end
        function p = get.perim(o), p = 2 * (o.w + o.h); end
    end
end
@main
r = Rect(); assert(r.area == 6 && r.perim == 10); r.w = 5; assert(r.area == 15);
t = false; try, r.area = 3; catch, t = true; end; assert(t); t = false; try, r.perim = 3; catch, t = true; end; assert(t);
s = evalc('disp(r)'); assert(contains(s, 'area: 15'));

### old/old_style
@file @oldc/oldc.m
function o = oldc(v)
if nargin == 0, v = 0; end
if isa(v, 'oldc'), o = v; return; end
s.v = v; o = class(s, 'oldc'); superiorto('double');
end
@file @oldc/double.m
function d = double(o)
d = o.v;
end
@file @oldc/plus.m
function r = plus(a, b)
r = oldc(double(a) + double(b));
end
@file @oldc/display.m
function display(o)
fprintf('oldc(%d)\n', o.v);
end
@file @oldc/get.m
function v = get(o, name)
v = o.(name);
end
@file @oldc/set.m
function o = set(o, name, val)
o.(name) = val;
end
@file @oldc/private/helper.m
function r = helper(x)
r = x * 100;
end
@file @oldc/big.m
function r = big(o)
r = helper(o.v);
end
@file @kid/kid.m
function k = kid(v, extra)
s.extra = extra; k = class(s, 'kid', oldc(v));
end
@file @kid/extraval.m
function r = extraval(k)
r = k.extra;
end
@file @double/twice.m
function r = twice(x)
r = 2 * x;
end
@main
a = oldc(2) + oldc(3); assert(double(a) == 5); assert(isa(a, 'oldc')); assert(isobject(a)); assert(strcmp(class(a), 'oldc'));
b = a + 1; assert(double(b) == 6); c = 1 + a; assert(double(c) == 6 && isa(c, 'oldc'));
t = false; try, x = a.v; catch, t = true; end; assert(t, 'fields are private'); assert(get(a, 'v') == 5); a = set(a, 'v', 8); assert(double(a) == 8);
s = evalc('a'); assert(strcmp(strtrim(s), 'oldc(8)')); st = struct(a); assert(st.v == 8); assert(a.big() == 800 || big(a) == 800);
m = methods('oldc'); assert(all(ismember({'double', 'plus', 'display', 'oldc'}, m)));
k = kid(4, 'e'); assert(isa(k, 'kid') && isa(k, 'oldc')); assert(double(k) == 4); assert(strcmp(extraval(k), 'e')); s2 = k + k; assert(double(s2) == 8);
assert(twice(4) == 8); assert(isequal(twice([1 2]), [2 4]));

### io/save_load
@file Pt.m
classdef Pt
    properties
        x = 0
        y = 0
    end
    properties (Transient)
        tmp = 1
    end
end
@file H.m
classdef H < handle
    properties
        v = 1
        peer
    end
end
@file Sv.m
classdef Sv
    properties
        data = [1 2 3]
        loaded = false
    end
    methods
        function s = saveobj(o)
            s.d = o.data * 2;
        end
    end
    methods (Static)
        function o = loadobj(s)
            o = Sv(); o.data = s.d / 2; o.loaded = true;
        end
    end
end
@main
p = Pt(); p.y = 2; p.tmp = 9; save('obj.mat', 'p'); clear p; S = load('obj.mat'); assert(isa(S.p, 'Pt') && S.p.y == 2 && S.p.tmp == 1);
a = H(); b = H(); a.peer = b; b.v = 5; arr = [a, b]; q = Sv(); q.data = [4 5]; x = 3;
save('two.mat', 'arr', 'q', 'x'); clear arr q x; load('two.mat'); assert(x == 3); assert(numel(arr) == 2 && arr(1).peer == arr(2) && arr(2).v == 5);
assert(isequal(q.data, [4 5]) && q.loaded);

### ip/input_parser
p = inputParser; addRequired(p, 'n', @isnumeric); addOptional(p, 'k', 2, @isnumeric); addParameter(p, 'Mode', 'fast', @ischar);
parse(p, 5, 3, 'Mode', 'slow'); r = p.Results; assert(r.n == 5 && r.k == 3 && strcmp(r.Mode, 'slow'));
parse(p, 1); assert(p.Results.k == 2 && any(strcmp(p.UsingDefaults, 'Mode')));
p.KeepUnmatched = true; parse(p, 1, 'Other', 9); assert(p.Unmatched.Other == 9);
t = false; try, parse(p, 'x'); catch, t = true; end; assert(t);
p2 = inputParser; p2.addParameter('a', 1); p2.parse('a', 4); assert(p2.Results.a == 4);
q = inputParser; q.FunctionName = 'myfn'; addParamValue(q, 'Tol', 1e-3, @(x) x > 0); addParameter(q, 'Verbose', false); parse(q, 'tol', 0.5); assert(q.Results.Tol == 0.5);
parse(q, 'Verb', true); assert(q.Results.Verbose && isequal(sort(q.Parameters), {'Tol', 'Verbose'}) && isequal(q.UsingDefaults, {'Tol'}));
t = false; try, parse(q, 'Tol', -1); catch e, t = contains(e.message, 'Tol'); end; assert(t);
q.CaseSensitive = true; t = false; try, parse(q, 'tol', 1); catch, t = true; end; assert(t); q.CaseSensitive = false;
q.PartialMatching = false; t = false; try, parse(q, 'Verb', true); catch, t = true; end; assert(t); q.PartialMatching = true;
s.Tol = 2; parse(q, s); assert(q.Results.Tol == 2); q.StructExpand = false; t = false; try, parse(q, s); catch, t = true; end; assert(t);
t = false; try, parse(q, 'Nope', 1); catch e, t = true; end; assert(t);
w = inputParser; addRequired(w, 'a'); addOptional(w, 'b', 'dflt'); addParameter(w, 'c', 0); parse(w, 1, 'c', 5); assert(strcmp(w.Results.b, 'dflt') && w.Results.c == 5);
t = false; try, parse(w, 1, 'given'); catch, t = true; end; assert(t); parse(w, 1, 'b', 'given', 'c', 2); assert(strcmp(w.Results.b, 'given') && w.Results.c == 2);

### cls/misc_semantics
@file Acc.m
classdef Acc < handle
    properties
        total = 0
    end
    properties (SetAccess = private)
        count = 0
    end
    methods
        function o = add(o, v)
            o.total = o.total + v; o.count = o.count + 1;
        end
        function o = set.total(o, v)
            assert(isnumeric(v)); o.total = v;
        end
        function r = plus(a, b)
            r = Acc(); r.total = a.total + b.total;
        end
        function tf = lt(a, b), tf = a.total < b.total; end
        function s = char(o), s = sprintf('Acc(%g)', o.total); end
        function disp(o), disp(char(o)); end
    end
end
@file Val.m
classdef Val
    properties
        x = 1
    end
    methods
        function o = bump(o), o.x = o.x + 1; end
        function o = set.x(o, v), o.x = v * 1; end
        function r = isequal(a, b), r = abs(a.x - b.x) < 0.5; end
    end
end
@file Base2.m
classdef Base2
    properties
        id = 0
    end
    methods
        function o = Base2(id), if nargin, o.id = id; end, end
        function r = who(o), r = 'base'; end
        function r = both(o), r = ['<' who(o) '>']; end
    end
end
@file Mid.m
classdef Mid < Base2
    methods
        function o = Mid(id), o@Base2(id * 2); end
        function r = who(o), r = ['mid+' who@Base2(o)]; end
    end
end
@file Leaf.m
classdef Leaf < Mid
    properties
        extra = 'e'
    end
    methods
        function o = Leaf(), o@Mid(5); end
    end
end
@file NoCtor.m
classdef NoCtor < Base2
end
@main
a = Acc(); a.add(2).add(3); assert(a.total == 5 && a.count == 2); b = a; b.add(1); assert(a.total == 6);
c = a + b; assert(c.total == 12 && c ~= a); assert(a < c); s = evalc('a'); assert(contains(s, 'Acc(6)')); assert(strcmp(char(a), 'Acc(6)'));
t = false; try, a.total = 'x'; catch, t = true; end; assert(t);
v = Val(); w = v.bump(); assert(v.x == 1 && w.x == 2); v2 = v; v2.x = 1.2; assert(isequal(v, v2) && ~isequal(v, w));
l = Leaf(); assert(l.id == 10 && strcmp(l.who(), 'mid+base') && strcmp(l.both(), '<mid+base>') && isa(l, 'Base2') && strcmp(l.extra, 'e'));
assert(isequal(superclasses(l), {'Mid'; 'Base2'})); n = NoCtor(7); assert(n.id == 7); n0 = NoCtor(); assert(n0.id == 0);
assert(isequal(properties(l), {'extra'; 'id'}));
h = [Acc(), Acc()]; h(2).add(4); assert(isequal([h.total], [0 4])); assert(isequal(h == h(2), [false true])); delete(h(1)); assert(isequal(isvalid(h), [false true]));
m = containers.Map(); m('k') = a; a.add(1); mk = m('k'); assert(mk.total == 7);
st.obj = a; st.obj.add(1); assert(a.total == 8); cl = {a}; cl{1}.add(1); assert(a.total == 9); a.total(1) = 20; assert(a.total == 20);

### cls/handle_cycles_and_isequal
@file Nd.m
classdef Nd < handle
    properties
        v
        next
    end
    methods
        function o = Nd(v), if nargin, o.v = v; end, end
    end
end
@main
a = Nd(1); b = Nd(2); a.next = b; b.next = a; c = Nd(1); d = Nd(2); c.next = d; d.next = c; assert(isequal(a, c)); assert(a ~= c);
e = Nd(1); assert(~isequal(a, e)); x = Nd(3); y = x; y.v = 4; assert(x.v == 4 && isequal(x, y) && x == y);
L = Nd(1); L.next = Nd(2); L.next.next = Nd(3); assert(L.next.next.v == 3); L.next.next.v = 30; assert(L.next.next.v == 30);
n = 0; p = L; while ~isempty(p), n = n + p.v; p = p.next; end; assert(n == 33);

### cls/static_and_constant
@file Util.m
classdef Util
    properties (Constant)
        Tol = 1e-6
        Names = {'a', 'b', 'c'}
        Tbl = struct('k', 5)
    end
    methods (Static)
        function r = dbl(x), r = 2 * x; end
        function r = viaConst(), r = Util.Tol * 2; end
        function r = chain(), r = Util.dbl(Util.dbl(1)); end
    end
end
@main
assert(Util.Tol == 1e-6); assert(strcmp(Util.Names{2}, 'b')); assert(Util.Tbl.k == 5); assert(Util.dbl(4) == 8); assert(Util.viaConst() == 2e-6 && Util.chain() == 4);
u = Util(); assert(u.Tol == 1e-6 && u.dbl(1) == 2); t = false; try, u.Tol = 1; catch, t = true; end; assert(t);
f = @Util.dbl; assert(f(3) == 6); assert(isequal(cellfun(@Util.dbl, {1, 2}), [2 4])); assert(numel(Util.Names) == 3);
mc = ?Util; p = findobj(mc.PropertyList, 'Name', 'Tol'); assert(p.Constant && p.DefaultValue == 1e-6);

### cls/enum_extra
@file Sz.m
classdef Sz < double
    enumeration
        Small (1.5), Large (10)
    end
end
@file St.m
classdef St
    enumeration
        On, Off
    end
    methods
        function r = flip(o)
            if o == St.On, r = St.Off; else, r = St.On; end
        end
    end
end
@main
assert(Sz.Small * 2 == 3); assert(Sz.Large > Sz.Small); assert(isa(Sz.Small, 'double')); x = [Sz.Small, Sz.Large]; assert(numel(x) == 2);
assert(St.On.flip() == St.Off); s = St('On'); assert(s == St.On); c = {St.On, St.Off}; assert(c{2} == St.Off);
m = containers.Map({'On', 'Off'}, {1, 0}); assert(m(char(St.On)) == 1); assert(strcmp(class(St.On), 'St'));
names = arrayfun(@(e) string(e), enumeration('St')); assert(isequal(names, ["On"; "Off"]));
assert(isequal(St.On, St.On) && ~isequal(St.On, St.Off)); t = evalc('disp(St.On)'); assert(contains(t, 'On'));

### cls/display_formats
@file P3.m
classdef P3
    properties
        x = 1
        name = 'abc'
        list = {1, 2}
    end
    properties (Hidden)
        h = 5
    end
    properties (Access = private)
        secret = 9
    end
end
@file E3.m
classdef E3
    enumeration
        Red, Green
    end
end
@file None3.m
classdef None3
end
@file H3.m
classdef H3 < handle
    properties
        v = 2
    end
end
@main
p = P3(); s = evalc('p'); assert(contains(s, 'p =') && contains(s, 'P3 with properties:') && contains(s, 'x: 1') && contains(s, 'name: ''abc''') && contains(s, 'list: {1×2 cell}'));
assert(~contains(s, 'secret') && ~contains(s, ' h:')); s = evalc('disp(p)'); assert(~contains(s, 'p ='));
arr = [P3(), P3(), P3()]; s = evalc('arr'); assert(contains(s, '1×3 P3 array with properties:') && contains(s, 'name'));
e = P3.empty(0, 1); s = evalc('e'); assert(contains(s, '0×1 empty P3 array'));
s = evalc('x = E3.Green'); assert(contains(s, 'E3 enumeration') && contains(s, 'Green')); s = evalc('y = [E3.Red, E3.Green]'); assert(contains(s, '1×2 E3 enumeration array') && contains(s, 'Red'));
s = evalc('n = None3()'); assert(contains(s, 'None3 with no properties'));
h = H3(); delete(h); s = evalc('h'); assert(contains(s, 'handle to deleted H3'));
c = {P3(), H3()}; s = evalc('c'); assert(contains(s, 'P3') && contains(s, 'H3')); st.a = P3(); s = evalc('st'); assert(contains(s, '1×1 P3'));
assert(strcmp(class(P3()), 'P3') && isobject(p) && ~isobject(5) && ~isstruct(p));

### cls/abstract_static_and_interfaces
@file Shape.m
classdef (Abstract) Shape
    properties (Abstract, Constant)
        Sides
    end
    methods (Abstract)
        a = area(obj)
    end
    methods (Abstract, Static)
        s = unit()
    end
    methods
        function r = describe(obj), r = sprintf('%d sides, area %g', obj.Sides, obj.area()); end
    end
end
@file Sq.m
classdef Sq < Shape
    properties (Constant)
        Sides = 4
    end
    properties
        w = 2
    end
    methods
        function a = area(o), a = o.w ^ 2; end
    end
    methods (Static)
        function s = unit(), s = Sq(); s.w = 1; end
    end
end
@file Printable.m
classdef (Abstract) Printable
    methods (Abstract)
        s = toText(o)
    end
end
@file Both.m
classdef Both < Sq & Printable
    methods
        function s = toText(o), s = ['Both:' o.describe()]; end
    end
end
@main
s = Sq(); assert(strcmp(s.describe(), '4 sides, area 4')); assert(Sq.unit().area() == 1); assert(Sq.Sides == 4);
t = false; try, Shape(); catch e, t = contains(e.message, 'abstract'); end; assert(t); t = false; try, Printable(); catch, t = true; end; assert(t);
b = Both(); assert(strcmp(b.toText(), 'Both:4 sides, area 4')); assert(isa(b, 'Printable') && isa(b, 'Shape') && isa(b, 'Sq'));
assert(isequal(sort(superclasses('Both')), sort({'Sq'; 'Printable'; 'Shape'}))); m = ?Shape; assert(m.Abstract);

### cls/save_load_more
@file Cfg2.m
classdef (ConstructOnLoad) Cfg2 < handle
    properties
        a = 1
        s = "txt"
        c = {1, 'two', [3 4]}
        st = struct('k', {{1, 2}}, 'z', 3)
        fh = @(x) x + 1
        kids = []
    end
    properties (Transient)
        loadedHere = false
    end
    properties (Dependent)
        twice
    end
    methods
        function o = Cfg2(), o.loadedHere = true; end
        function v = get.twice(o), v = 2 * o.a; end
    end
end
@file Kid.m
classdef Kid
    properties
        n = 0
    end
    methods
        function o = Kid(n), if nargin, o.n = n; end, end
    end
end
@file Lv.m
classdef Lv < uint8
    enumeration
        Lo (1), Hi (2)
    end
end
@main
o = Cfg2(); o.a = 5; o.kids = [Kid(1), Kid(2)]; e = Lv.Hi; k = Kid(3);
save('m1.mat', 'o', 'e', 'k'); clear o e k; S = load('m1.mat');
assert(S.o.a == 5 && S.o.twice == 10 && S.o.s == "txt" && isequal(S.o.c, {1, 'two', [3 4]}) && S.o.st.z == 3 && isequal(S.o.st.k, {1, 2}));
assert(S.o.fh(1) == 2 && numel(S.o.kids) == 2 && S.o.kids(2).n == 2 && S.o.loadedHere); assert(S.e == Lv.Hi && S.k.n == 3 && isa(S.k, 'Kid'));
load('m1.mat'); assert(isa(o, 'Cfg2') && isa(o, 'handle') && isvalid(o));

### cls/inputparser_more
function r = area2(varargin)
end
p = inputParser; p.FunctionName = 'area2'; p.addRequired('w', @(x) isnumeric(x) && x > 0); p.addOptional('h', 1, @isnumeric);
p.addParameter('Units', 'm', @(s) any(strcmp(s, {'m', 'cm'}))); p.addParameter('Verbose', false, @islogical); p.KeepUnmatched = true;
p.parse(3, 4, 'Units', 'cm', 'Extra', 1, 'Verbose', true); r = p.Results; assert(r.w == 3 && r.h == 4 && strcmp(r.Units, 'cm') && r.Verbose);
assert(isequal(fieldnames(r), {'h'; 'Units'; 'Verbose'; 'w'})); assert(isequal(p.Unmatched, struct('Extra', 1))); assert(isempty(p.UsingDefaults));
p.parse(3, 'Units', 'cm'); assert(p.Results.h == 1 && isequal(p.UsingDefaults, {'h', 'Verbose'}));
t = false; try, p.parse(-1); catch e, t = contains(e.message, 'w') && contains(e.message, 'area2'); end; assert(t);
t = false; try, p.parse(); catch e, t = true; end; assert(t); t = false; try, p.parse(1, 2, 'Units', 'km'); catch e, t = contains(e.message, 'Units'); end; assert(t);
p.parse(2, 'units', 'cm'); assert(strcmp(p.Results.Units, 'cm')); p.parse(2, 'Un', 'cm'); assert(strcmp(p.Results.Units, 'cm'));
q = inputParser; q.addParameter('A', 1); q.addParameter('AB', 2); q.parse('A', 5); assert(q.Results.A == 5 && q.Results.AB == 2);
q.parse(struct('A', 7, 'AB', 8)); assert(q.Results.A == 7 && q.Results.AB == 8); assert(isequal(q.Parameters, {'A', 'AB'}));
c = createCopy(q); c.addParameter('C', 3); assert(numel(c.Parameters) == 3 && numel(q.Parameters) == 2);
t = false; try, q.addParameter('A', 2); catch, t = true; end; assert(t);
"""


def _parse(text):
    cases, cur = [], None
    for raw in text.split("\n"):
        if raw.startswith("### "):
            cur = {"id": raw[4:].strip(), "files": {}, "main": [], "target": "main"}
            cases.append(cur)
        elif cur is None:
            continue
        elif raw.startswith("@file "):
            cur["target"] = raw[6:].strip()
            cur["files"][cur["target"]] = []
        elif raw.strip() == "@main":
            cur["target"] = "main"
        else:
            (cur["main"] if cur["target"] == "main" else cur["files"][cur["target"]]).append(raw)
    return cases


_ALL = _parse(CASES)


@pytest.mark.parametrize("case", _ALL, ids=[c["id"] for c in _ALL])
def test_case(case, tmp_path, monkeypatch):
    for rel, body in case["files"].items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(body).strip("\n") + "\n", encoding="utf-8")
    main = tmp_path / "main_script.m"
    main.write_text("\n".join(case["main"]).strip("\n") + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = []
    interp = Interpreter(out=out.append)
    interp.run_file(os.fspath(main))
