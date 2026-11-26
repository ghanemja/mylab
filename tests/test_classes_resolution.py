# ruff: noqa: E501  (the MATLAB programs below are kept as written, one statement group per line)
"""Function resolution: persistent variables and clear, private folders, imports, and which definition of a name wins.

Each case below is a small multi-file MATLAB program: the lines after ``@file name`` are written to that
file in a temporary folder and the lines after ``@main`` are run there as a script. A case passes when
the script reaches its end without an error; the expected values in the asserts were worked out by hand
from MATLAB's documented behaviour.
"""

import os

import pytest

from mylab import Interpreter

CASES = r"""
### res/persist_clear
@file counter.m
function n = counter()
persistent k
if isempty(k), k = 0; end
k = k + 1; n = k;
end
@main
assert(counter() == 1); assert(counter() == 2); clear counter; assert(counter() == 1);
counter(); clear functions; assert(counter() == 1); counter(); clear all; assert(counter() == 1);
counter(); clear classes; assert(counter() == 1);
x = 3; counter(); clear x; assert(counter() == 3);

### res/private
@file private/hidden_helper.m
function r = hidden_helper()
r = 7 + sib();
end
@file private/sib.m
function r = sib()
r = 0;
end
@file uses_private.m
function r = uses_private()
f = @hidden_helper; g = @() hidden_helper() + 1;
r = hidden_helper() + 1 + f() - 7 + g() - 8;
end
@file give_handle.m
function f = give_handle()
f = @hidden_helper;
end
@file other/outsider.m
function t = outsider()
t = false; try, hidden_helper(); catch, t = true; end
end
@main
assert(uses_private() == 8); assert(hidden_helper() == 7);
addpath(fullfile(pwd, 'other')); assert(outsider());
f = give_handle(); assert(f() == 7);

### res/imports
@file +geo/numel.m
function a = numel(r)
a = -1;
end
@file +geo/circarea.m
function a = circarea(r)
a = pi * r^2;
end
@file imp.m
function r = imp()
import geo.*
r = numel([1 2 3]);
end
@file imp2.m
function r = imp2()
import geo.numel
r = numel([1 2 3]) + abs(numel(1) + 0);
end
@file noimp.m
function r = noimp()
r = numel([1 2 3]);
end
@file localwins.m
function r = localwins()
import geo.*
r = circarea(1);
end
function a = circarea(r)
a = 99;
end
@file explicitwins.m
function r = explicitwins()
import geo.circarea
r = circarea(1);
end
function a = circarea(r)
a = 99;
end
@main
assert(imp() == -1); assert(numel([1 2 3]) == 3); assert(noimp() == 3); assert(imp2() == 0);
assert(localwins() == 99); assert(abs(explicitwins() - pi) < 1e-12);
import geo.circarea
assert(abs(circarea(1) - pi) < 1e-12);
f = @circarea; assert(abs(f(1) - pi) < 1e-12);
g = @(r) circarea(r) * 2; assert(abs(g(1) - 2 * pi) < 1e-12);
L = import; assert(iscell(L) && any(strcmp(L, 'geo.circarea')));

### res/user_shadows_builtin
@file numel.m
function n = numel(x)
n = 42;
end
@file sub/fliplr.m
function y = fliplr(x)
y = 'mine';
end
@main
assert(numel([1 2 3]) == 42);
addpath(fullfile(pwd, 'sub')); assert(strcmp(fliplr([1 2]), 'mine'));

### res/precedence
@file @Wd/Wd.m
classdef Wd
    properties
        v = 1
    end
    methods
        function r = describe(o), r = 'method'; end
        function r = callLocal(o), r = helper2(o); end
    end
end
function r = helper2(o)
r = o.v + 100;
end
@file describe.m
function r = describe(x)
r = 'path';
end
@file nest.m
function r = nest()
x = 5;
    function y = inner()
        y = x + 1;
    end
r = inner();
end
@file locwin.m
function r = locwin()
w = Wd();
r = {describe(w), describe(1)};
end
@file locwin2.m
function r = locwin2()
w = Wd();
r = describe(w);
end
function r = describe(x)
r = 'local';
end
@file varwins.m
function r = varwins()
describe = [10 20 30];
r = describe(2);
end
@main
w = Wd(); assert(strcmp(describe(w), 'method')); assert(strcmp(describe(5), 'path')); assert(strcmp(w.describe(), 'method'));
r = locwin(); assert(isequal(r, {'method', 'path'})); assert(strcmp(locwin2(), 'local')); assert(varwins() == 20); assert(nest() == 6);
assert(w.callLocal() == 101);
f = @describe; assert(strcmp(f(w), 'method') && strcmp(f(1), 'path'));
assert(exist('describe') == 2); assert(any(strcmp(methods(w), 'describe')));

### res/import_class
@file +zoo/Cat.m
classdef Cat
    properties
        n = 9
    end
    methods (Static)
        function r = sound(), r = 'meow'; end
    end
end
@file +zoo/feed.m
function r = feed(x)
r = x + 1;
end
@file +zoo/+deep/inner.m
function r = inner()
r = 'deep';
end
@file useimp.m
function r = useimp()
import zoo.Cat
import zoo.deep.*
c = Cat(); r = {c.n, Cat.sound(), inner()};
end
@file noimp2.m
function t = noimp2()
t = false; try, Cat(); catch, t = true; end
end
@main
r = useimp(); assert(isequal(r, {9, 'meow', 'deep'})); assert(noimp2());
import zoo.*
c = Cat(); assert(c.n == 9); assert(feed(1) == 2); assert(strcmp(Cat.sound(), 'meow')); assert(isa(c, 'zoo.Cat'));
f = @zoo.feed; assert(f(2) == 3); g = @zoo.Cat.sound; assert(strcmp(g(), 'meow'));
clear import; t = false; try, feed(1); catch, t = true; end; assert(t);

### res/import_beats_builtin_in_expressions
@file +mine/abs.m
function r = abs(x)
r = 100;
end
@file +mine/sqrt.m
function r = sqrt(x)
r = -1;
end
@file wild.m
function r = wild()
import mine.*
x = -3; r = abs(x) + 1;
s = 0; for k = 1:5, s = s + sqrt(4); end
r = [r, s];
end
@file plain.m
function r = plain()
x = -3; r = abs(x) + 1;
s = 0; for k = 1:5, s = s + sqrt(4); end
r = [r, s];
end
@main
assert(isequal(wild(), [101 -5])); assert(isequal(plain(), [4 10])); assert(isequal(wild(), [101 -5]));

### res/private_shadows_builtin
@file private/numel.m
function n = numel(x)
n = -7;
end
@file usesit.m
function r = usesit()
v = [1 2 3]; r = numel(v) + 0;
t = 0; for k = 1:3, t = t + numel(v); end
r = [r t];
end
@main
assert(isequal(usesit(), [-7 -21])); assert(numel([1 2 3]) == -7);
addpath(pwd); cd(tempdir); assert(isequal(usesit(), [-7 -21]));

### res/clear_classes_rereads
@file mk.m
function mk(v)
fid = fopen('Vers.m', 'w');
fprintf(fid, 'classdef Vers\n    properties\n        v = %d\n    end\nend\n', v);
fclose(fid);
end
@main
mk(1); a = Vers(); assert(a.v == 1); pause(0.02); mk(22); clear classes; b = Vers(); assert(b.v == 22);

### res/handles_and_locals
@file hl.m
function r = hl()
f = @loc; g = @(x) loc(x) + 1; c = {@loc, @numel};
r = [f(1), g(1), c{1}(2), c{2}([1 2 3]), feval(@loc, 5), feval('numel', [1 2])];
    function y = nestedOne(x)
        y = x * 1000;
    end
h = @nestedOne; r(end + 1) = h(2);
end
function y = loc(x)
y = x * 10;
end
@main
assert(isequal(hl(), [10 11 20 3 50 2 2000]));
fs = localfunctions; assert(isempty(fs));
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
