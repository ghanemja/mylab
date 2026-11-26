# ruff: noqa: E501  (the MATLAB programs below are kept as written, one statement group per line)
"""Indexing and operator overloads of classdef classes, classes derived from built-in types, and the indexing mixins.

Each case below is a small multi-file MATLAB program: the lines after ``@file name`` are written to that
file in a temporary folder and the lines after ``@main`` are run there as a script. A case passes when
the script reaches its end without an error; the expected values in the asserts were worked out by hand
from MATLAB's documented behaviour.
"""

import os

import pytest

from mylab import Interpreter

CASES = r"""
### idx/subsref_full
@file Wrap.m
classdef Wrap
    properties
        data = [10 20 30]
        tag = 'w'
    end
    methods
        function varargout = subsref(obj, s)
            switch s(1).type
                case '()'
                    r = obj.data(s(1).subs{:}) * 2;
                    if numel(s) > 1, r = subsref(r, s(2:end)); end
                    varargout{1} = r;
                case '{}'
                    varargout{1} = -obj.data(s(1).subs{:});
                otherwise
                    [varargout{1:nargout}] = builtin('subsref', obj, s);
            end
        end
        function obj = subsasgn(obj, s, v)
            if strcmp(s(1).type, '()')
                obj.data(s(1).subs{:}) = v + 1;
            else
                obj = builtin('subsasgn', obj, s, v);
            end
        end
        function n = total(obj, k)
            if nargin < 2, k = 1; end
            n = sum(obj.data) * k;
        end
        function r = inner(obj)
            r = obj.data(2);
        end
        function n = numel(~)
            n = 1;
        end
        function e = end(obj, k, n)
            e = numel(obj.data);
        end
    end
end
@main
w = Wrap(); assert(w(2) == 40); assert(isequal(w.data, [10 20 30])); w(1) = 5; assert(w.data(1) == 6);
assert(w{3} == -30); assert(w.total() == 56); assert(w.total(2) == 112); assert(total(w) == 56);
assert(w.inner() == 20); assert(w(end) == 60); assert(isequal(w(2:end), [40 60]));
w.tag = 'zz'; assert(strcmp(w.tag, 'zz')); w.data(2) = 7; assert(isequal(w.data, [6 7 30]));
assert(w.data(end) == 30); x = w(1:2); assert(isequal(x, [12 14]));
s.w = w; assert(s.w(2) == 14); c = {w}; assert(c{1}(3) == 60);
s.w(3) = 1; assert(s.w.data(3) == 2);
S = substruct('()', {2}); assert(subsref(w, S) == 14); assert(subsref([5 6 7], S) == 6);
q = subsasgn([1 2 3], substruct('()', {2}), 9); assert(isequal(q, [1 9 3]));
st = subsasgn(struct('a', 1), substruct('.', 'b', '()', {2}), 4); assert(isequal(st.b, [0 4]));

### idx/subsindex_colon
@file Ix.m
classdef Ix
    properties
        k = 1
    end
    methods
        function o = Ix(k), o.k = k; end
        function i = subsindex(o)
            i = o.k - 1;
        end
        function r = colon(a, b)
            r = a.k:b.k;
        end
    end
end
@main
v = [10 20 30 40]; assert(v(Ix(3)) == 30); r = Ix(2):Ix(4); assert(isequal(r, [2 3 4]));
c = {1, 2, 3}; assert(c{Ix(2)} == 2);

### idx/builtin_double
@file Meters.m
classdef Meters < double
    properties
        unit = 'm'
    end
    methods
        function o = Meters(v, u)
            if nargin == 0, v = []; end
            o = o@double(v);
            if nargin > 1, o.unit = u; end
        end
        function s = label(o)
            s = sprintf('%g %s', double(o), o.unit);
        end
        function r = plus(a, b)
            r = Meters(double(a) + double(b));
        end
    end
end
@file Plain.m
classdef Plain < uint8
    methods
        function o = Plain(v)
            o = o@uint8(v);
        end
    end
end
@file Name.m
classdef Name < char
end
@main
m = Meters(5); assert(strcmp(m.label(), '5 m')); assert(double(m) + 1 == 6); assert(isa(m, 'double'));
assert(isa(m, 'Meters') && isa(m, 'numeric') && isa(m, 'float') && ~isa(m, 'integer'));
assert(strcmp(class(m), 'Meters')); s = m + m; assert(isa(s, 'Meters') && double(s) == 10);
t = m * 2; assert(strcmp(class(t), 'double') && t == 10);
v = Meters([1 2 3], 'km'); assert(numel(v) == 3 && isequal(size(v), [1 3]));
e = v(2); assert(isa(e, 'Meters') && double(e) == 2 && strcmp(e.unit, 'km'));
v(2) = 9; assert(isequal(double(v), [1 9 3]) && isa(v, 'Meters'));
assert(sum(v) == 13 && strcmp(class(sum(v)), 'double'));
w = [v, v]; assert(isa(w, 'Meters') && numel(w) == 6);
r = reshape(v, 3, 1); assert(isa(r, 'Meters') && isequal(size(r), [3 1]));
assert(isa(v', 'Meters') && isequal(size(v'), [3 1]));
assert(isequal(v(end), Meters(3, 'km')) || double(v(end)) == 3);
p = Plain(200); assert(isa(p, 'uint8') && isa(p, 'integer')); q = p + 100; assert(q == 255 && isa(q, 'uint8'));
n = Name('abc'); assert(ischar(char(n)) && strcmp(char(n), 'abc') && numel(n) == 3); assert(strcmp(upper(n), 'ABC'));
assert(isnumeric(m) && isfloat(m) && ~isnumeric(n)); assert(isobject(m));
x = evalc('disp(Meters(7))'); assert(contains(x, '7'));

### idx/operators_all
@file N.m
classdef N
    properties
        v = 0
    end
    methods
        function o = N(v), if nargin, o.v = v; end, end
        function r = plus(a, b), r = N(val(a) + val(b)); end
        function r = minus(a, b), r = N(val(a) - val(b)); end
        function r = times(a, b), r = N(val(a) .* val(b)); end
        function r = mtimes(a, b), r = N(val(a) * val(b)); end
        function r = rdivide(a, b), r = N(val(a) ./ val(b)); end
        function r = ldivide(a, b), r = N(val(a) .\ val(b)); end
        function r = mrdivide(a, b), r = N(val(a) / val(b)); end
        function r = mldivide(a, b), r = N(val(a) \ val(b)); end
        function r = power(a, b), r = N(val(a) .^ val(b)); end
        function r = mpower(a, b), r = N(val(a) ^ val(b)); end
        function r = lt(a, b), r = val(a) < val(b); end
        function r = gt(a, b), r = val(a) > val(b); end
        function r = le(a, b), r = val(a) <= val(b); end
        function r = ge(a, b), r = val(a) >= val(b); end
        function r = ne(a, b), r = val(a) ~= val(b); end
        function r = eq(a, b), r = val(a) == val(b); end
        function r = and(a, b), r = val(a) & val(b); end
        function r = or(a, b), r = val(a) | val(b); end
        function r = not(a), r = ~val(a); end
        function r = uminus(a), r = N(-val(a)); end
        function r = uplus(a), r = N(abs(val(a))); end
        function r = ctranspose(a), r = N(val(a) + 100); end
        function r = transpose(a), r = N(val(a) + 200); end
        function r = horzcat(varargin), r = N(sum(cellfun(@val, varargin))); end
        function r = vertcat(varargin), r = N(prod(cellfun(@val, varargin))); end
        function r = colon(a, b), r = val(a):val(b); end
        function r = double(a), r = a.v; end
        function r = char(a), r = sprintf('N%d', a.v); end
        function r = string(a), r = "N" + a.v; end
        function r = isempty(a), r = a.v == 0; end
        function r = length(a), r = 77; end
        function r = size(a, varargin), r = [3 4]; end
        function r = numel(a), r = 12; end
        function r = abs(a), r = N(abs(a.v)); end
        function r = sum(a), r = a.v; end
    end
end
@file val.m
function x = val(a)
if isa(a, 'N'), x = a.v; else, x = a; end
end
@main
a = N(6); b = N(2);
assert((a + b).v == 8 && (a - b).v == 4 && (a .* b).v == 12 && (a * b).v == 12 && (a ./ b).v == 3);
assert((a .\ b).v == 2/6 && (a / b).v == 3 && (a \ b).v == 2/6 && (a .^ b).v == 36 && (a ^ b).v == 36);
assert(a > b && ~(a < b) && a >= b && ~(a <= b) && a ~= b && ~(a == b));
assert((a & b) && (a | N(0)) && ~(~a)); assert((-a).v == -6 && (+N(-3)).v == 3);
assert((a').v == 106 && (a.').v == 206); assert((2 + a).v == 8 && (a + 2).v == 8 && (10 - a).v == 4);
h = [a, b, N(1)]; assert(h.v == 9); g = [a; b]; assert(g.v == 12); assert(isequal(b:a, 2:6));
assert(double(a) == 6 && strcmp(char(a), 'N6') && string(a) == "N6");
assert(isempty(N(0)) && ~isempty(a) && length(a) == 77 && isequal(size(a), [3 4]) && numel(a) == 12);
assert(abs(N(-4)).v == 4 && sum(a) == 6); assert(plus(a, b).v == 8 && uminus(a).v == -6 && horzcat(a, b).v == 8);
z = builtin('horzcat', a, b); assert(isa(z, 'N') && builtin('numel', z) == 2);
c = cellfun(@double, {a, b}); assert(isequal(c, [6 2])); d = arrayfun(@(k) double(N(k)), 1:3); assert(isequal(d, 1:3));

### idx/inferior_classes
@file Lo.m
classdef Lo
    methods
        function r = plus(a, b), r = 'lo'; end
    end
end
@file Hi.m
classdef (InferiorClasses = {?Lo}) Hi
    methods
        function r = plus(a, b), r = 'hi'; end
    end
end
@main
assert(strcmp(Lo() + Hi(), 'hi')); assert(strcmp(Hi() + Lo(), 'hi')); assert(strcmp(Lo() + Lo(), 'lo')); assert(strcmp(Lo() + 1, 'lo'));
assert(strcmp(1 + Lo(), 'lo'));

### idx/disp_display
@file Dz.m
classdef Dz
    properties
        v = 3
    end
    methods
        function disp(o), fprintf('<Dz %d>\n', o.v); end
    end
end
@file Dy.m
classdef Dy
    methods
        function display(o), fprintf('custom display\n'); end
    end
end
@main
d = Dz(); s = evalc('d'); assert(contains(s, 'd =') && contains(s, '<Dz 3>'));
s = evalc('disp(d)'); assert(strcmp(strtrim(s), '<Dz 3>'));
y = Dy(); s = evalc('y'); assert(strcmp(strtrim(s), 'custom display'));
s = evalc('builtin(''disp'', d)'); assert(contains(s, 'v: 3'));

### idx/end_in_method
@file Stack.m
classdef Stack < handle
    properties
        items = {}
    end
    methods
        function push(o, x), o.items{end+1} = x; end
        function x = pop(o), x = o.items{end}; o.items(end) = []; end
        function x = peek(o), x = o.items{end}; end
        function n = count(o), n = numel(o.items); end
        function x = lastTwo(o), x = o.items(end-1:end); end
    end
end
@main
s = Stack(); s.push(1); s.push('two'); s.push(3); assert(s.count() == 3); assert(s.pop() == 3); assert(strcmp(s.peek(), 'two'));
c = s.lastTwo(); assert(isequal(c, {1, 'two'})); assert(strcmp(s.items{end}, 'two')); s.items{end+1} = 9; assert(s.count() == 3);

### rp/redefines_paren
@file Lab.m
classdef Lab < matlab.mixin.indexing.RedefinesParen
    properties (Access = private)
        Arr
    end
    properties
        Label = "x"
    end
    methods
        function obj = Lab(a)
            if nargin, obj.Arr = a; end
        end
        function out = value(obj)
            out = obj.Arr;
        end
        function varargout = size(obj, varargin)
            [varargout{1:nargout}] = size(obj.Arr, varargin{:});
        end
        function out = cat(dim, varargin)
            parts = cellfun(@(o) o.Arr, varargin, 'UniformOutput', false);
            out = Lab(cat(dim, parts{:}));
        end
        function out = sum(obj)
            out = sum(obj.Arr, 'all');
        end
    end
    methods (Access = protected)
        function varargout = parenReference(obj, indexOp)
            obj.Arr = obj.Arr.(indexOp(1));
            if isscalar(indexOp)
                varargout{1} = obj;
                return;
            end
            [varargout{1:nargout}] = obj.(indexOp(2:end));
        end
        function obj = parenAssign(obj, indexOp, varargin)
            if isscalar(indexOp)
                rhs = varargin{1};
                if isa(rhs, 'Lab'), rhs = rhs.Arr; end
                obj.Arr.(indexOp) = rhs;
                return;
            end
            [obj.(indexOp(2:end))] = varargin{:};
        end
        function n = parenListLength(obj, indexOp, ctx)
            n = 1;
        end
        function obj = parenDelete(obj, indexOp)
            obj.Arr.(indexOp) = [];
        end
    end
    methods (Static)
        function obj = empty()
            obj = Lab([]);
        end
    end
end
@main
a = Lab([10 20 30 40]); b = a(2:3); assert(isa(b, 'Lab') && isequal(b.value(), [20 30])); assert(isequal(size(a), [1 4]) && numel(a) == 4 && ~isempty(a));
assert(a(2).value() == 20); assert(a(end).value() == 40); assert(strcmp(a.Label, "x")); a.Label = "y"; assert(a.Label == "y");
a(1) = 99; assert(isequal(a.value(), [99 20 30 40])); a(2:3) = Lab([7 8]); assert(isequal(a.value(), [99 7 8 40]));
a(4) = []; assert(numel(a) == 3); assert(sum(a) == 114); assert(a(1:2).sum() == 106);
e = Lab.empty(); assert(isempty(e)); assert(length(a) == 3 && isrow(a));

### rp/redefines_dot_brace
@file Bag.m
classdef Bag < matlab.mixin.indexing.RedefinesDot & matlab.mixin.indexing.RedefinesBrace
    properties
        store = struct()
        items = {}
    end
    methods (Access = protected)
        function varargout = dotReference(obj, indexOp)
            [varargout{1:nargout}] = obj.store.(indexOp);
        end
        function obj = dotAssign(obj, indexOp, varargin)
            [obj.store.(indexOp)] = varargin{:};
        end
        function n = dotListLength(obj, indexOp, ctx)
            n = 1;
        end
        function varargout = braceReference(obj, indexOp)
            [varargout{1:nargout}] = obj.items.(indexOp);
        end
        function obj = braceAssign(obj, indexOp, varargin)
            [obj.items.(indexOp)] = varargin{:};
        end
        function n = braceListLength(obj, indexOp, ctx)
            n = listLength(obj.items, indexOp, ctx);
        end
    end
end
@main
b = Bag(); b.alpha = 5; b.beta = [1 2 3]; assert(b.alpha == 5); assert(b.beta(2) == 2); assert(isequal(fieldnames(b.store), {'alpha'; 'beta'}));
b{1} = 'one'; b{2} = 22; assert(strcmp(b{1}, 'one') && b{2} == 22); assert(isequal(b.items, {'one', 22}));
b.beta(3) = 9; assert(isequal(b.beta, [1 2 9]));

### idx/num_args_and_brace
@file Multi.m
classdef Multi
    properties
        c = {10, 20, 30}
    end
    methods
        function varargout = subsref(obj, s)
            switch s(1).type
                case '{}'
                    [varargout{1:nargout}] = obj.c{s(1).subs{:}};
                otherwise
                    [varargout{1:nargout}] = builtin('subsref', obj, s);
            end
        end
        function n = numArgumentsFromSubscript(obj, s, ctx)
            if strcmp(s(1).type, '{}')
                n = numel(obj.c(s(1).subs{:}));
            else
                n = 1;
            end
        end
    end
end
@file Hd.m
classdef Hd < handle
    properties
        data = [1 2 3]
        calls = 0
    end
    methods
        function varargout = subsref(o, s)
            o.calls = o.calls + 1;
            [varargout{1:nargout}] = builtin('subsref', o, s);
        end
        function o = subsasgn(o, s, v)
            if strcmp(s(1).type, '.') && strcmp(s(1).subs, 'data') && numel(s) == 1
                v = v * 10;
            end
            o = builtin('subsasgn', o, s, v);
        end
        function r = twice(o, k), r = o.data(k) * 2; end
        function reset(o), o.data = [0 0 0]; end
    end
end
@main
m = Multi(); [a, b] = m{1:2}; assert(a == 10 && b == 20); x = {m{:}}; assert(isequal(x, {10, 20, 30})); assert(m{3} == 30); v = [m{2:3}]; assert(isequal(v, [20 30]));
assert(isequal(m.c, {10, 20, 30}));
h = Hd(); assert(h.twice(2) == 4); assert(h.calls == 2); h.data = [1 1 1]; assert(isequal(h.data, [10 10 10])); h.data(2) = 5; assert(isequal(h.data, [10 5 10]));
h.reset(); assert(isequal(h.data, [0 0 0]));

### idx/builtin_subclass_more
@file Tagged.m
classdef Tagged < double
    properties
        tag = ''
    end
    methods
        function o = Tagged(v, t)
            if nargin == 0, v = 0; end
            o = o@double(v);
            if nargin > 1, o.tag = t; end
        end
        function r = times(a, b)
            r = Tagged(double(a) .* double(b), 'prod');
        end
        function disp(o)
            fprintf('Tagged[%s] %s\n', o.tag, mat2str(double(o)));
        end
    end
end
@file U8.m
classdef U8 < uint8
    methods
        function o = U8(v), o = o@uint8(v); end
    end
end
@main
t = Tagged([1 2; 3 4], 'm'); assert(isequal(size(t), [2 2])); assert(isequal(double(t(2, :)), [3 4])); assert(strcmp(t(2, :).tag, 'm'));
p = t .* 2; assert(isa(p, 'Tagged') && strcmp(p.tag, 'prod') && isequal(double(p), [2 4; 6 8]));
q = t + 1; assert(strcmp(class(q), 'double') && isequal(q, [2 3; 4 5])); assert(isequal(max(t(:)), 4) && strcmp(class(max(t(:))), 'double'));
s = evalc('disp(t)'); assert(strcmp(strtrim(s), 'Tagged[m] [1 2;3 4]')); t(1, 1) = 9; assert(double(t(1)) == 9 && isa(t, 'Tagged'));
c = [t; t]; assert(isa(c, 'Tagged') && isequal(size(c), [4 2])); assert(isequal(double(t'), [9 3; 2 4]));
assert(all(isfinite(t(:))) && ~any(isnan(t(:)))); z = zeros(1, 2); z(1) = Tagged(5); assert(z(1) == 5 && strcmp(class(z), 'double'));
u = U8([250 10]); assert(strcmp(class(u), 'U8') && strcmp(class(u + 10), 'uint8') && isequal(u + 10, uint8([255 20])));
assert(isequal(u(2), U8(10)) || uint8(u(2)) == 10); assert(isinteger(u) && isa(u, 'uint8')); f = find(Tagged([0 1 1])); assert(isequal(f, [2 3]));
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
