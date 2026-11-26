"""System functions: warning states, whos, profile, dbstack, functions, memoize, names, futures."""

import os

import numpy as np
import pytest

from mylab.errors import MatlabError


def test_warning_state_is_returned_and_restored(m):
    out = m.run("w = warning('off', 'all'); warning('hidden'); warning(w); warning('My:id', 'shown %d', 5);")
    assert "hidden" not in out and "Warning: shown 5" in out
    assert m.str("w(1).identifier") == "all" and m.str("w(1).state") == "on"
    out = m.run("old = warning('off', 'My:id'); warning('My:id', 'quiet'); q = warning('query', 'My:id');"
                "warning(old); warning('My:id', 'loud');")
    assert "quiet" not in out and "loud" in out
    assert m.str("old.state") == "on" and m.str("q.state") == "off" and m.str("q.identifier") == "My:id"
    m.run("[msg, id] = lastwarn;")
    assert m.str("msg") == "loud" and m.str("id") == "My:id"
    m.run("s = warning; d = warning('query', 'MATLAB:singularMatrix');")
    assert m.num("isstruct(s) && isfield(s, 'state') && isfield(d, 'identifier')") and m.str("d.state") == "on"


def test_warning_on_one_while_all_are_off_and_error_state(m):
    out = m.run("warning('off', 'all'); warning('on', 'Keep:this'); warning('Keep:this', 'kept');"
                "warning('Other:one', 'dropped'); warning('on', 'all');")
    assert "kept" in out and "dropped" not in out
    m.run("warning('error', 'Bad:thing');")
    with pytest.raises(MatlabError, match="now an error") as e:
        m.run("warning('Bad:thing', 'now an error')")
    assert e.value.identifier == "Bad:thing"
    out = m.run("warning('on', 'Bad:thing'); warning('Bad:thing', 'a warning again'); warning('off', 'backtrace');")
    assert "Warning: a warning again" in out
    with pytest.raises(MatlabError, match="state"):
        m.run("warning(struct('identifier', 'a:b', 'state', 'maybe'))")


def test_whos_returns_a_struct_array(m):
    m.run("x = 5; s = whos('x'); name = 'four'; z = [1+2i 3]; t = whos; g = whos('-regexp', '^n');")
    assert m.str("s.class") == "double" and m.num("s.bytes") == 8 and m.num("isequal(s.size, [1 1])")
    assert m.str("strjoin({t.name}, ',')") == "name,s,x,z"
    assert m.num("t(1).bytes") == 8 and m.num("t(4).bytes") == 32 and m.num("t(4).complex")
    assert m.str("g.name") == "name"
    m.run("c = who; d = who('x*');")
    assert m.num("iscell(c) && any(strcmp(c, 'z'))") and m.str("d{1}") == "x"


def test_profile_measures_user_functions(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slowsum.m").write_text("function s = slowsum(n)\ns = 0;\nfor k = 1:n\n  s = s + helper(k);\nend\nend\n"
                                        "function y = helper(k)\ny = k * 2;\nend\n")
    m.run("profile on; a = slowsum(20); b = slowsum(5); profile off; p = profile('info'); st = profile('status');")
    assert m.num("a") == 420 and m.str("st.ProfilerStatus") == "off"
    m.run("names = {p.FunctionTable.FunctionName}; top = p.FunctionTable(strcmp(names, 'slowsum'));"
          "low = p.FunctionTable(strcmp(names, 'helper'));")
    assert m.num("top.NumCalls") == 2 and m.num("low.NumCalls") == 25
    assert m.num("top.TotalTime >= low.TotalTime && low.TotalTime > 0")       # the caller's time includes the callee
    assert m.num("size(top.ExecutedLines, 2)") == 3 and m.num("any(top.ExecutedLines(:, 1) == 4)")
    assert m.str("top.FileName").endswith("slowsum.m")
    out = m.run("profile viewer")
    assert "slowsum" in out and "Calls" in out
    m.run("profile clear; q = profile('info');")
    assert m.num("isempty(q.FunctionTable)")
    with pytest.raises(MatlabError, match="unknown option"):
        m.run("profile sideways")


def test_dbstack_names_lines_and_outputs(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outerf.m").write_text("function [s, k] = outerf()\n[s, k] = innerf();\nend\n"
                                       "function [s, k] = innerf()\nx = 1; %#ok\n[s, k] = dbstack;\nend\n")
    m.run("[s, k] = outerf(); t = dbstack;")
    assert m.str("s(1).name") == "innerf" and m.str("s(2).name") == "outerf"
    assert m.num("s(1).line") == 6 and m.num("s(2).line") == 2 and m.num("k") == 1
    assert m.str("s(1).file") == "outerf.m" and m.num("isempty(t) && isstruct(t)")


def test_functions_describes_handles(m):
    m.run("a = 3; f = functions(@(x) a * x + 1); g = functions(@sin); h = functions(str2func('cos'));")
    assert m.str("f.type") == "anonymous" and m.str("f.function") == "@(x)a*x+1"
    assert m.num("f.workspace{1}.a") == 3
    assert m.str("g.function") == "sin" and m.str("g.type") == "simple" and m.str("h.function") == "cos"
    with pytest.raises(MatlabError, match="function handle"):
        m.run("functions(5)")


def test_memoize_caches_by_input_and_output_count(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "counted.m").write_text("function [q, r] = counted(a, b)\nglobal CALLS\nCALLS = CALLS + 1;\n"
                                        "q = floor(a / b); r = mod(a, b);\nend\n")
    m.run("global CALLS; CALLS = 0; f = memoize(@counted); x = f(7, 2); y = f(7, 2); [q, r] = f(7, 2); [q2, r2] = f(7, 2);")
    assert m.num("x == 3 && y == 3 && q == 3 && r == 1 && q2 == 3 && r2 == 1") and m.num("CALLS") == 2
    m.run("s = stats(f);")
    assert m.num("s.Cache.TotalHits") == 2 and m.num("s.Cache.TotalMisses") == 2
    assert m.num("s.CacheHitRatePercent") == 50 and m.num("s.CacheOccupancyPercent") == 20
    m.run("g = memoize(@counted); same = isequal(f, g); clearCache(f); f(7, 2); t = stats(f);")
    assert m.num("same") and m.num("CALLS") == 3 and m.num("t.Cache.TotalMisses") == 1
    m.run("f.CacheSize = 2; f(1, 1); f(2, 1); f(3, 1); u = stats(f); f.Enabled = false; f(3, 1); f(3, 1);")
    assert m.num("numel(u.Cache.Inputs)") == 2 and m.num("CALLS") == 8
    m.run("sq = memoize(@(v) v.^2); a = sq([1 2 3]); b = cellfun(sq, {2, 3});")
    assert m.num("a").tolist() == [[1, 4, 9]] and m.num("b").tolist() == [[4, 9]]
    m.run("f.Enabled = true; clearAllMemoizedCaches; e = stats(f);")
    assert m.num("isempty(e.Cache.Inputs)")
    with pytest.raises(MatlabError, match="positive"):
        m.run("f.CacheSize = 0;")


def test_make_valid_name(m):
    m.run("a = matlab.lang.makeValidName('1 bad name'); [c, changed] = matlab.lang.makeValidName({'ok', 'for', "
          "'a-b', '9x', 'two words', 'x_1'});")
    assert m.str("a") == "x1BadName"
    assert m.str("strjoin(c, ',')") == "ok,xFor,a_b,x9x,twoWords,x_1"
    assert m.num("changed").tolist() == [[False, True, True, True, True, False]]
    assert m.str("matlab.lang.makeValidName('a-b', 'ReplacementStyle', 'delete')") == "ab"
    assert m.str("matlab.lang.makeValidName('a-b', 'ReplacementStyle', 'hex')") == "a0x2Db"
    assert m.str("matlab.lang.makeValidName('9x', 'Prefix', 'v_')") == "v_9x"
    m.run("s = matlab.lang.makeValidName([\"my var\" \"ok\"]); long = matlab.lang.makeValidName(repmat('a', 1, 80));")
    assert m.num("isstring(s) && s(1) == \"myVar\"") and m.num("numel(long)") == 63 == m.num("namelengthmax")
    with pytest.raises(MatlabError, match="ReplacementStyle"):
        m.run("matlab.lang.makeValidName('a', 'ReplacementStyle', 'other')")


def test_make_unique_strings(m):
    m.run("[u, changed] = matlab.lang.makeUniqueStrings({'a', 'a', 'b', 'a'}); v = matlab.lang.makeUniqueStrings('x', "
          "{'x', 'x_1'}); w = matlab.lang.makeUniqueStrings([\"v\" \"v\" \"w\"]);"
          "k = matlab.lang.makeUniqueStrings({'a', 'a', 'a'}, [false true true]);"
          "t = matlab.lang.makeUniqueStrings({'abcdef', 'abcdef'}, {}, 4);")
    assert m.str("strjoin(u, ',')") == "a,a_1,b,a_2" and m.num("changed").tolist() == [[False, True, False, True]]
    assert m.str("v") == "x_2" and m.num("isequal(w, [\"v\" \"v_1\" \"w\"])")
    assert m.str("strjoin(k, ',')") == "a,a_1,a_2" and m.str("strjoin(t, ',')") == "abcd,ab_1"


def test_threads_locks_and_computer(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "locker.m").write_text("function locker()\nmlock;\nend\n")
    m.run("n = maxNumCompThreads; old = maxNumCompThreads(1); one = maxNumCompThreads; maxNumCompThreads('automatic');"
          "back = maxNumCompThreads; locker(); a = mislocked('locker'); munlock('locker'); b = mislocked('locker');")
    assert m.num("n") == (os.cpu_count() or 1) and m.num("old == n && one == 1 && back == n")
    assert m.num("a && ~b && ~mislocked")
    m.run("arch = computer('arch'); [str, maxsize, endian] = computer;")
    assert m.str("arch") == m.str("lower(str)") or m.str("str") in ("PCWIN64",)
    assert m.num("maxsize") == 2 ** 48 - 1 and m.str("endian") in ("L", "B")
    with pytest.raises(MatlabError, match="inside a function"):
        m.run("mlock")


def test_rats(m):
    assert m.str("rats(pi)") == "    355/113   "
    assert m.str("rats(0.5)") == "      1/2     " and m.str("rats(3)") == "      3       "
    assert m.str("rats(-0.75)") == "     -3/4     "
    m.run("r = rats([0.5 0.25; 1/3 2]); w = rats(pi, 20);")
    assert m.num("size(r)").tolist() == [[2, 28]] and "1/3" in m.str("r(2, :)")
    assert "104348/33215" in m.str("w")       # more room allows a closer ratio


def test_inputname(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "namesof.m").write_text("function n = namesof(a, b, c)\nn = {inputname(1), inputname(2)};\n"
                                        "if nargin > 2, n{3} = inputname(3); end\nend\n")
    m.run("alpha = 1; beta = 2; n = namesof(alpha, beta + 1, beta); k = namesof(5, alpha);")
    assert m.str("strjoin(n, '|')") == "alpha||beta" and m.str("strjoin(k, '|')") == "|alpha"
    assert m.str("inputname(1)") == ""
    (tmp_path / "third.m").write_text("function n = third(a)\nn = inputname(3);\nend\n")
    with pytest.raises(MatlabError, match="argument number 3"):
        m.run("third(alpha)")


def test_release_and_version_checks(m):
    assert not m.num("isMATLABReleaseOlderThan('R2016a')") and m.num("isMATLABReleaseOlderThan('R2030a')")
    assert m.num("isMATLABReleaseOlderThan('R2023a', 'release', 1)")       # an update of the same release is newer
    assert not m.num("isMATLABReleaseOlderThan('R2023a', 'prerelease')")
    assert not m.num("verLessThan('matlab', '7.0')") and m.num("verLessThan('matlab', '99.0')")
    with pytest.raises(MatlabError, match="R2021b"):
        m.run("isMATLABReleaseOlderThan('2021b')")
    with pytest.raises(MatlabError, match="stage"):
        m.run("isMATLABReleaseOlderThan('R2021b', 'beta')")


def test_nargin_of_named_functions_and_mat2str(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "two_in.m").write_text("function [a, b, c] = two_in(x, y)\na = x; b = y; c = 0;\nend\n")
    (tmp_path / "var_in.m").write_text("function varargout = var_in(x, varargin)\nvarargout = {x};\nend\n")
    assert m.num("nargin('two_in')") == 2 and m.num("nargout('two_in')") == 3
    assert m.num("nargin('var_in')") == -2 and m.num("nargout('var_in')") == -1
    assert m.num("nargin(@two_in)") == 2 and m.num("nargin(@(a, b, c) a)") == 3
    assert m.str("mat2str([true false])") == "[true false]" and m.str("mat2str('abc')") == "'abc'"
    assert m.str("mat2str(['ab'; 'cd'])") == "['ab';'cd']" and m.str("mat2str('')") == "''"
    assert m.str("mat2str([0.1 pi], 3)") == "[0.1 3.14]" and m.str("mat2str(int8([1 2]), 'class')") == "int8([1 2])"
    assert m.str("mat2str([1+2i 3])") == "[1+2i 3+0i]" and m.str("mat2str(\"s\")") == '"s"'


def test_parfeval_futures(m):
    m.run("f = parfeval(@max, 1, [1 5 2]); v = fetchOutputs(f); g = parfeval(@size, 2, ones(3, 4));"
          "[r, c] = fetchOutputs(g); st = f.State; h = parfeval(backgroundPool, @(x) x + 1, 1, 4); w = fetchOutputs(h);")
    assert m.num("v") == 5 and m.num("r") == 3 and m.num("c") == 4 and m.str("st") == "finished" and m.num("w") == 5
    m.run("e = parfeval(@() error('my:id', 'went wrong'), 0); none = isempty(f.Error); has = ~isempty(e.Error);")
    assert m.num("none && has")
    with pytest.raises(MatlabError, match="went wrong") as err:
        m.run("fetchOutputs(e)")
    assert err.value.identifier == "my:id"
    out = m.run("p = parfeval(@() disp('printed inside'), 0); d = p.Diary;")
    assert "printed inside" not in out and "printed inside" in m.str("d")
    with pytest.raises(MatlabError, match="future"):
        m.run("fetchOutputs(5)")


def test_java_and_mex_fail_with_a_clear_message(m):
    for code in ("s = java.lang.String('abc');", "al = javaObject('java.util.ArrayList');", "javaaddpath('x.jar')",
                 "javaMethod('size', 1)"):
        with pytest.raises(MatlabError, match="no Java virtual machine") as e:
            m.run(code)
        assert e.value.identifier == "MATLAB:Java:NotAvailable"
    assert not m.num("isjava(5)") and not m.num("usejava('jvm')")
    with pytest.raises(MatlabError, match="MEX files are not supported"):
        m.run("mex('addone.c')")


def test_callable_objects_work_with_feval_and_arrayfun(m):
    m.run("sq = memoize(@(v) v^2); a = feval(sq, 4); b = arrayfun(sq, [1 2 3]);")
    assert m.num("a") == 16 and np.array_equal(m.num("b"), [[1, 4, 9]])
