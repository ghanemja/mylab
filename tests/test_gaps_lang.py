"""onCleanup, inputParser, RandStream/rng, timer, message and small environment functions."""

import numpy as np
import pytest

from mylab.errors import MatlabError


# -- onCleanup ----------------------------------------------------------------------------------

def test_oncleanup_runs_when_function_returns(m):
    out = m.run("""
work();
disp('after')
function work()
    c = onCleanup(@() disp('cleanup'));
    disp('working')
end
""")
    assert out.split() == ["working", "cleanup", "after"]


def test_oncleanup_runs_on_error_and_clear(m):
    out = m.run("""
try
    fails();
catch err
    disp(err.message)
end
function fails()
    c = onCleanup(@() disp('closed'));
    error('boom');
end
""")
    assert out.split() == ["closed", "boom"]
    out = m.run("c = onCleanup(@() disp('bye')); disp('one'); clear c; disp('two')")
    assert out.split() == ["one", "bye", "two"]
    assert m.interp.cleanup_count == 0


def test_oncleanup_returned_object_is_kept(m):
    out = m.run("""
c = make();
disp('have it')
clear c
function c = make()
    c = onCleanup(@() disp('done'));
end
""")
    assert out.split() == ["have", "it", "done"]
    with pytest.raises(MatlabError, match="function handle"):
        m.run("onCleanup(5)")


# -- inputParser -------------------------------------------------------------------------------------

PARSE = """
function r = opts(varargin)
    p = inputParser;
    p.FunctionName = 'opts';
    addRequired(p, 'n', @isnumeric);
    addOptional(p, 'mode', 'fast', @ischar);
    addParameter(p, 'Tol', 1e-3, @(x) x > 0);
    p.addParamValue('Verbose', false);
    p.parse(varargin{:});
    r = p.Results;
    r.defaults = p.UsingDefaults;
end
"""


def test_inputparser_defaults_and_values(m):
    m.run("r = opts(5);" + PARSE)
    assert m.num("r.n") == 5 and m.str("r.mode") == "fast" and m.num("r.Tol") == 1e-3
    assert m.num("numel(r.defaults)") == 3
    m.run("r = opts(5, 'slow', 'tol', 0.5, 'Verbose', true);" + PARSE)
    assert m.str("r.mode") == "slow" and m.num("r.Tol") == 0.5 and m.num("r.Verbose")
    m.run("r = opts(5, 'Tol', 2);" + PARSE)  # a parameter name is not taken as the optional argument
    assert m.str("r.mode") == "fast" and m.num("r.Tol") == 2
    m.run("r = opts(1, struct('Tol', 7));" + PARSE)
    assert m.num("r.Tol") == 7
    assert list(m.val("r").fields)[:4] == ["mode", "n", "Tol", "Verbose"]


def test_inputparser_errors(m):
    with pytest.raises(MatlabError, match="'Tol' is invalid"):
        m.run("opts(5, 'Tol', -1);" + PARSE)
    with pytest.raises(MatlabError, match="not a recognized parameter"):
        m.run("opts(5, 'fast', 'Nope', 1);" + PARSE)
    with pytest.raises(MatlabError, match="required argument 'n'"):
        m.run("opts();" + PARSE)
    with pytest.raises(MatlabError, match="'n' is invalid"):
        m.run("opts('text');" + PARSE)
    with pytest.raises(MatlabError, match="No value was given"):
        m.run("opts(5, 'fast', 'Tol');" + PARSE)


def test_inputparser_keep_unmatched(m):
    m.run("""
p = inputParser; p.KeepUnmatched = true;
addParameter(p, 'A', 1);
parse(p, 'B', 2, 'a', 3);
u = p.Unmatched; res = p.Results;
""")
    assert m.num("u.B") == 2 and m.num("res.A") == 3
    assert "inputParser with properties" in m.run("p")


# -- random streams ------------------------------------------------------------------------------------

def test_randstream_global_and_rng_state(m):
    m.run("""
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 7));
a = rand(1, 3);
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 7));
b = rand(1, 3);
s = RandStream('mlfg6331_64', 'Seed', 1); c = rand(s, 2, 2); d = randn(s, 3, 1); k = randi(s, 10);
reset(s); c2 = rand(s, 2, 2);
st = rng; x = rand(1, 4); rng(st); y = rand(1, 4);
rng(42); u = rand; rng(42, 'twister'); v = rand;
g = RandStream.getGlobalStream;
""")
    np.testing.assert_array_equal(m.val("a"), m.val("b"))
    np.testing.assert_array_equal(m.val("c"), m.val("c2"))
    np.testing.assert_array_equal(m.val("x"), m.val("y"))
    assert m.num("u") == m.num("v")
    assert m.val("d").shape == (3, 1) and 1 <= m.num("k") <= 10
    assert m.str("class(g)") == "RandStream" and m.str("s.Type") == "mlfg6331_64"
    assert m.num("st.Seed") >= 0 and m.str("st.Type") == "twister"
    with pytest.raises(MatlabError, match="Unknown random number generator"):
        m.run("RandStream('nope')")
    m.run("rand('state', 5); a1 = rand; randn('seed', 5); a2 = rand; rng(sum(100*[1.5 2.25])); "
          "s2 = RandStream('mt19937ar', 'Seed', 12.7);")
    assert m.num("a1") == m.num("a2") and m.num("s2.Seed") == 12


# -- timer ---------------------------------------------------------------------------------------------

def test_timer_runs_callbacks_synchronously(m):
    out = m.run("""
t = timer('TimerFcn', @(src, evt) fprintf('tick %d\\n', src.TasksExecuted + 1), ...
          'StartFcn', @(~, ~) disp('start'), 'StopFcn', @(~, ~) disp('stop'), ...
          'ExecutionMode', 'fixedRate', 'Period', 0.01, 'TasksToExecute', 3);
start(t); wait(t);
n = t.TasksExecuted; r = t.Running;
delete(t); ok = isvalid(t);
""")
    assert out.split("\n")[:5] == ["start", "tick 1", "tick 2", "tick 3", "stop"]
    assert m.num("n") == 3 and m.str("r") == "off" and not m.num("ok")


def test_timer_single_shot_and_endless(m):
    m.run("""
global count; count = 0;
t = timer('TimerFcn', 'global count; count = count + 1;');
start(t);
single = count;
u = timer('ExecutionMode', 'fixedSpacing', 'TimerFcn', @bump);
start(u); wait(u);
function bump(src, ~)
    global count
    count = count + 1;
    if src.TasksExecuted >= 4, stop(src); end
end
""")
    assert m.num("single") == 1 and m.num("count") == 6
    with pytest.raises(MatlabError, match="needs a timer"):
        m.run("start(5)")


# -- message, getReport, small functions ------------------------------------------------------------------

def test_message_and_error_reports(m):
    m.run("msg = message('My:id', 3); s = getString(msg);")
    assert m.str("s") == "My:id (3)"
    m.run("try, error(message('Pkg:bad')); catch e, id = e.identifier; end")
    assert m.str("id") == "Pkg:bad"
    m.run("""
try
    inner();
catch err
    rep = getReport(err); rep2 = err.getReport('basic');
end
function inner()
    error('Deep:problem', 'value was %d', 4);
end
""")
    assert "inner" in m.str("rep") and "value was 4" in m.str("rep")
    assert m.str("rep2") == "value was 4"


def test_map_methods_in_function_form(m):
    m.run("mp = containers.Map(); mp('a') = 1; k = isKey(mp, 'a'); ks = keys(mp); vs = values(mp); "
          "remove(mp, 'a'); n = mp.Count;")
    assert m.num("k") and m.num("n") == 0 and m.str("ks{1}") == "a"
    with pytest.raises(MatlabError, match="containers.Map"):
        m.run("isKey(5, 'a')")


def test_harmless_environment_functions(m):
    m.run("uistack(1, 'top'); n = feature('numcores'); cs = feature('DefaultCharacterSet'); "
          "st = web('https://example.com');")
    assert m.num("n") >= 1 and m.str("cs") == "UTF-8" and m.num("st") == 0


def test_system_object_step(m):
    m.run("""
acc = Accum(); a = step(acc, 2); b = acc.step(3);
classdef Accum < matlab.System
    properties
        total = 0
    end
    methods
        function setupImpl(obj)
            obj.total = 10;
        end
        function y = stepImpl(obj, x)
            obj.total = obj.total + x;
            y = obj.total;
        end
    end
end
""")
    assert m.num("a") == 12 and m.num("b") == 15
